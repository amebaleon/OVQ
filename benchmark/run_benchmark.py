from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any


TARGET_DEFAULT = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    import torch

    lowered = name.lower()
    if lowered in ("float16", "fp16", "half"):
        return torch.float16
    if lowered in ("bfloat16", "bf16"):
        return torch.bfloat16
    if lowered in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def cuda_memory_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def cleanup() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_tokenizer(config: dict[str, Any], method: dict[str, Any] | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = dtype_from_name(config.get("dtype", "float16"))
    load_kwargs: dict[str, Any] = {
        "token": token,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if method and method.get("kind") == "bnb_nf4":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("bitsandbytes NF4 requires a transformers build with BitsAndBytesConfig.") from exc
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=bool(method.get("double_quant", True)),
        )
    else:
        load_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return model, tokenizer


def first_device(model):
    return next(model.parameters()).device


def get_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise TypeError("Could not locate transformer layers for this model architecture.")


def resolve_layers(model, layer_spec: Any) -> list[int]:
    layers = get_layers(model)
    if layer_spec == "all":
        return list(range(len(layers)))
    return [int(x) for x in layer_spec]


def iter_target_linears(model, layer_indices: list[int], target_modules: tuple[str, ...]):
    import torch

    layers = get_layers(model)
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for name, module in layer.named_modules():
            leaf = name.split(".")[-1]
            if leaf in target_modules and isinstance(module, torch.nn.Linear):
                yield layer_idx, leaf, module


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def quantized_bits_for_tensor(
    numel: int,
    bits: int,
    block_size: int,
    overlap_ratio: float,
    kind: str,
    metadata_mode: str = "window",
    metadata_group_size: int = 256,
) -> tuple[int, int, int]:
    metadata_bits_per_record = 32
    if numel < block_size:
        return numel * 16, 0, 0
    usable = (numel // block_size) * block_size
    tail = numel - usable
    if kind == "vq":
        windows = usable // block_size
    else:
        stride = max(1, int(round(block_size * (1.0 - overlap_ratio))))
        starts = list(range(0, max(1, usable - block_size + 1), stride))
        if not starts or starts[-1] != usable - block_size:
            starts.append(usable - block_size)
        windows = len(starts)
    payload_bits = windows * block_size * bits
    if metadata_mode == "none":
        metadata_records = 0
    elif metadata_mode == "group":
        group_size = max(block_size, int(metadata_group_size))
        metadata_records = math.ceil(usable / group_size)
    else:
        metadata_records = windows
    total_bits = payload_bits + metadata_records * metadata_bits_per_record + tail * 16
    return int(total_bits), int(windows), int(metadata_records)


def estimate_method_budget(model, config: dict[str, Any], method: dict[str, Any], layer_bits: dict[int, int] | None) -> dict[str, Any]:
    total_params = count_params(model)
    target_modules = tuple(config.get("target_modules", TARGET_DEFAULT))
    layer_indices = resolve_layers(model, config.get("layers", "all"))
    block_size = int(config.get("block_size", 16))
    overlap_ratio = float(config.get("overlap_ratio", 0.5))
    metadata_mode = str(config.get("metadata_mode", "window"))
    metadata_group_size = int(config.get("metadata_group_size", 256))
    kind = method["kind"]
    if kind == "bnb_nf4":
        estimated_bpw = float(method.get("estimated_bpw", 4.5))
        total_bits = total_params * estimated_bpw
        return {
            "total_params": total_params,
            "target_params": total_params,
            "target_fraction": 1.0,
            "target_only_bpw_est": estimated_bpw,
            "total_model_bpw_est": estimated_bpw,
            "estimated_model_size_mb": total_bits / 8 / (1024**2),
            "quant_windows": "",
            "metadata_records": "",
            "metadata_mode": "external_bitsandbytes_nf4",
            "metadata_group_size": "",
            "metadata_note": "external_baseline_estimate:bitsandbytes_nf4;actual_storage_varies_by_library_and_double_quant",
        }
    target_params = 0
    target_bits = 0
    target_windows = 0
    target_metadata_records = 0
    for layer_idx, _, module in iter_target_linears(model, layer_indices, target_modules):
        numel = module.weight.numel()
        target_params += numel
        if kind == "reference":
            q_bits = numel * 16
            windows = 0
            metadata_records = 0
        else:
            bits = layer_bits[layer_idx] if layer_bits is not None else int(method["bits"])
            q_kind = "vq" if kind == "vq" else "ovq"
            method_metadata_mode = str(method.get("metadata_mode", metadata_mode))
            method_metadata_group_size = int(method.get("metadata_group_size", metadata_group_size))
            q_bits, windows, metadata_records = quantized_bits_for_tensor(
                numel,
                bits,
                block_size,
                overlap_ratio,
                q_kind,
                method_metadata_mode,
                method_metadata_group_size,
            )
        target_bits += q_bits
        target_windows += windows
        target_metadata_records += metadata_records
    non_target_params = total_params - target_params
    total_bits = target_bits + non_target_params * 16
    return {
        "total_params": total_params,
        "target_params": target_params,
        "target_fraction": target_params / max(total_params, 1),
        "target_only_bpw_est": target_bits / max(target_params, 1),
        "total_model_bpw_est": total_bits / max(total_params, 1),
        "estimated_model_size_mb": total_bits / 8 / (1024**2),
        "quant_windows": target_windows,
        "metadata_records": target_metadata_records,
        "metadata_mode": method.get("metadata_mode", metadata_mode),
        "metadata_group_size": method.get("metadata_group_size", metadata_group_size),
        "metadata_note": "metadata_estimate:minmax_fp16_pair_per_record;non_target_params_fp16;fake_dequant_quality_run",
    }


def vq_dequant_torch(weight, bits: int, block_size: int):
    import torch

    src = weight.detach().float().flatten()
    usable = (src.numel() // block_size) * block_size
    tail = src[usable:]
    if usable == 0:
        return weight
    x = src[:usable].view(-1, block_size)
    lo = x.min(dim=1, keepdim=True).values
    hi = x.max(dim=1, keepdim=True).values
    levels = (1 << bits) - 1
    scale = (hi - lo).clamp_min(1e-12)
    q = torch.round((x - lo) / scale * levels).clamp_(0, levels)
    out = q / levels * scale + lo
    out = torch.cat([out.flatten(), tail], dim=0) if tail.numel() else out.flatten()
    return out.view_as(weight).to(dtype=weight.dtype, device=weight.device)


def ovq_dequant_torch(weight, bits: int, block_size: int, overlap_ratio: float, weight_kind: str):
    import torch

    src = weight.detach().float().flatten()
    if src.numel() < block_size:
        return vq_dequant_torch(weight, bits, block_size)
    step = max(1, int(round(block_size * (1.0 - overlap_ratio))))
    levels = (1 << bits) - 1
    acc = torch.zeros_like(src)
    den = torch.zeros_like(src)
    if weight_kind == "uniform":
        win = torch.ones(block_size, device=src.device)
    else:
        pos = torch.arange(block_size, device=src.device).float()
        center = (block_size - 1) / 2.0
        win = (1.0 - (pos - center).abs() / (center + 1.0)).clamp_min(0.05)
        if weight_kind == "center_heavy":
            win = win * win
    starts = torch.arange(0, max(1, src.numel() - block_size + 1), step, device=src.device)
    if starts.numel() == 0 or starts[-1].item() != src.numel() - block_size:
        starts = torch.cat([starts, torch.tensor([src.numel() - block_size], device=src.device)])
    windows = src[starts[:, None] + torch.arange(block_size, device=src.device)[None, :]]
    lo = windows.min(dim=1, keepdim=True).values
    hi = windows.max(dim=1, keepdim=True).values
    scale = (hi - lo).clamp_min(1e-12)
    q = torch.round((windows - lo) / scale * levels).clamp_(0, levels)
    dec = q / levels * scale + lo
    weighted = dec * win.view(1, -1)
    offsets = starts[:, None] + torch.arange(block_size, device=src.device)[None, :]
    acc.scatter_add_(0, offsets.reshape(-1), weighted.reshape(-1))
    den.scatter_add_(0, offsets.reshape(-1), win.expand_as(dec).reshape(-1))
    out = acc / den.clamp_min(1e-12)
    untouched = den == 0
    if untouched.any():
        out[untouched] = src[untouched]
    return out.view_as(weight).to(dtype=weight.dtype, device=weight.device)


def layer_bits_for_method(model, config: dict[str, Any], method: dict[str, Any]) -> dict[int, int] | None:
    kind = method["kind"]
    if kind in {"reference", "bnb_nf4"}:
        return None
    layer_indices = resolve_layers(model, config.get("layers", "all"))
    if kind in ("vq", "ovq"):
        return {i: int(method["bits"]) for i in layer_indices}
    if kind == "mixed_ovq":
        ranking = [int(x) for x in config.get("sensitivity_ranking", [])]
        top_k = int(method["top_k"])
        protected = set(ranking[:top_k])
        base_bits = int(method["base_bits"])
        protected_bits = int(method["protected_bits"])
        return {i: protected_bits if i in protected else base_bits for i in layer_indices}
    raise ValueError(f"Unsupported method kind: {kind}")


def patch_model(model, config: dict[str, Any], method: dict[str, Any], layer_bits: dict[int, int] | None) -> int:
    if method["kind"] in {"reference", "bnb_nf4"}:
        return 0
    target_modules = tuple(config.get("target_modules", TARGET_DEFAULT))
    layer_indices = resolve_layers(model, config.get("layers", "all"))
    block_size = int(config.get("block_size", 16))
    overlap_ratio = float(config.get("overlap_ratio", 0.5))
    overlap_weight = str(config.get("overlap_weight", "uniform"))
    patched = 0
    for layer_idx, module_name, module in iter_target_linears(model, layer_indices, target_modules):
        bits = layer_bits[layer_idx]
        if method["kind"] == "vq":
            new_w = vq_dequant_torch(module.weight.data, bits, block_size)
        else:
            new_w = ovq_dequant_torch(module.weight.data, bits, block_size, overlap_ratio, overlap_weight)
        module.weight.data.copy_(new_w)
        patched += 1
    return patched


def collect_reference_outputs(model, tokenizer, prompts: list[str], max_len: int):
    import torch

    logits = []
    hidden_by_prompt = []
    device = first_device(model)
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            out = model(**inputs, output_hidden_states=True)
            logits.append(out.logits[:, -1, :].detach().float().cpu())
            hidden_by_prompt.append([h.detach().float().cpu() for h in out.hidden_states])
    return torch.cat(logits, dim=0), hidden_by_prompt


def distribution_metrics(fp_logits, q_logits) -> dict[str, float]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    p = F.softmax(fp_logits.float(), dim=-1).clamp_min(1e-12)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    kl = F.kl_div(log_q, p, reduction="batchmean").item()
    ce = -(p * log_q).sum(dim=-1).mean().item()
    first = (fp_logits.argmax(dim=-1) == q_logits.argmax(dim=-1)).float().mean().item()
    top5_a = torch.topk(fp_logits, 5, dim=-1).indices
    top5_b = torch.topk(q_logits, 5, dim=-1).indices
    overlaps = []
    for i in range(top5_a.shape[0]):
        overlaps.append(len(set(top5_a[i].tolist()) & set(top5_b[i].tolist())) / 5)
    return {
        "kl_vs_reference": kl,
        "cross_entropy_vs_reference": ce,
        "first_token_match": first,
        "top5_overlap": float(np.mean(overlaps)),
    }


def hidden_cosine_rows(method_name: str, fp_hidden, q_hidden) -> tuple[list[dict[str, Any]], float]:
    import numpy as np
    import torch.nn.functional as F

    rows: list[dict[str, Any]] = []
    max_layers = min(len(fp_hidden[0]), len(q_hidden[0]))
    final_cos = float("nan")
    for layer_idx in range(max_layers):
        values = []
        for fp_seq, q_seq in zip(fp_hidden, q_hidden):
            a = fp_seq[layer_idx].flatten()
            b = q_seq[layer_idx].flatten()
            values.append(F.cosine_similarity(a, b, dim=0).item())
        mean_cos = float(np.mean(values))
        if layer_idx == max_layers - 1:
            final_cos = mean_cos
        rows.append({"method": method_name, "layer": layer_idx, "hidden_cosine": mean_cos})
    return rows, final_cos


def ppl_eval_texts(config: dict[str, Any]) -> list[str]:
    if config.get("ppl_dataset"):
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("Dataset PPL requires the `datasets` package. Install requirements.txt again.") from exc
        spec = config["ppl_dataset"]
        name = spec.get("name", "wikitext")
        subset = spec.get("subset", "wikitext-2-raw-v1")
        split = spec.get("split", "test")
        text_column = spec.get("text_column", "text")
        max_samples = int(spec.get("max_samples", 32))
        dataset = load_dataset(name, subset, split=split)
        texts = []
        for row in dataset:
            text = str(row.get(text_column, "")).strip()
            if text:
                texts.append(text)
            if len(texts) >= max_samples:
                break
        if texts:
            return texts
    return list(config.get("ppl_texts", []))


def perplexity(model, tokenizer, config: dict[str, Any]) -> float:
    import torch

    texts = ppl_eval_texts(config)
    if not texts:
        return float("nan")
    max_length = int(config.get("ppl_max_length", 128))
    max_batches = int(config.get("ppl_max_batches", 8))
    device = first_device(model)
    losses = []
    seen = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=False)
            ids = enc.input_ids[0]
            for start in range(0, ids.numel(), max_length):
                chunk = ids[start : start + max_length]
                if chunk.numel() < 2:
                    continue
                batch = chunk.unsqueeze(0).to(device)
                out = model(batch, labels=batch)
                losses.append(out.loss.detach().float().cpu())
                seen += 1
                if seen >= max_batches:
                    break
            if seen >= max_batches:
                break
    if not losses:
        return float("nan")
    mean_loss = torch.stack(losses).mean().item()
    return float(math.exp(mean_loss))


def generate_samples(model, tokenizer, config: dict[str, Any], method_name: str) -> tuple[list[dict[str, Any]], float]:
    import torch

    prompts = list(config.get("generation_prompts", []))
    max_new = int(config.get("generation_max_new_tokens", 32))
    repeats = int(config.get("generation_repeats", 1))
    device = first_device(model)
    rows = []
    total_new_tokens = 0
    elapsed = 0.0
    with torch.no_grad():
        for prompt in prompts:
            if config.get("use_chat_template_for_generation", False) and getattr(tokenizer, "chat_template", None):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = prompt
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            for repeat in range(repeats):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                new_tokens = max(0, int(output.shape[-1] - inputs.input_ids.shape[-1]))
                total_new_tokens += new_tokens
                rows.append(
                    {
                        "method": method_name,
                        "repeat": repeat,
                        "prompt": prompt,
                        "formatted_prompt": formatted,
                        "new_tokens": new_tokens,
                        "text": tokenizer.decode(output[0], skip_special_tokens=True),
                    }
                )
    tok_s = total_new_tokens / elapsed if elapsed > 0 else float("nan")
    return rows, float(tok_s)


def optional_baseline_status(config: dict[str, Any]) -> list[dict[str, Any]]:
    requested = config.get("optional_baselines", {})
    rows = []
    for name, enabled in requested.items():
        if not enabled:
            rows.append({"baseline": name, "status": "disabled", "reason": "disabled in config"})
        elif name == "bitsandbytes_nf4" and any(method.get("kind") == "bnb_nf4" for method in config.get("methods", [])):
            rows.append({"baseline": name, "status": "configured", "reason": "bnb_nf4 method is present in methods"})
        else:
            rows.append(
                {
                    "baseline": name,
                    "status": "not_implemented_in_v1",
                    "reason": "v1 records OVQ/VQ quality gate first; add real external baseline adapter before using for claims",
                }
            )
    return rows


def method_runtime_label(method: dict[str, Any]) -> str:
    kind = method.get("kind")
    if kind == "bnb_nf4":
        return "bitsandbytes_nf4_runtime"
    if kind in {"vq", "ovq", "mixed_ovq"}:
        return "fake_dequant_runtime"
    return "normal_transformers_runtime"


def run(config_path: Path) -> None:
    import torch

    config = load_json(config_path)
    out_dir = (config_path.parent.parent / config.get("output_dir", "outputs/default")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(int(config.get("seed", 42)))

    manifest = {
        "benchmark_name": config.get("benchmark_name"),
        "config_path": str(config_path),
        "model_id": config.get("model_id"),
        "dtype": config.get("dtype"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device_name(),
        "cuda_total_gb": cuda_memory_gb(),
        "torch": torch.__version__,
        "runtime_label": "fake_dequant_runtime_for_vq_ovq_methods",
        "created_unix": time.time(),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    write_csv(out_dir / "baseline_status.csv", optional_baseline_status(config))

    print(f"[1/2] Loading reference model: {config['model_id']}", flush=True)
    ref_model, tokenizer = load_model_and_tokenizer(config)
    prompts = list(config.get("eval_prompts", []))
    fp_logits, fp_hidden = collect_reference_outputs(ref_model, tokenizer, prompts, int(config.get("max_sequence_length", 128)))
    ref_ppl = perplexity(ref_model, tokenizer, config)
    ref_generation, ref_tok_s = generate_samples(ref_model, tokenizer, config, "fp16_reference")
    ref_budget = estimate_method_budget(ref_model, config, {"name": "fp16_reference", "kind": "reference"}, None)
    del ref_model
    cleanup()

    summary_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    generation_rows = ref_generation
    summary_rows.append(
        {
            "method": "fp16_reference",
            "kind": "reference",
            "status": "ok",
            "patched_modules": 0,
            "final_hidden_cosine": 1.0,
            "kl_vs_reference": 0.0,
            "cross_entropy_vs_reference": "",
            "first_token_match": 1.0,
            "top5_overlap": 1.0,
            "ppl": ref_ppl,
            "tokens_per_second": ref_tok_s,
            "runtime_label": "normal_transformers_runtime",
            **ref_budget,
        }
    )

    for method in config.get("methods", []):
        if method["kind"] == "reference":
            continue
        print(f"[2/2] Running method: {method['name']}", flush=True)
        model, _ = load_model_and_tokenizer(config, method)
        layer_bits = layer_bits_for_method(model, config, method)
        budget = estimate_method_budget(model, config, method, layer_bits)
        patched = patch_model(model, config, method, layer_bits)
        q_logits, q_hidden = collect_reference_outputs(model, tokenizer, prompts, int(config.get("max_sequence_length", 128)))
        metrics = distribution_metrics(fp_logits, q_logits)
        rows, final_cos = hidden_cosine_rows(method["name"], fp_hidden, q_hidden)
        hidden_rows.extend(rows)
        method_ppl = perplexity(model, tokenizer, config)
        gen_rows, tok_s = generate_samples(model, tokenizer, config, method["name"])
        generation_rows.extend(gen_rows)
        summary_rows.append(
            {
                "method": method["name"],
                "kind": method["kind"],
                "status": "ok",
                "patched_modules": patched,
                "final_hidden_cosine": final_cos,
                "ppl": method_ppl,
                "tokens_per_second": tok_s,
                "runtime_label": method_runtime_label(method),
                "bits": method.get("bits", ""),
                "base_bits": method.get("base_bits", ""),
                "protected_bits": method.get("protected_bits", ""),
                "top_k": method.get("top_k", ""),
                **metrics,
                **budget,
            }
        )
        del model
        cleanup()

    hidden_rows.insert(0, {"method": "fp16_reference", "layer": "all", "hidden_cosine": 1.0})
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "layer_hidden_cosine.csv", hidden_rows)
    append_jsonl(out_dir / "generation_samples.jsonl", generation_rows)
    print(f"Done. Results: {out_dir}", flush=True)


def validate_config(config_path: Path) -> None:
    config = load_json(config_path)
    required = ["model_id", "methods", "eval_prompts", "generation_prompts", "ppl_texts"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    names = [method.get("name") for method in config["methods"]]
    if len(names) != len(set(names)):
        raise ValueError("Method names must be unique.")
    if not any(method.get("kind") == "reference" for method in config["methods"]):
        raise ValueError("Config must include one reference method.")
    supported = {"reference", "vq", "ovq", "mixed_ovq", "bnb_nf4"}
    for method in config["methods"]:
        kind = method.get("kind")
        if kind not in supported:
            raise ValueError(f"Unsupported method kind: {kind}")
        if kind in {"vq", "ovq"} and "bits" not in method:
            raise ValueError(f"{method.get('name')} requires bits.")
        if kind == "mixed_ovq":
            for key in ("base_bits", "protected_bits", "top_k"):
                if key not in method:
                    raise ValueError(f"{method.get('name')} requires {key}.")
    print(f"Config OK: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nexus-OVQ due diligence benchmark v1")
    parser.add_argument("--config", type=Path, required=True, help="Path to a benchmark JSON config.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the JSON config without loading a model.")
    args = parser.parse_args()
    if args.validate_only:
        validate_config(args.config.resolve())
        return
    run(args.config.resolve())


if __name__ == "__main__":
    main()
