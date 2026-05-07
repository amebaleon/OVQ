from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


MODULE_PATTERNS = {
    "q_proj": "model.layers.{layer}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{layer}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{layer}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{layer}.self_attn.o_proj.weight",
    "gate_proj": "model.layers.{layer}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{layer}.mlp.up_proj.weight",
    "down_proj": "model.layers.{layer}.mlp.down_proj.weight",
}


def read_safetensors_header(path: Path) -> Tuple[int, Dict]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def build_header_cache(data_dir: Path) -> Dict[Path, Tuple[int, Dict]]:
    cache = {}
    for shard_path in sorted(data_dir.glob("*.safetensors")):
        cache[shard_path] = read_safetensors_header(shard_path)
    if not cache:
        raise FileNotFoundError(f"No .safetensors files found in {data_dir}")
    return cache


def find_index_file(data_dir: Path) -> Optional[Path]:
    candidates = [data_dir / "safetensors.index.json", data_dir / "model.safetensors.index.json"]
    candidates.extend(sorted(data_dir.glob("*.safetensors.index.json")))
    return next((path for path in candidates if path.exists()), None)


def load_weight_map(data_dir: Path) -> Optional[Dict[str, str]]:
    index_path = find_index_file(data_dir)
    if index_path is None:
        return None
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a valid weight_map")
    return weight_map


def find_tensor(data_dir: Path, tensor_key: str, cache, weight_map):
    if weight_map is not None:
        shard_name = weight_map.get(tensor_key)
        if shard_name is None:
            raise KeyError(f"{tensor_key!r} not found in safetensors index")
        shard = data_dir / shard_name
        return shard, cache[shard][0], cache[shard][1][tensor_key]
    for shard, (header_len, header) in cache.items():
        if tensor_key in header:
            return shard, header_len, header[tensor_key]
    raise KeyError(f"Tensor not found: {tensor_key}")


def decode_float(raw: bytes, dtype: str, shape) -> np.ndarray:
    if dtype == "F32":
        arr = np.frombuffer(raw, dtype="<f4").copy()
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "BF16":
        bf16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        arr = (bf16 << 16).view(np.float32)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")
    return arr.reshape(tuple(shape)).astype(np.float32, copy=False)


def load_tensor(data_dir: Path, tensor_key: str, cache, weight_map) -> np.ndarray:
    shard, header_len, info = find_tensor(data_dir, tensor_key, cache, weight_map)
    begin, end = info["data_offsets"]
    with shard.open("rb") as f:
        f.seek(8 + header_len + begin)
        raw = f.read(end - begin)
    if len(raw) != end - begin:
        raise IOError(f"Short read for {tensor_key}")
    return decode_float(raw, info["dtype"], info["shape"])


def iter_weight_samples(data_dir: Path, layers: Iterable[int], max_eval_values: int):
    cache = build_header_cache(data_dir)
    weight_map = load_weight_map(data_dir)
    for layer in layers:
        for module, pattern in MODULE_PATTERNS.items():
            key = pattern.format(layer=layer)
            weight = load_tensor(data_dir, key, cache, weight_map)
            flat = weight.reshape(-1)
            usable = min(flat.size, max_eval_values)
            yield layer, module, key, flat[:usable].astype(np.float32, copy=True)


def quant_dequant_block(block: np.ndarray, bits: int) -> np.ndarray:
    qmax = (1 << bits) - 1
    mn = float(block.min())
    mx = float(block.max())
    if mx == mn:
        return np.full_like(block, mn, dtype=np.float32)
    scale = (mx - mn) / qmax
    q = np.clip(np.rint((block - mn) / scale), 0, qmax).astype(np.float32)
    return (mn + q * scale).astype(np.float32)


def quant_dequant_blocks(blocks: np.ndarray, bits: int) -> np.ndarray:
    qmax = float((1 << bits) - 1)
    mn = blocks.min(axis=1, keepdims=True)
    mx = blocks.max(axis=1, keepdims=True)
    scale = (mx - mn) / qmax
    same = scale == 0
    scale = np.where(same, 1.0, scale)
    q = np.clip(np.rint((blocks - mn) / scale), 0, qmax).astype(np.float32)
    restored = (mn + q * scale).astype(np.float32)
    if np.any(same):
        restored = np.where(same, mn, restored)
    return restored


def blend_weights(block: int, kind: str = "triangular") -> np.ndarray:
    if kind == "uniform":
        return np.ones(block, dtype=np.float32)
    center = (block - 1) * 0.5
    dist = np.abs(np.arange(block, dtype=np.float32) - center)
    tri = (center + 1.0 - dist).astype(np.float32)
    if kind == "center_heavy":
        return tri * tri
    return tri


def vq_dequant(flat: np.ndarray, bits: int, block: int = 16) -> np.ndarray:
    usable = flat.size - (flat.size % block)
    if usable == 0:
        return np.empty(0, dtype=np.float32)
    blocks = flat[:usable].reshape(-1, block)
    return quant_dequant_blocks(blocks, bits).reshape(-1)


def ovq_dequant(
    flat: np.ndarray,
    bits: int,
    block: int = 16,
    overlap_ratio: float = 0.5,
    weight_kind: str = "triangular",
    error_weighted: bool = False,
) -> tuple[np.ndarray, int, float]:
    usable = flat.size - (flat.size % block)
    flat = flat[:usable]
    stride = max(1, min(block, int(round(block * (1.0 - overlap_ratio)))))
    eff = 1.0 - stride / block
    accum = np.zeros_like(flat, dtype=np.float32)
    weight_sum = np.zeros_like(flat, dtype=np.float32)
    base_blend = blend_weights(block, weight_kind if stride != block else "uniform")
    starts = np.arange(0, usable - block + 1, stride, dtype=np.int64)
    if starts.size == 0 or starts[-1] != usable - block:
        starts = np.append(starts, usable - block)
    windows = np.lib.stride_tricks.sliding_window_view(flat, block)[starts]
    restored_blocks = quant_dequant_blocks(windows, bits)
    if error_weighted:
        err = np.mean((windows - restored_blocks) ** 2, axis=1).astype(np.float32)
        blends = base_blend[None, :] / (err[:, None] + 1e-12)
    else:
        blends = np.broadcast_to(base_blend, restored_blocks.shape)
    weighted = restored_blocks * blends
    offsets = starts[:, None] + np.arange(block, dtype=np.int64)[None, :]
    np.add.at(accum, offsets.reshape(-1), weighted.reshape(-1))
    np.add.at(weight_sum, offsets.reshape(-1), blends.reshape(-1))
    return (accum / np.maximum(weight_sum, 1e-8)).astype(np.float32), stride, eff


def group_local_ovq(flat: np.ndarray, bits: int, group_size: int = 64, block: int = 16) -> np.ndarray:
    usable = flat.size - (flat.size % group_size)
    flat = flat[:usable]
    out = np.empty_like(flat, dtype=np.float32)
    for base in range(0, usable, group_size):
        restored, _, _ = ovq_dequant(flat[base : base + group_size], bits, block=block, overlap_ratio=0.5)
        out[base : base + group_size] = restored
    return out


def clipped_ovq(flat: np.ndarray, bits: int, percentile: float = 99.9, block: int = 16) -> np.ndarray:
    limit = float(np.percentile(np.abs(flat), percentile))
    clipped = np.clip(flat, -limit, limit).astype(np.float32)
    restored, _, _ = ovq_dequant(clipped, bits, block=block, overlap_ratio=0.5)
    return restored


def outlier_protected_ovq(flat: np.ndarray, bits: int, percentile: float = 99.9, block: int = 16) -> tuple[np.ndarray, float]:
    usable = flat.size - (flat.size % block)
    source = flat[:usable]
    threshold = float(np.percentile(np.abs(source), percentile))
    mask = np.abs(source) >= threshold
    clipped = source.copy()
    clipped[mask] = 0.0
    restored, _, _ = ovq_dequant(clipped, bits, block=block, overlap_ratio=0.5)
    restored[mask] = source[mask]
    return restored, float(np.mean(mask))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def outlier_score(flat: np.ndarray, block: int = 16) -> float:
    usable = flat.size - (flat.size % block)
    blocks = flat[:usable].reshape(-1, block)
    ranges = blocks.max(axis=1) - blocks.min(axis=1)
    return float(np.mean(ranges) + np.std(ranges))


def hessian_proxy(flat: np.ndarray) -> float:
    sample = flat[: min(flat.size, 262144)]
    rough = np.diff(sample)
    return float(np.mean(sample * sample) + np.mean(rough * rough))


def time_call(fn, repeats: int = 3):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return result, best
