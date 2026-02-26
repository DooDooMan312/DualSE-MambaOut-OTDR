
"""
End-to-end benchmark (txt -> Recurrence Plot -> model inference) with detailed timing logs.

Why this script:
- Your current pipeline generates RP images and saves them to disk; saving dominates latency.
- Reviewer asks whether the reported throughput includes RP generation. This script measures:
    (1) model-only inference speed
    (2) end-to-end speed including RP generation (with/without disk saving)

Outputs (saved to --out_dir):
- e2e_timing_per_sample.csv      : per-sample breakdown
- e2e_timing_summary.csv         : group (per class + ALL) mean/median/p95 + throughput
- e2e_meta.json                  : hardware + software + benchmark params

Typical usage:
1) Model-only inference (network forward only):
    python end2end_rp_infer_benchmark.py --bench model_only --model mambaout --batch_size 64 --iters 2000

2) End-to-end compute-only (txt -> RP -> tensor -> inference), NO saving:
    python end2end_rp_infer_benchmark.py --bench e2e --model mambaout --N 1024 --eps 0.05 --steps 3 --block 512 \
        --dir_wind data/0wind --dir_manual data/1manual --dir_digger data/2digger \
        --batch_size 8 --max_per_class 200 --transform minimal

3) End-to-end including save_png (debug/logging overhead):
    python end2end_rp_infer_benchmark.py --bench e2e --save_png --save_dir rp_debug_pngs ...

Notes:
- For 'dse_mambaout' (dual-input model), you can choose:
    --transform minimal      -> feed the same view to both inputs (fastest)
    --transform weak_strong  -> weak/strong views like your training pipeline (slower, includes augmentation)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional: faster resize
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from PIL import Image

import torch
from torchvision import transforms


# ----------------------------
# Timing helpers
# ----------------------------

def _now() -> float:
    return time.perf_counter()

def _is_cuda(device: torch.device) -> bool:
    return (device.type == "cuda") and torch.cuda.is_available()

def _sync(device: torch.device) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(device)

def _time_gpu_ms(device: torch.device, fn):
    """
    Measure GPU time of fn() using CUDA events (ms).
    Falls back to CPU wall time if not CUDA.
    """
    if not _is_cuda(device):
        t0 = _now()
        out = fn()
        t1 = _now()
        return out, (t1 - t0) * 1000.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    _sync(device)
    start.record()
    out = fn()
    end.record()
    _sync(device)
    return out, start.elapsed_time(end)


# ----------------------------
# RP generator (optimized)
# ----------------------------

@dataclass
class RPTiming:
    load_txt_s: float = 0.0
    preprocess_s: float = 0.0
    rp_compute_s: float = 0.0
    rp_map_s: float = 0.0
    resize224_s: float = 0.0

class RecurrencePlotGenerator:
    """
    Generate recurrence plot image (uint8, HxW) from a txt file, with timing breakdown.
    Core logic adapted from your txt2RP.py (chunked RP + fast standardization),
    with an additional LUT mapping to avoid float conversion during rp->grayscale.
    """

    def __init__(
        self,
        eps: float = 0.05,
        steps: int = 3,
        N: Optional[int] = 1024,
        downsample: int = 1,
        block: int = 512,
        out_hw: int = 224,
        use_fast_std: bool = True,
    ):
        assert steps >= 1
        assert downsample >= 1
        assert block >= 1
        self.eps = float(eps)
        self.steps = int(steps)
        self.N = N if N is None else int(N)
        self.downsample = int(downsample)
        self.block = int(block)
        self.out_hw = int(out_hw)
        self.use_fast_std = bool(use_fast_std)

        # LUT: rp value (0..steps) -> grayscale 0..255 (binary-like)
        # 0 -> 255 (white), steps -> 0 (black)
        lut = np.round(255.0 * (1.0 - (np.arange(self.steps + 1, dtype=np.float32) / float(self.steps)))).astype(np.uint8)
        self._lut = lut  # shape (steps+1,)

        # Precompute 1/eps for speed
        self._inv_eps = np.float32(1.0 / self.eps)

    @staticmethod
    def _standardize_1d(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        mu = float(x.mean())
        sd = float(x.std()) + 1e-8
        return (x - mu) / sd

    def _recurrence_plot_1d_chunked(self, signal_1d: np.ndarray) -> np.ndarray:
        """
        RP definition (same as your code):
            R[i,j] = floor(|x_i - x_j| / eps), clipped to [0, steps]
        Implemented with block computation to avoid NxN float peak memory.
        Output dtype: uint8, values in [0, steps].
        """
        x = np.asarray(signal_1d, dtype=np.float32)
        n = x.shape[0]
        rp = np.empty((n, n), dtype=np.uint8)

        x_row = x[None, :]  # (1, n)
        inv_eps = self._inv_eps

        for i0 in range(0, n, self.block):
            i1 = min(i0 + self.block, n)

            # diff: (block, n)
            diff = np.abs(x[i0:i1, None] - x_row)

            # Since diff >= 0, astype(int) truncates == floor for positive values
            q = (diff * inv_eps).astype(np.int16, copy=False)
            q[q > self.steps] = self.steps
            rp[i0:i1, :] = q.astype(np.uint8, copy=False)

        return rp

    def _resize_to_hw(self, img_u8: np.ndarray) -> np.ndarray:
        if img_u8.shape[0] == self.out_hw and img_u8.shape[1] == self.out_hw:
            return img_u8
        if cv2 is not None:
            return cv2.resize(img_u8, (self.out_hw, self.out_hw), interpolation=cv2.INTER_AREA)
        # PIL fallback
        return np.array(Image.fromarray(img_u8, mode="L").resize((self.out_hw, self.out_hw), resample=Image.BILINEAR))

    def txt_to_rp_image224(self, txt_path: str) -> Tuple[np.ndarray, Dict[str, float], Dict[str, int | float]]:
        """
        Return:
          - img224_u8: (224,224) uint8
          - timing dict (seconds)
          - meta dict (N_used, downsample, eps, steps, block)
        """
        tt = RPTiming()
        t0 = _now()
        raw = np.loadtxt(txt_path)
        t1 = _now()
        tt.load_txt_s = t1 - t0

        # preprocess: reshape -> downsample -> crop N -> standardize
        p0 = _now()
        x = raw.reshape(-1)
        if self.downsample > 1:
            x = x[:: self.downsample]
        if (self.N is not None) and (x.shape[0] > self.N):
            x = x[: self.N]
        if self.use_fast_std:
            signal = self._standardize_1d(x)
        else:
            # slower sklearn StandardScaler avoided by default
            mu = float(x.mean())
            sd = float(x.std()) + 1e-8
            signal = ((x.astype(np.float32) - mu) / sd).astype(np.float32, copy=False)
        p1 = _now()
        tt.preprocess_s = p1 - p0

        # RP compute
        r0 = _now()
        rp = self._recurrence_plot_1d_chunked(signal)
        r1 = _now()
        tt.rp_compute_s = r1 - r0

        # RP -> grayscale via LUT (fast)
        m0 = _now()
        rp_clip = np.minimum(rp, self.steps)
        img_u8 = self._lut[rp_clip]  # (N,N) uint8
        m1 = _now()
        tt.rp_map_s = m1 - m0

        # Resize to 224
        z0 = _now()
        img224 = self._resize_to_hw(img_u8)
        z1 = _now()
        tt.resize224_s = z1 - z0

        timing = asdict(tt)
        meta = {
            "N_used": int(signal.shape[0]),
            "downsample": int(self.downsample),
            "eps": float(self.eps),
            "steps": int(self.steps),
            "block": int(self.block),
        }
        return img224, timing, meta


# ----------------------------
# Model building (3 variants)
# ----------------------------

def build_model(model_kind: str, device: torch.device, weights_path: Optional[str] = None):
    """
    model_kind:
      - mambaout           : from mambaout_withoutAttention.py (single input)
      - se_mambaout        : from mambaout.py (single input)
      - dse_mambaout       : from mambaout_weak_strong.py (dual input)
    Uses the same depths/dims as your my_mamba_321.py.
    """
    model_kind = model_kind.lower()

    if model_kind == "mambaout":
        from mambaout_withoutAttention import MambaOut as Net
        is_dual = False
    elif model_kind == "se_mambaout":
        from mambaout import MambaOut as Net
        is_dual = False
    elif model_kind == "dse_mambaout":
        from mambaout_weak_strong import MambaOut as Net
        is_dual = True
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    model = Net(
        depths=[3, 3, 9, 3],
        dims=[48, 64, 192, 288],
    ).to(device)

    if weights_path:
        sd = torch.load(weights_path, map_location=device)
        model.load_state_dict(sd, strict=True)

    model.eval()
    return model, is_dual


# ----------------------------
# Transforms (minimal vs weak/strong)
# ----------------------------

def make_transform(transform_mode: str, is_dual: bool):
    """
    transform_mode:
      - minimal: no PIL, no augmentation. Equivalent to ToTensor for a grayscale uint8 image.
      - weak_strong: match your data_weak_strong.py weak/strong idea (includes augmentation)

    Returns:
      - if is_dual=False: fn(img224_u8) -> x [1,224,224] float32
      - if is_dual=True : fn(img224_u8) -> (x1, x2)
    """
    transform_mode = transform_mode.lower()
    if transform_mode not in {"minimal", "weak_strong"}:
        raise ValueError("transform_mode must be one of: minimal, weak_strong")

    if transform_mode == "minimal":
        def to_tensor_minimal(img_u8: np.ndarray) -> torch.Tensor:
            # img_u8: (224,224) uint8
            # output: (1,224,224) float32 in [0,1]
            if img_u8.ndim != 2:
                raise ValueError(f"Expected grayscale (H,W), got shape {img_u8.shape}")
            t = torch.from_numpy(img_u8).unsqueeze(0).float().div(255.0)
            return t

        if is_dual:
            return lambda img: (to_tensor_minimal(img), to_tensor_minimal(img))
        return lambda img: to_tensor_minimal(img)

    # weak_strong
    weak_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    strong_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    if is_dual:
        return lambda img: (weak_transform(img), strong_transform(img))
    return lambda img: weak_transform(img)


# ----------------------------
# CSV summarization
# ----------------------------

def summarize(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
    arr = np.array([r[key] for r in rows], dtype=np.float64)
    if arr.size == 0:
        return {}
    mean_s = float(arr.mean())
    return {
        f"{key}_mean_ms": float(mean_s * 1e3),
        f"{key}_median_ms": float(np.median(arr) * 1e3),
        f"{key}_p95_ms": float(np.percentile(arr, 95) * 1e3),
        f"{key}_throughput_sps": float(1.0 / mean_s) if mean_s > 0 else float("inf"),
    }


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        raise ValueError("No rows to write.")
    keys = []
    # union keys, stable ordering: keep common first then rest sorted
    keyset = set()
    for r in rows:
        keyset.update(r.keys())
    # prefer some keys first
    preferred = [
        "rec", "category", "label", "pred",
        "N_used", "downsample", "eps", "steps", "block",
        "load_txt_s", "preprocess_s", "rp_compute_s", "rp_map_s", "resize224_s",
        "transform_s", "h2d_s", "infer_s",
        "total_compute_s", "total_e2e_s",
        "save_png_s", "total_with_save_s",
        "batch_size", "model_kind", "transform_mode", "device",
    ]
    for k in preferred:
        if k in keyset:
            keys.append(k)
            keyset.remove(k)
    keys += sorted(keyset)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_summary(path: str, rows: List[Dict], group_key: str = "category"):
    summary_rows: List[Dict] = []

    def add_group(name: str, rs: List[Dict]):
        s = {"group": name, "count": int(len(rs))}
        for k in [
            "load_txt_s", "preprocess_s", "rp_compute_s", "rp_map_s", "resize224_s",
            "transform_s", "h2d_s", "infer_s",
            "total_compute_s", "total_e2e_s",
        ]:
            if rs and (k in rs[0]):
                s.update(summarize(rs, k))
        if rs and ("total_e2e_s" in rs[0]):
            mean_total = np.mean([r["total_e2e_s"] for r in rs])
            s["end2end_samples_per_s"] = float(1.0 / mean_total) if mean_total > 0 else float("inf")
        summary_rows.append(s)

    # groups
    groups = {}
    for r in rows:
        g = str(r.get(group_key, "ALL"))
        groups.setdefault(g, []).append(r)

    for gname in sorted(groups.keys()):
        add_group(gname, groups[gname])
    add_group("ALL", rows)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = set()
    for r in summary_rows:
        keys.update(r.keys())
    keys = ["group", "count"] + sorted([k for k in keys if k not in {"group", "count"}])

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary_rows)


# ----------------------------
# Dataset scanning
# ----------------------------

def list_txt_files(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".txt")]
    files.sort()
    return [os.path.join(dir_path, f) for f in files]

def build_file_list(
    dir_wind: str,
    dir_manual: str,
    dir_digger: str,
    split: Optional[str] = None,
    max_per_class: Optional[int] = None,
) -> List[Dict]:
    """
    Returns list of sample dict:
      {path, rec, category, label}
    label mapping follows your convention: wind=0, manual=1, digger=2.
    """
    mapping = [
        ("wind", 0, dir_wind),
        ("manual", 1, dir_manual),
        ("digger", 2, dir_digger),
    ]

    samples = []
    for cat, lab, base in mapping:
        use_dir = os.path.join(base, split) if (split and os.path.isdir(os.path.join(base, split))) else base
        paths = list_txt_files(use_dir)
        if max_per_class is not None:
            paths = paths[: int(max_per_class)]
        for p in paths:
            rec = os.path.splitext(os.path.basename(p))[0]
            samples.append({"path": p, "rec": rec, "category": cat, "label": lab})
    return samples


# ----------------------------
# Benchmarks
# ----------------------------

@torch.no_grad()
def bench_model_only(model, is_dual: bool, device: torch.device, batch_size: int, iters: int, warmup: int, amp: bool):
    """
    Model-only forward benchmark:
      - no RP generation
      - no file I/O
      - random input(s)
    """
    x = torch.randn(batch_size, 1, 224, 224, device=device)
    if is_dual:
        x2 = torch.randn(batch_size, 1, 224, 224, device=device)

    # warmup
    for _ in range(warmup):
        if amp and _is_cuda(device):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x, x2) if is_dual else model(x)
        else:
            _ = model(x, x2) if is_dual else model(x)
    _sync(device)

    # timing
    t0 = _now()
    for _ in range(iters):
        if amp and _is_cuda(device):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(x, x2) if is_dual else model(x)
        else:
            _ = model(x, x2) if is_dual else model(x)
    _sync(device)
    t1 = _now()

    total_s = t1 - t0
    total_samples = iters * batch_size
    sps = total_samples / total_s if total_s > 0 else float("inf")
    ms_per_sample = (total_s / total_samples) * 1000.0 if total_samples > 0 else float("inf")
    return {
        "iters": iters,
        "batch_size": batch_size,
        "total_s": total_s,
        "samples_per_s": sps,
        "ms_per_sample": ms_per_sample,
        "amp": amp,
        "device": str(device),
        "is_dual": is_dual,
    }


@torch.no_grad()
def bench_end2end(
    model,
    is_dual: bool,
    device: torch.device,
    rpgen: RecurrencePlotGenerator,
    transform_fn,
    samples: List[Dict],
    batch_size: int,
    warmup: int,
    amp: bool,
    save_png: bool,
    save_dir: Optional[str],
    out_dir: str,
):
    """
    End-to-end benchmark:
      txt -> preprocess/std -> RP -> grayscale -> resize224 -> transform -> H2D -> inference
    Records per-sample timing breakdown + summary.
    """
    if save_png and not save_dir:
        save_dir = os.path.join(out_dir, "saved_png")
    if save_png and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # warmup
    for i in range(min(warmup, len(samples))):
        img224, _, _ = rpgen.txt_to_rp_image224(samples[i]["path"])
        _ = transform_fn(img224)
        if is_dual:
            x1, x2 = _
            x1 = x1.unsqueeze(0).to(device)
            x2 = x2.unsqueeze(0).to(device)
            if amp and _is_cuda(device):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x1, x2)
            else:
                _ = model(x1, x2)
        else:
            x = _.unsqueeze(0).to(device)
            if amp and _is_cuda(device):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)
        _sync(device)

    rows: List[Dict] = []
    wall0 = _now()

    # process in batches
    for b0 in range(0, len(samples), batch_size):
        batch = samples[b0:b0 + batch_size]
        cpu_inputs_1: List[torch.Tensor] = []
        cpu_inputs_2: List[torch.Tensor] = []

        # store partial rows to fill infer/h2d later
        batch_rows: List[Dict] = []

        # (A) RP + transform (CPU)
        for s in batch:
            img224, rp_timing, rp_meta = rpgen.txt_to_rp_image224(s["path"])

            # transform timing
            t_tf0 = _now()
            tf_out = transform_fn(img224)
            t_tf1 = _now()

            if is_dual:
                x1, x2 = tf_out
                cpu_inputs_1.append(x1)
                cpu_inputs_2.append(x2)
            else:
                cpu_inputs_1.append(tf_out)

            # optional save png timing (DEBUG/logging)
            save_png_s = 0.0
            if save_png and save_dir:
                sp0 = _now()
                Image.fromarray(img224, mode="L").save(os.path.join(save_dir, f"{s['category']}_{s['rec']}.png"), format="PNG", optimize=True)
                sp1 = _now()
                save_png_s = sp1 - sp0

            row = {
                "rec": s["rec"],
                "category": s["category"],
                "label": int(s["label"]),
                "N_used": int(rp_meta["N_used"]),
                "downsample": int(rp_meta["downsample"]),
                "eps": float(rp_meta["eps"]),
                "steps": int(rp_meta["steps"]),
                "block": int(rp_meta["block"]),
                "load_txt_s": float(rp_timing["load_txt_s"]),
                "preprocess_s": float(rp_timing["preprocess_s"]),
                "rp_compute_s": float(rp_timing["rp_compute_s"]),
                "rp_map_s": float(rp_timing["rp_map_s"]),
                "resize224_s": float(rp_timing["resize224_s"]),
                "transform_s": float(t_tf1 - t_tf0),
                "save_png_s": float(save_png_s),
                # fill later
                "h2d_s": 0.0,
                "infer_s": 0.0,
                "pred": -1,
                "batch_size": int(batch_size),
                "model_kind": getattr(model, "__class__", type(model)).__name__,
                "transform_mode": getattr(transform_fn, "__name__", "callable"),
                "device": str(device),
            }
            batch_rows.append(row)

        # (B) stack + H2D
        x1 = torch.stack(cpu_inputs_1, dim=0)  # (B,1,224,224)
        if is_dual:
            x2 = torch.stack(cpu_inputs_2, dim=0)

        h0 = _now()
        x1 = x1.to(device, non_blocking=True)
        if is_dual:
            x2 = x2.to(device, non_blocking=True)
        _sync(device)
        h1 = _now()
        h2d_total_s = h1 - h0
        h2d_per_s = h2d_total_s / len(batch)

        # (C) inference
        def _forward():
            if is_dual:
                return model(x1, x2)
            return model(x1)

        if amp and _is_cuda(device):
            def _forward_amp():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return _forward()
            logits, infer_ms = _time_gpu_ms(device, _forward_amp)
        else:
            logits, infer_ms = _time_gpu_ms(device, _forward)

        infer_total_s = infer_ms / 1000.0
        infer_per_s = infer_total_s / len(batch)

        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()

        # fill rows
        for i, row in enumerate(batch_rows):
            row["h2d_s"] = float(h2d_per_s)
            row["infer_s"] = float(infer_per_s)
            row["pred"] = int(preds[i])

            # totals
            total_compute = (
                row["preprocess_s"] + row["rp_compute_s"] + row["rp_map_s"] + row["resize224_s"] +
                row["transform_s"] + row["h2d_s"] + row["infer_s"]
            )
            row["total_compute_s"] = float(total_compute)
            row["total_e2e_s"] = float(row["load_txt_s"] + total_compute)
            row["total_with_save_s"] = float(row["total_e2e_s"] + row["save_png_s"])
            rows.append(row)

    wall1 = _now()
    wall_s = wall1 - wall0
    overall_sps = len(rows) / wall_s if wall_s > 0 else float("inf")

    # outputs
    per_csv = os.path.join(out_dir, "e2e_timing_per_sample.csv")
    sum_csv = os.path.join(out_dir, "e2e_timing_summary.csv")
    write_csv(per_csv, rows)
    write_summary(sum_csv, rows, group_key="category")

    return {
        "count": len(rows),
        "wall_s": wall_s,
        "overall_samples_per_s": overall_sps,
        "per_sample_csv": per_csv,
        "summary_csv": sum_csv,
        "save_png": save_png,
        "transform_mode": transform_fn.__name__ if hasattr(transform_fn, "__name__") else "callable",
        "amp": amp,
    }


# ----------------------------
# Meta info
# ----------------------------

def collect_meta(args, device: torch.device) -> Dict:
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": None,
        "args": vars(args),
    }
    try:
        import torchvision
        meta["torchvision"] = torchvision.__version__
    except Exception:
        meta["torchvision"] = None

    if _is_cuda(device):
        meta["device_name"] = torch.cuda.get_device_name(device)
        meta["cuda_version"] = torch.version.cuda
        meta["cudnn_version"] = torch.backends.cudnn.version()
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["model_only", "e2e"], default="e2e")
    p.add_argument("--model", choices=["mambaout", "se_mambaout", "dse_mambaout"], default="mambaout")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")  # auto/cpu/cuda
    p.add_argument("--amp", action="store_true")

    # model-only
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=50)

    # dataset / RP params
    p.add_argument("--dir_wind", type=str, default="data/0wind")
    p.add_argument("--dir_manual", type=str, default="data/1manual")
    p.add_argument("--dir_digger", type=str, default="data/2digger")
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--max_per_class", type=int, default=200)

    p.add_argument("--N", type=int, default=1024)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--out_hw", type=int, default=224)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--transform", choices=["minimal", "weak_strong"], default="minimal")

    # saving (optional)
    p.add_argument("--save_png", action="store_true")
    p.add_argument("--save_dir", type=str, default=None)

    p.add_argument("--out_dir", type=str, default="e2e_benchmark_results")

    args = p.parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    # meta
    meta = collect_meta(args, device)

    # build model
    model, is_dual = build_model(args.model, device, weights_path=args.weights)

    # run selected benchmark
    if args.bench == "model_only":
        res = bench_model_only(
            model=model,
            is_dual=is_dual,
            device=device,
            batch_size=args.batch_size,
            iters=args.iters,
            warmup=args.warmup,
            amp=args.amp,
        )
        meta["result"] = res
        print(json.dumps(res, indent=2))

    else:
        rpgen = RecurrencePlotGenerator(
            eps=args.eps,
            steps=args.steps,
            N=args.N,
            downsample=args.downsample,
            block=args.block,
            out_hw=args.out_hw,
            use_fast_std=True,
        )
        transform_fn = make_transform(args.transform, is_dual=is_dual)

        samples = build_file_list(
            dir_wind=args.dir_wind,
            dir_manual=args.dir_manual,
            dir_digger=args.dir_digger,
            split=args.split,
            max_per_class=args.max_per_class,
        )

        res = bench_end2end(
            model=model,
            is_dual=is_dual,
            device=device,
            rpgen=rpgen,
            transform_fn=transform_fn,
            samples=samples,
            batch_size=args.batch_size,
            warmup=min(5, max(1, args.warmup // 10)),
            amp=args.amp,
            save_png=args.save_png,
            save_dir=args.save_dir,
            out_dir=args.out_dir,
        )
        meta["result"] = res
        print(json.dumps(res, indent=2))
        print(f"\nPer-sample CSV: {res['per_sample_csv']}")
        print(f"Summary CSV:    {res['summary_csv']}")
        print(f"Overall throughput (wall): {res['overall_samples_per_s']:.3f} samples/s")

    # write meta
    meta_path = os.path.join(args.out_dir, "e2e_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved: {meta_path}")


if __name__ == "__main__":
    main()
