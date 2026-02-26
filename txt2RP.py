import os
import csv
import time
import numpy as np
from os import listdir
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from PIL import Image


class NumpyArrayProcessor:
    def __init__(self, input_dirs, output_dirs, stats_dir="rp_timing_stats"):
        """
        :param input_dirs: {"wind": "path1", "knocking": "path2", "machine": "path3"}
        :param output_dirs: {"wind": "path1", "knocking": "path2", "machine": "path3"}
        """
        self.input_dirs = input_dirs
        self.output_dirs = output_dirs
        self.stats_dir = stats_dir

        for path in self.output_dirs.values():
            os.makedirs(path, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

    def get_all_records(self, category):
        """获取指定类别的所有记录文件名（去除后缀）"""
        input_dir = self.input_dirs[category]
        return [file.split(".")[0] for file in listdir(input_dir) if file.endswith(".txt")]

    @staticmethod
    def _standardize_1d(x: np.ndarray) -> np.ndarray:
        """比 StandardScaler() 更轻量的 1D 标准化（同等效果：zero-mean unit-std）"""
        x = x.astype(np.float32, copy=False)
        mu = float(x.mean())
        sd = float(x.std()) + 1e-8
        return (x - mu) / sd

    @staticmethod
    def recurrence_plot_1d_chunked(
        signal_1d: np.ndarray,
        eps: float = 0.10,
        steps: int = 3,
        block: int = 512,
        out_dtype=np.uint8,
    ) -> np.ndarray:
        """
        1D 信号的 RP：R[i,j] = floor(|x_i - x_j| / eps), clip to [0, steps]
        分块计算，避免一次性创建 N×N float 距离矩阵的巨大峰值内存。
        输出用 uint8 存，速度和内存更友好。
        """
        x = np.asarray(signal_1d, dtype=np.float32)
        n = x.shape[0]

        rp = np.empty((n, n), dtype=out_dtype)

        # 预先准备一份整列/整行参与广播的数组，减少重复创建
        x_row = x[None, :]  # (1, n)

        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            # 这里的 diff 是 (block, n)，峰值内存约 block*n*4 bytes
            diff = np.abs(x[i0:i1, None] - x_row)  # float32
            q = np.floor(diff / eps).astype(np.int16, copy=False)
            q[q > steps] = steps
            rp[i0:i1, :] = q.astype(out_dtype, copy=False)

        return rp

    @staticmethod
    def rp_to_grayscale_image(rp: np.ndarray, steps: int) -> np.ndarray:
        """
        将 RP(0..steps) 映射为灰度图 uint8：
        - 模仿 cmap='binary' 的视觉：0 更白，steps 更黑
        """
        rp_f = rp.astype(np.float32, copy=False)
        img = 255.0 * (1.0 - (rp_f / float(steps)))
        return img.clip(0, 255).astype(np.uint8, copy=False)

    @staticmethod
    def save_png_fast(img_u8: np.ndarray, save_path: str):
        """PIL 直接保存，避免 matplotlib 开销"""
        Image.fromarray(img_u8, mode="L").save(save_path, format="PNG", optimize=True)

    def numpy_array2RP(
        self,
        rec: str,
        category: str,
        eps: float = 0.05,
        steps: int = 3,
        N: int | None = 1024,
        downsample: int = 1,
        block: int = 512,
        use_fast_std: bool = True,
    ):
        """
        处理单个记录并生成 RP，同时返回计时信息：
        - preprocess_s：读入+reshape+标准化（不含RP）
        - rp_compute_s：RP 计算
        - save_s：保存 PNG
        - total_s：端到端（preprocess + rp + save）
        """
        input_dir = self.input_dirs[category]
        t0 = time.perf_counter()

        # 读入 (2, 9600) -> reshape 成一列
        np_array = np.loadtxt(f"{input_dir}/{rec}.txt")
        np_array = np_array.reshape(-1)

        # 可选：下采样（非常关键：RP 是 O(N^2)，想实时就必须控 N）
        if downsample > 1:
            np_array = np_array[::downsample]

        # 可选：截取窗口长度 N（默认 1024；你可以按论文设定）
        if (N is not None) and (np_array.shape[0] > N):
            np_array = np_array[:N]

        # 标准化
        if use_fast_std:
            signal = self._standardize_1d(np_array)
        else:
            scaler = StandardScaler()
            signal = scaler.fit_transform(np_array.reshape(-1, 1)).flatten().astype(np.float32)

        t1 = time.perf_counter()

        # RP 计算（分块）
        rp = self.recurrence_plot_1d_chunked(signal, eps=eps, steps=steps, block=block)
        t2 = time.perf_counter()

        # 转灰度 + 保存
        img_u8 = self.rp_to_grayscale_image(rp, steps=steps)
        save_path = f"{self.output_dirs[category]}/{rec}.png"
        self.save_png_fast(img_u8, save_path)
        t3 = time.perf_counter()

        return {
            "rec": rec,
            "category": category,
            "N_used": int(signal.shape[0]),
            "downsample": int(downsample),
            "eps": float(eps),
            "steps": int(steps),
            "block": int(block),
            "preprocess_s": float(t1 - t0),
            "rp_compute_s": float(t2 - t1),
            "save_s": float(t3 - t2),
            "total_s": float(t3 - t0),
            "save_path": save_path,
        }

    @staticmethod
    def _summarize_times(rows, key):
        arr = np.array([r[key] for r in rows], dtype=np.float64)
        if arr.size == 0:
            return {}
        return {
            f"{key}_mean_ms": float(arr.mean() * 1e3),
            f"{key}_median_ms": float(np.median(arr) * 1e3),
            f"{key}_p95_ms": float(np.percentile(arr, 95) * 1e3),
            f"{key}_throughput_sps": float(1.0 / arr.mean()) if arr.mean() > 0 else float("inf"),
        }

    def process_all_records(
        self,
        eps: float = 0.05,
        steps: int = 3,
        N: int | None = 1024,
        downsample: int = 1,
        block: int = 512,
        use_fast_std: bool = True,
        warmup: int = 3,
    ):
        """
        处理所有记录并输出 timing CSV：
        - per_sample CSV：每个样本的 preprocess/rp/save/total
        - summary CSV：每类/总体的 mean/median/p95 + samples/s
        """
        all_rows = []

        for category in self.input_dirs.keys():
            print(f"\nProcessing category: {category}")

            recs = self.get_all_records(category)

            # warmup：避免首次运行（IO、Python 解释器缓存、PIL 初始化）影响统计
            for i in range(min(warmup, len(recs))):
                _ = self.numpy_array2RP(
                    recs[i], category, eps=eps, steps=steps, N=N,
                    downsample=downsample, block=block, use_fast_std=use_fast_std
                )

            for rec in tqdm(recs, desc=f"{category}"):
                row = self.numpy_array2RP(
                    rec, category, eps=eps, steps=steps, N=N,
                    downsample=downsample, block=block, use_fast_std=use_fast_std
                )
                all_rows.append(row)

        # 写 per-sample CSV
        per_sample_csv = os.path.join(self.stats_dir, "rp_timing_per_sample.csv")
        with open(per_sample_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        # 写 summary CSV（按类+总体）
        summary_csv = os.path.join(self.stats_dir, "rp_timing_summary.csv")
        summary_rows = []

        def add_summary(name, rows):
            s = {"group": name}
            s.update(self._summarize_times(rows, "preprocess_s"))
            s.update(self._summarize_times(rows, "rp_compute_s"))
            s.update(self._summarize_times(rows, "save_s"))
            s.update(self._summarize_times(rows, "total_s"))
            # 额外：端到端吞吐（以 total_mean）
            total_mean_s = np.mean([r["total_s"] for r in rows])
            s["end2end_samples_per_s"] = float(1.0 / total_mean_s) if total_mean_s > 0 else float("inf")
            summary_rows.append(s)

        for category in self.input_dirs.keys():
            rows_c = [r for r in all_rows if r["category"] == category]
            add_summary(category, rows_c)

        add_summary("ALL", all_rows)

        # 写 summary
        with open(summary_csv, "w", newline="") as f:
            # union all keys
            keys = set()
            for r in summary_rows:
                keys.update(r.keys())
            keys = ["group"] + sorted([k for k in keys if k != "group"])
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)

        print("\n=== Timing CSV saved ===")
        print(f"Per-sample: {per_sample_csv}")
        print(f"Summary:    {summary_csv}")
        return per_sample_csv, summary_csv


if __name__ == "__main__":
    processor = NumpyArrayProcessor(
        {"wind": "data/0wind", "knocking": "data/1manual", "machine": "data/2digger"},
        {"wind": "Recurrent_map_DAS/0wind", "knocking": "Recurrent_map_DAS/1manual", "machine": "Recurrent_map_DAS/2digger"},
        stats_dir="rp_timing_stats"
    )

    # 关键建议：
    # - N 默认 1024：RP O(N^2)，N=9600 非常难“实时”，且内存也非常大
    # - block 调大能更快，但峰值内存更高；你可以试 256/512/1024
    processor.process_all_records(
        eps=0.05,
        steps=3,
        N=1024,          
        downsample=1,    # 如果你想更快：downsample=2/4
        block=512,
        use_fast_std=True,
        warmup=3
    )
