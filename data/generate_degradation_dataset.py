# -*- coding: utf-8 -*-
"""
data/generate_degradation_dataset.py
生成"设备退化"数据集，供 LSTM-RUL（剩余使用寿命）预测模型使用。

说明：
- 模拟多台设备（多个 run）从健康到故障的退化过程。
- 每台设备退化时间轴固定为 RUL_MAX 个时间步；
  每个时间步生成一段振动信号，提取退化健康指标（RMS、峰度等）。
- RUL 标签 = 该时间步距失效的剩余步数（线性退化假设，便于学习）。
- 输出 CSV：run_id, step, rms, kurtosis, ..., rul（剩余寿命）。

用法：
    python data/generate_degradation_dataset.py
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_simulator import DataSimulator
from core.feature_engineering import extract_time_features

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 10000
WINDOW_SIZE = 1024
N_RUNS = 60              # 设备台数（run）
RUL_MAX = 120            # 每台设备时间步数（退化序列长度）
OUTPUT = os.path.join(DATA_DIR, "degradation_dataset.csv")


def simulate_run(sim: DataSimulator, n_steps: int) -> pd.DataFrame:
    """模拟一台设备的退化过程。

    故障强度从 0（健康）线性增长到 3（严重故障），
    提取每时间步的健康指标，并标注剩余寿命 RUL。
    """
    severities = np.linspace(0.0, 3.0, n_steps)
    rows = []
    for step, sev in enumerate(severities):
        win = sim.generate(fault="inner_fault", n=WINDOW_SIZE,
                           severity=sev, snr_db=20.0)
        feats = extract_time_features(win)
        rul = n_steps - 1 - step          # 剩余寿命（步数）
        rows.append({"run_id": 0, "step": step, "rms": feats["rms"],
                     "kurtosis": feats["kurtosis"], "crest_factor": feats["crest_factor"],
                     "severity": sev, "rul": rul})
    return pd.DataFrame(rows)


def main() -> None:
    all_frames = []
    for run_id in range(N_RUNS):
        sim = DataSimulator(fs=FS, seed=100 + run_id)
        frame = simulate_run(sim, RUL_MAX)
        frame["run_id"] = run_id
        all_frames.append(frame)

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(OUTPUT, index=False)
    print(f"退化数据集已保存: {OUTPUT}")
    print(f"形状: {df.shape}，共 {N_RUNS} 台设备，每台 {RUL_MAX} 步")
    print("预览:")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
