# -*- coding: utf-8 -*-
"""
data/generate_raw_signal_dataset.py
生成"原始振动信号窗口"数据集，供 1D-CNN 端到端故障诊断模型使用。

说明：
- 复用 DataSimulator 的信号合成逻辑，保存每个滑动窗口的原始 1024 点信号与标签。
- 输出 CSV：每行一个窗口，列名 win_0 ... win_1023, label。
- 与特征数据集（bearing_dataset.csv）状态定义保持一致（4 类）。

用法：
    python data/generate_raw_signal_dataset.py
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_simulator import DataSimulator, STATES

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FS = 10000
WINDOW_SIZE = 1024
WINDOWS_PER_STATE = 300          # 每类状态窗口数
OUTPUT = os.path.join(DATA_DIR, "raw_signal_dataset.csv")


def main() -> None:
    sim = DataSimulator(fs=FS, seed=42)
    records = []
    for state in STATES:
        for _ in range(WINDOWS_PER_STATE):
            win = sim.generate(fault=state, n=WINDOW_SIZE, snr_db=20.0)
            records.append(list(win) + [state])
        print(f"[{state}] 生成 {WINDOWS_PER_STATE} 个窗口")

    df = pd.DataFrame(records, columns=[f"win_{i}" for i in range(WINDOW_SIZE)] + ["label"])
    df.to_csv(OUTPUT, index=False)
    print(f"原始信号数据集已保存: {OUTPUT}")
    print(f"形状: {df.shape}，类别分布:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
