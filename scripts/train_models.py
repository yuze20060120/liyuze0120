# -*- coding: utf-8 -*-
"""
scripts/train_models.py
一键训练全部模型：随机森林 / XGBoost / 1D-CNN（故障诊断） + LSTM-RUL（寿命预测）。

用法：
    python scripts/train_models.py            # 完整训练
    python scripts/train_models.py --quick    # 快速训练（用于测试/演示环境）
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.diagnosis_models import train_and_save_models, DEFAULT_MODELS_DIR
from core.rul_model import train_and_save_rul

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速模式（缩短训练）")
    args = parser.parse_args()

    print("=" * 56)
    print("开始训练全部模型")
    print("=" * 56)

    train_feat = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
    test_feat = pd.read_csv(os.path.join(DATA_DIR, "test_processed.csv"))
    raw = pd.read_csv(os.path.join(DATA_DIR, "raw_signal_dataset.csv"))
    degen = pd.read_csv(os.path.join(DATA_DIR, "degradation_dataset.csv"))

    # 1) 故障诊断三模型（特征模型用独立测试集评估）
    diag_results = train_and_save_models(train_feat, raw,
                                         test_features=test_feat,
                                         models_dir=DEFAULT_MODELS_DIR,
                                         quick=args.quick)

    # 2) LSTM-RUL
    rul_results = train_and_save_rul(degen, models_dir=DEFAULT_MODELS_DIR,
                                     quick=args.quick)

    print("\n" + "=" * 56)
    print("训练完成，模型评估汇总：")
    print("=" * 56)
    for name, m in diag_results.items():
        print(f"[故障诊断] {name:12s} 准确率={m['accuracy']:.4f} "
              f"F1={m['f1_macro']:.4f}")
    print(f"[RUL预测]  lstm_rul    RMSE={rul_results['rmse']:.2f} "
          f"MAE={rul_results['mae']:.2f}（样本{rul_results['samples']}）")

    summary = {"diagnosis": diag_results, "rul": rul_results}
    with open(os.path.join(DEFAULT_MODELS_DIR, "training_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n评估汇总已保存: {os.path.join(DEFAULT_MODELS_DIR, 'training_summary.json')}")


if __name__ == "__main__":
    main()
