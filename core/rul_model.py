# -*- coding: utf-8 -*-
"""
core/rul_model.py
剩余使用寿命（RUL）预测模型模块（《制造智能技术》——机器学习与智能决策方向）

实现基于 LSTM 的时序 RUL 预测：
- 输入：设备最近 T 步的健康指标序列（RMS、峰度、波峰因数等）
- 输出：剩余使用寿命（RUL，步数）
- 训练数据：退化数据集（多台设备 run × 时间步）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "models")

# RUL 模型使用的健康指标特征列
RUL_FEATURE_COLUMNS = ["rms", "kurtosis", "crest_factor"]


class _RUL_LSTM(nn.Module):
    """两层层 LSTM + 回归头，输出 RUL 预测值。"""

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.regressor(out[:, -1, :]).squeeze(-1)


class RULPredictor:
    """LSTM-RUL 预测器。"""

    def __init__(self, n_features: int = 3, seq_len: int = 20,
                 hidden_size: int = 64, num_layers: int = 2,
                 epochs: int = 60, batch_size: int = 128, lr: float = 1e-3,
                 device: Optional[str] = None, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.n_features = n_features
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _RUL_LSTM(n_features, hidden_size, num_layers).to(self.device)
        self.trained = False
        # 数据统计（推理归一化用）
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None
        self.rul_max: Optional[float] = None

    # ---------------- 数据准备 ----------------
    @staticmethod
    def _make_sequences(df: pd.DataFrame, seq_len: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """按 run 滑动窗口构造 (序列, RUL) 样本。"""
        X_list, y_list = [], []
        for _, group in df.groupby("run_id"):
            group = group.sort_values("step")
            feats = group[RUL_FEATURE_COLUMNS].values.astype(float)
            rul = group["rul"].values.astype(float)
            for i in range(len(group) - seq_len + 1):
                X_list.append(feats[i:i + seq_len])
                y_list.append(rul[i + seq_len - 1])
        return np.array(X_list), np.array(y_list)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feat_mean = X.reshape(-1, self.n_features).mean(axis=0)
            self.feat_std = X.reshape(-1, self.n_features).std(axis=0)
            self.feat_std[self.feat_std < 1e-8] = 1.0
        return (X - self.feat_mean) / self.feat_std

    # ---------------- 训练 ----------------
    def train(self, df: pd.DataFrame, X_val=None, y_val=None) -> "RULPredictor":
        """在退化数据集上训练。

        参数:
            df: 退化数据集（run_id / step / 特征列 / rul）
        """
        X, y = self._make_sequences(df, self.seq_len)
        self.rul_max = float(np.max(y))
        y_norm = y / self.rul_max          # RUL 归一化到 [0,1]
        X = self._normalize(X, fit=True)

        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y_norm, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(Xt, yt)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                             shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            if (epoch + 1) % 20 == 0 or epoch == self.epochs - 1:
                print(f"[LSTM-RUL] epoch {epoch + 1}/{self.epochs} "
                      f"loss={total_loss / len(dataset):.5f}")
        self.net.eval()
        self.trained = True
        return self

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def predict(self, seq_features: np.ndarray) -> float:
        """输入最近 seq_len 步的特征序列，返回 RUL 预测值（步数）。"""
        if not self.trained or self.feat_mean is None or self.rul_max is None:
            raise RuntimeError("模型尚未训练或未加载")
        X = np.asarray(seq_features, dtype=float)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        if X.shape[1] != self.seq_len:
            raise ValueError(f"序列长度必须为 {self.seq_len}，当前 {X.shape[1]}")
        X = self._normalize(X)
        self.net.eval()
        xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        pred_norm = self.net(xb)
        return float(pred_norm.cpu().numpy()[0] * self.rul_max)

    def predict_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """对整台设备（按 run 分组）逐窗预测 RUL，返回含 pred_rul 列的 DataFrame。"""
        rows = []
        for run_id, group in df.groupby("run_id"):
            group = group.sort_values("step").reset_index(drop=True)
            feats = group[RUL_FEATURE_COLUMNS].values.astype(float)
            for i in range(self.seq_len - 1, len(group)):
                pred = self.predict(feats[i - self.seq_len + 1: i + 1])
                rows.append({"run_id": run_id, "step": int(group.loc[i, "step"]),
                             "true_rul": float(group.loc[i, "rul"]),
                             "pred_rul": pred})
        return pd.DataFrame(rows)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算 RMSE / MAE。"""
        out = self.predict_series(df)
        return {
            "model": "lstm_rul",
            "rmse": float(np.sqrt(mean_squared_error(out["true_rul"], out["pred_rul"]))),
            "mae": float(mean_absolute_error(out["true_rul"], out["pred_rul"])),
            "samples": int(len(out)),
        }

    # ---------------- 存取 ----------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "n_features": self.n_features,
            "seq_len": self.seq_len,
            "feat_mean": self.feat_mean,
            "feat_std": self.feat_std,
            "rul_max": self.rul_max,
            "trained": self.trained,
        }, path)

    @classmethod
    def load(cls, path: str) -> "RULPredictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(n_features=ckpt["n_features"], seq_len=ckpt["seq_len"])
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.feat_mean = ckpt["feat_mean"]
        obj.feat_std = ckpt["feat_std"]
        obj.rul_max = ckpt["rul_max"]
        obj.trained = ckpt.get("trained", True)
        obj.net.eval()
        return obj


def train_and_save_rul(df: pd.DataFrame, models_dir: str = DEFAULT_MODELS_DIR,
                       quick: bool = False) -> Dict[str, float]:
    """训练并保存 LSTM-RUL 模型，返回在独立测试设备上的评估指标。

    按 run_id 划分：80% 设备训练，20% 设备测试，避免同设备序列泄漏。
    """
    os.makedirs(models_dir, exist_ok=True)
    run_ids = sorted(df["run_id"].unique())
    n_test = max(1, int(len(run_ids) * 0.2))
    test_runs = run_ids[:n_test]
    train_runs = run_ids[n_test:]
    df_train = df[df["run_id"].isin(train_runs)]
    df_test = df[df["run_id"].isin(test_runs)]

    predictor = RULPredictor(epochs=15 if quick else 60)
    predictor.train(df_train)
    metrics = predictor.evaluate(df_test)
    predictor.save(os.path.join(models_dir, "lstm_rul.joblib"))
    print(f"[LSTM-RUL] 训练完成（测试设备 {len(test_runs)} 台）"
          f"RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f}")
    return metrics
