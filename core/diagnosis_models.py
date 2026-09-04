# -*- coding: utf-8 -*-
"""
core/diagnosis_models.py
故障诊断模型模块（《制造智能技术》——机器学习与智能决策方向）

实现三种故障诊断模型（统一接口，可插拔）：
1. RandomForestClassifier  随机森林（基线模型，基于手工特征）
2. XGBoostClassifier       XGBoost（对比模型，基于手工特征）
3. CNN1DClassifier         1D-CNN（主模型，端到端学习原始振动信号）

统一接口：train / predict / predict_proba / evaluate / save / load
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SkRF
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import torch
import torch.nn as nn

from core.data_simulator import STATE_ENCODER
from core.feature_engineering import FEATURE_COLUMNS


# ============================================================
# 统一接口基类
# ============================================================

class BaseFaultClassifier(ABC):
    """故障诊断模型统一接口。"""

    name: str = "base"

    @abstractmethod
    def train(self, X, y) -> "BaseFaultClassifier":
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        ...

    def evaluate(self, X, y_true) -> Dict[str, Any]:
        """在给定数据上评估，返回指标字典。"""
        y_pred = self.predict(X)
        return {
            "model": self.name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                     zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                               zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def save(self, path: str) -> None:
        """序列化模型。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseFaultClassifier":
        ...


# ============================================================
# 1. 随机森林
# ============================================================

class RandomForestDiagnoser(BaseFaultClassifier):
    name = "random_forest"

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None,
                 random_state: int = 42, **kwargs):
        self.model = SkRF(n_estimators=n_estimators, max_depth=max_depth,
                          random_state=random_state, n_jobs=-1, **kwargs)
        self.feature_columns = FEATURE_COLUMNS

    def train(self, X, y) -> "RandomForestDiagnoser":
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)))

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model.predict_proba(np.asarray(X, dtype=float)))

    @classmethod
    def load(cls, path: str) -> "RandomForestDiagnoser":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} 不是 RandomForestDiagnoser")
        return obj


# ============================================================
# 2. XGBoost
# ============================================================

class XGBoostDiagnoser(BaseFaultClassifier):
    name = "xgboost"

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        self.model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   learning_rate=learning_rate, random_state=random_state,
                                   eval_metric="mlogloss", **kwargs)
        self.feature_columns = FEATURE_COLUMNS

    def train(self, X, y) -> "XGBoostDiagnoser":
        self.model.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(X, dtype=float)))

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.model.predict_proba(np.asarray(X, dtype=float)))

    @classmethod
    def load(cls, path: str) -> "XGBoostDiagnoser":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} 不是 XGBoostDiagnoser")
        return obj


# ============================================================
# 3. 1D-CNN
# ============================================================

class _CNN1D(nn.Module):
    """一维卷积神经网络：卷积块 → 全局池化 → 全连接分类。"""

    def __init__(self, input_len: int = 1024, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=4, padding=15),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=16, stride=4, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 (B, L) → (B, 1, L)
        return self.classifier(self.features(x.unsqueeze(1)))


class CNN1DDiagnoser(BaseFaultClassifier):
    """基于 PyTorch 的 1D-CNN 端到端故障诊断模型（输入为原始信号窗口）。"""

    name = "cnn1d"

    def __init__(self, input_len: int = 1024, n_classes: int = 4,
                 epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
                 device: Optional[str] = None, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.input_len = input_len
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _CNN1D(input_len, n_classes).to(self.device)
        self.trained = False

    # ---------- 数据准备 ----------
    def _prepare(self, X) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_len:
            # 支持任意长度窗口：尾部截断/零填充至 input_len
            if X.shape[1] > self.input_len:
                X = X[:, :self.input_len]
            else:
                pad = np.zeros((X.shape[0], self.input_len - X.shape[1]), dtype=np.float32)
                X = np.concatenate([X, pad], axis=1)
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _labels(self, y) -> torch.Tensor:
        return torch.tensor(np.asarray(y, dtype=np.int64), device=self.device)

    # ---------- 训练 ----------
    def train(self, X, y, X_val=None, y_val=None) -> "CNN1DDiagnoser":
        Xt = self._prepare(X)
        yt = self._labels(y)
        dataset = torch.utils.data.TensorDataset(Xt, yt)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                             shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.net.train()
        for epoch in range(self.epochs):
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.net(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total += xb.size(0)
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                acc = correct / total
                print(f"[CNN1D] epoch {epoch + 1}/{self.epochs} loss={total_loss / total:.4f} "
                      f"acc={acc:.4f}")
        self.net.eval()
        self.trained = True
        return self

    # ---------- 推理 ----------
    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("模型尚未训练")
        self.net.eval()
        out = self.net(self._prepare(X))
        return out.argmax(1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("模型尚未训练")
        self.net.eval()
        out = torch.softmax(self.net(self._prepare(X)), dim=1)
        return out.cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(),
                    "input_len": self.input_len,
                    "n_classes": self.n_classes,
                    "trained": self.trained}, path)

    @classmethod
    def load(cls, path: str) -> "CNN1DDiagnoser":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(input_len=ckpt["input_len"], n_classes=ckpt["n_classes"])
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.trained = ckpt.get("trained", True)
        obj.net.eval()
        return obj


# ============================================================
# 模型注册与工厂
# ============================================================

MODEL_REGISTRY: Dict[str, Any] = {
    "random_forest": RandomForestDiagnoser,
    "xgboost": XGBoostDiagnoser,
    "cnn1d": CNN1DDiagnoser,
}

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "models")


def load_diagnoser(name: str, models_dir: str = DEFAULT_MODELS_DIR
                   ) -> BaseFaultClassifier:
    """按名称加载已训练模型。

    参数:
        name: random_forest / xgboost / cnn1d
        models_dir: 模型目录

    返回:
        模型实例
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}，可选: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    path = os.path.join(models_dir, f"{name}_diagnoser.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}，请先运行 scripts/train_models.py")
    return cls.load(path)


def train_and_save_models(train_features: pd.DataFrame,
                          train_raw: pd.DataFrame,
                          test_features: Optional[pd.DataFrame] = None,
                          models_dir: str = DEFAULT_MODELS_DIR,
                          quick: bool = False) -> Dict[str, Dict[str, Any]]:
    """训练三种模型并保存，返回各模型在测试集上的评估指标。

    参数:
        train_features: 特征数据集（含 FEATURE_COLUMNS 与 label 列，训练用）
        train_raw: 原始信号窗口数据集（含 win_* 与 label 列，训练用）
        test_features: 特征数据集（评估用）；None 时从训练集内部切分
        models_dir: 保存目录
        quick: 快速模式（减少训练量，用于测试）
    """
    os.makedirs(models_dir, exist_ok=True)
    from sklearn.model_selection import train_test_split as tts

    X_feat = train_features[FEATURE_COLUMNS].values.astype(float)
    y_feat = train_features["label"].values.astype(int)

    raw_cols = [c for c in train_raw.columns if c.startswith("win_")]
    X_raw = train_raw[raw_cols].values.astype(float)
    # 原始信号数据集标签为字符串，编码为与特征数据集一致的数值（STATE_ENCODER）
    y_raw = train_raw["label"].map(STATE_ENCODER).values.astype(int)

    # 特征模型：优先使用外部测试集，否则内部按 75/25 切分
    if test_features is not None:
        X_feat_te = test_features[FEATURE_COLUMNS].values.astype(float)
        y_feat_te = test_features["label"].values.astype(int)
    else:
        (X_feat_tr, X_feat_te, y_feat_tr, y_feat_te) = tts(
            X_feat, y_feat, test_size=0.25, random_state=42, stratify=y_feat)
        X_feat, y_feat = X_feat_tr, y_feat_tr

    # 原始信号模型：内部按 75/25 分层切分
    X_raw_tr, X_raw_te, y_raw_tr, y_raw_te = tts(
        X_raw, y_raw, test_size=0.25, random_state=42, stratify=y_raw)

    results: Dict[str, Dict[str, Any]] = {}

    # 1) 随机森林（手工特征）
    rf = RandomForestDiagnoser(n_estimators=100 if quick else 300)
    rf.train(X_feat, y_feat)
    results[rf.name] = rf.evaluate(X_feat_te, y_feat_te)
    rf.save(os.path.join(models_dir, "random_forest_diagnoser.joblib"))
    print(f"[RF] 训练完成 测试准确率={results[rf.name]['accuracy']:.4f}")

    # 2) XGBoost（手工特征）
    xgb = XGBoostDiagnoser(n_estimators=100 if quick else 300)
    xgb.train(X_feat, y_feat)
    results[xgb.name] = xgb.evaluate(X_feat_te, y_feat_te)
    xgb.save(os.path.join(models_dir, "xgboost_diagnoser.joblib"))
    print(f"[XGBoost] 训练完成 测试准确率={results[xgb.name]['accuracy']:.4f}")

    # 3) 1D-CNN（原始信号）
    cnn = CNN1DDiagnoser(epochs=8 if quick else 30)
    cnn.train(X_raw_tr, y_raw_tr)
    results[cnn.name] = cnn.evaluate(X_raw_te, y_raw_te)
    cnn.save(os.path.join(models_dir, "cnn1d_diagnoser.joblib"))
    print(f"[CNN1D] 训练完成 测试准确率={results[cnn.name]['accuracy']:.4f}")

    with open(os.path.join(models_dir, "diagnosis_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results
