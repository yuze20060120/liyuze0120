# -*- coding: utf-8 -*-
"""
backend/service.py
后端业务服务层：模型加载、故障诊断推理、RUL 预测、数据持久化联动。

统一管理：
- 特征标准化器（scaler.pkl）与标签编码器（label_encoder.pkl）
- 三种故障诊断模型（随机森林 / XGBoost / 1D-CNN）
- LSTM-RUL 预测器
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from core.data_simulator import STATE_LABELS_CN
from core.diagnosis_models import (CNN1DDiagnoser, RandomForestDiagnoser,
                                   XGBoostDiagnoser, load_diagnoser)
from core.feature_engineering import (FEATURE_COLUMNS, extract_features,
                                      preprocess_signal)
from core.rul_model import RULPredictor, RUL_FEATURE_COLUMNS
from backend import database as db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "data", "artifacts")
MODELS = os.path.join(ROOT, "data", "models")
FS = 10000  # 采样率

# 注意：训练阶段特征提取未做带通滤波，为保持一致，推理阶段也不滤波


class InferenceService:
    """全局推理服务（单例加载模型）。"""

    def __init__(self):
        self._scaler = None
        self._label_encoder = None
        self._models: Dict[str, Any] = {}
        self._rul: Optional[RULPredictor] = None

    # ---------------- 模型加载 ----------------
    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = joblib.load(os.path.join(ARTIFACTS, "scaler.pkl"))
        return self._scaler

    @property
    def label_encoder(self):
        if self._label_encoder is None:
            self._label_encoder = joblib.load(
                os.path.join(ARTIFACTS, "label_encoder.pkl"))
        return self._label_encoder

    def get_diagnoser(self, name: str):
        if name not in self._models:
            self._models[name] = load_diagnoser(name, MODELS)
        return self._models[name]

    @property
    def rul_model(self) -> RULPredictor:
        if self._rul is None:
            self._rul = RULPredictor.load(os.path.join(MODELS, "lstm_rul.joblib"))
        return self._rul

    # ---------------- 诊断推理 ----------------
    def label_to_cn(self, label: int) -> str:
        name = self.label_encoder.inverse_transform([int(label)])[0]
        return STATE_LABELS_CN.get(name, name)

    def diagnose_features(self, features: List[float],
                          model: str = "random_forest") -> Dict[str, Any]:
        """基于 10 维特征向量诊断（RF / XGBoost）。"""
        import pandas as _pd
        x = _pd.DataFrame([np.asarray(features, dtype=float)],
                          columns=FEATURE_COLUMNS)
        x_scaled = self.scaler.transform(x)
        diagnoser = self.get_diagnoser(model)
        proba = diagnoser.predict_proba(x_scaled)[0]
        label = int(np.argmax(proba))
        return {
            "model": model,
            "fault_label": self.label_encoder.inverse_transform([label])[0],
            "fault_cn": self.label_to_cn(label),
            "confidence": float(proba[label]),
            "probabilities": {self.label_to_cn(i): float(p)
                              for i, p in enumerate(proba)},
        }

    def diagnose_waveform(self, waveform: List[float],
                          model: str = "random_forest") -> Dict[str, Any]:
        """基于原始振动信号诊断：
        - RF / XGBoost：预处理 → 提取特征 → 标准化 → 诊断
        - cnn1d：直接输入原始信号端到端诊断
        """
        sig = np.asarray(waveform, dtype=float)
        if model == "cnn1d":
            diagnoser = self.get_diagnoser("cnn1d")
            proba = diagnoser.predict_proba(sig)[0]
            label = int(np.argmax(proba))
            return {
                "model": "cnn1d",
                "fault_label": self.label_encoder.inverse_transform([label])[0],
                "fault_cn": self.label_to_cn(label),
                "confidence": float(proba[label]),
                "probabilities": {self.label_to_cn(i): float(p)
                                  for i, p in enumerate(proba)},
            }
        feats = extract_features(sig, FS)
        result = self.diagnose_features([feats[c] for c in FEATURE_COLUMNS], model)
        result["features"] = {c: round(float(feats[c]), 6) for c in FEATURE_COLUMNS}
        return result

    def diagnose_all_models(self, waveform: List[float]) -> Dict[str, Any]:
        """三种模型联合诊断，返回各模型结果与多数表决结论。"""
        sig = np.asarray(waveform, dtype=float)
        results = {}
        votes: List[str] = []
        for m in ("random_forest", "xgboost", "cnn1d"):
            try:
                r = self.diagnose_waveform(sig, m)
                results[m] = r
                votes.append(r["fault_label"])
            except Exception as e:  # 单个模型失败不阻塞整体
                results[m] = {"error": str(e)}
        # 多数表决
        from collections import Counter
        cnt = Counter(votes)
        final_label = cnt.most_common(1)[0][0]
        return {
            "model_results": results,
            "final_fault_label": final_label,
            "final_fault_cn": STATE_LABELS_CN.get(final_label, final_label),
            "votes": votes,
        }

    # ---------------- RUL 预测 ----------------
    def predict_rul(self, recent_features: List[List[float]]) -> float:
        """输入最近 seq_len 步健康指标序列，返回 RUL（步数）。"""
        return self.rul_model.predict(np.asarray(recent_features, dtype=float))

    def predict_rul_from_waveforms(self, waveforms: List[np.ndarray]) -> float:
        """输入最近若干段振动信号，逐段提取健康指标后预测 RUL。"""
        feats = []
        for w in waveforms:
            f = extract_features(w, FS)
            feats.append([f[c] for c in RUL_FEATURE_COLUMNS])
        # 若不足 seq_len，用首段重复填充补齐
        seq_len = self.rul_model.seq_len
        while len(feats) < seq_len:
            feats.insert(0, feats[0])
        return self.predict_rul(feats[-seq_len:])


# 全局单例
_service: Optional[InferenceService] = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


def run_diagnosis_and_record(device_id: str, waveform: List[float],
                             model: str = "random_forest") -> Dict[str, Any]:
    """执行诊断并写入数据库，异常时自动产生告警。"""
    svc = get_service()
    result = svc.diagnose_waveform(waveform, model)
    db.insert_diagnosis_record(
        device_id=device_id,
        signal_len=len(waveform),
        fault_label=result["fault_label"],
        fault_cn=result["fault_cn"],
        confidence=result["confidence"],
        model=result["model"],
    )
    # 故障状态 → 告警
    if result["fault_label"] != "healthy":
        db.insert_alarm(device_id, "warning",
                        f"{result['fault_cn']}（置信度 {result['confidence']:.2f}）")
        dev = db.get_device(device_id)
        health = max(10.0, (dev["health"] if dev else 100.0) - 5.0)
        db.update_device_health(device_id, health, "异常")
    return result


def run_rul_and_record(device_id: str, waveforms: List[np.ndarray]) -> Dict[str, Any]:
    """执行 RUL 预测并写入数据库，低寿命时产生告警。"""
    svc = get_service()
    rul = svc.predict_rul_from_waveforms(waveforms)
    health = max(0.0, min(100.0, rul * 100.0 / svc.rul_model.rul_max))
    db.insert_rul_record(device_id, float(rul), float(health))
    db.update_device_health(device_id, health, "正常" if rul > 20 else "需维护")
    if rul <= 20:
        db.insert_alarm(device_id, "critical",
                        f"剩余寿命仅 {rul:.0f} 步，建议安排维护")
    return {"device_id": device_id, "pred_rul": float(rul),
            "health": float(health), "rul_max": svc.rul_model.rul_max}
