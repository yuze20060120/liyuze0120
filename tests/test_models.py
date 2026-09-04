# -*- coding: utf-8 -*-
"""故障诊断与 RUL 模型测试（快速模式，验证可训练、可推理、可存取）。"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_simulator import DataSimulator, STATE_ENCODER
from core.diagnosis_models import (CNN1DDiagnoser, RandomForestDiagnoser,
                                   XGBoostDiagnoser, load_diagnoser)
from core.feature_engineering import FEATURE_COLUMNS, extract_features
from core.rul_model import RULPredictor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "models")


@pytest.fixture(scope="module")
def feature_data():
    """构造小型特征训练数据（4 类 × 每类 30 样本）。"""
    sim = DataSimulator(fs=10000, seed=0)
    rows = []
    for state in ["healthy", "inner_fault", "outer_fault", "ball_fault"]:
        for _ in range(30):
            w = sim.generate(state, n=1024, snr_db=20.0, advance_time=False)
            f = extract_features(w, 10000)
            f["label"] = STATE_ENCODER[state]
            rows.append(f)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def raw_data():
    sim = DataSimulator(fs=10000, seed=1)
    rows = []
    for state in ["healthy", "inner_fault", "outer_fault", "ball_fault"]:
        for _ in range(20):
            w = sim.generate(state, n=1024, snr_db=20.0, advance_time=False)
            rows.append(list(w) + [state])
    return pd.DataFrame(rows, columns=[f"win_{i}" for i in range(1024)] + ["label"])


def test_random_forest_train_predict(feature_data):
    X = feature_data[FEATURE_COLUMNS].values
    y = feature_data["label"].values
    model = RandomForestDiagnoser(n_estimators=20).train(X, y)
    pred = model.predict(X[:5])
    assert pred.shape == (5,)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 4)
    assert np.isclose(proba.sum(axis=1), 1.0).all()


def test_xgboost_train_predict(feature_data):
    X = feature_data[FEATURE_COLUMNS].values
    y = feature_data["label"].values
    model = XGBoostDiagnoser(n_estimators=20).train(X, y)
    assert model.predict(X[:5]).shape == (5,)


def test_cnn1d_train_predict(raw_data):
    cols = [c for c in raw_data.columns if c.startswith("win_")]
    X = raw_data[cols].values
    y = raw_data["label"].map(STATE_ENCODER).values
    model = CNN1DDiagnoser(epochs=2, batch_size=16)
    model.train(X, y)
    pred = model.predict(X[:5])
    assert pred.shape == (5,)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 4)


def test_model_save_load_roundtrip(tmp_path, feature_data):
    X = feature_data[FEATURE_COLUMNS].values
    y = feature_data["label"].values
    model = RandomForestDiagnoser(n_estimators=20).train(X, y)
    path = str(tmp_path / "rf.joblib")
    model.save(path)
    loaded = RandomForestDiagnoser.load(path)
    assert (loaded.predict(X[:3]) == model.predict(X[:3])).all()


def test_cnn_save_load_roundtrip(tmp_path, raw_data):
    cols = [c for c in raw_data.columns if c.startswith("win_")]
    X = raw_data[cols].values
    y = raw_data["label"].map(STATE_ENCODER).values
    model = CNN1DDiagnoser(epochs=2, batch_size=16)
    model.train(X, y)
    path = str(tmp_path / "cnn.pt")
    model.save(path)
    loaded = CNN1DDiagnoser.load(path)
    assert (loaded.predict(X[:3]) == model.predict(X[:3])).all()


def test_rul_train_predict(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for run in range(6):
        n = 40
        sev = np.linspace(0, 3, n)
        for step, s in enumerate(sev):
            frames.append({"run_id": run, "step": step,
                           "rms": 0.4 + s * 0.2 + rng.normal(0, 0.02),
                           "kurtosis": 2.0 + s * 0.3 + rng.normal(0, 0.05),
                           "crest_factor": 2.0 + s * 0.2,
                           "rul": n - 1 - step})
    df = pd.DataFrame(frames)
    model = RULPredictor(seq_len=10, epochs=3, batch_size=32)
    model.train(df)
    # 取测试设备预测
    test = df[df["run_id"] == 5].sort_values("step")
    feats = test[["rms", "kurtosis", "crest_factor"]].values[:10]
    pred = model.predict(feats)
    assert 0 <= pred <= model.rul_max + 5
    # evaluate 返回指标
    metrics = model.evaluate(df[df["run_id"] == 5])
    assert {"rmse", "mae"} <= set(metrics.keys())


def test_pretrained_models_loadable():
    """验证已训练的生产模型可加载（保证演示环境可用）。"""
    for name in ("random_forest", "xgboost", "cnn1d"):
        path = os.path.join(MODELS_DIR, f"{name}_diagnoser.joblib")
        if os.path.exists(path):
            model = load_diagnoser(name)
            assert model.name == name
    rul_path = os.path.join(MODELS_DIR, "lstm_rul.joblib")
    if os.path.exists(rul_path):
        rul = RULPredictor.load(rul_path)
        assert rul.trained
