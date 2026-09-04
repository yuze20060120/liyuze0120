# -*- coding: utf-8 -*-
"""数据模拟器单元测试。"""
import numpy as np
import pytest

from core.data_simulator import (BEARING_PARAMS, STATE_ENCODER, STATES,
                                 DataSimulator)
from core.feature_engineering import bearing_characteristic_frequencies


def test_states_consistent_with_label_encoder():
    # 与 data/artifacts/label_encoder.pkl 的编码保持一致
    assert STATE_ENCODER == {"ball_fault": 0, "healthy": 1,
                             "inner_fault": 2, "outer_fault": 3}
    assert len(STATES) == 4


def test_generate_signal_shape():
    sim = DataSimulator(fs=10000, seed=0)
    sig = sim.generate("healthy", n=1024)
    assert sig.shape == (1024,)
    assert np.isfinite(sig).all()


def test_generate_all_states_distinct():
    sim = DataSimulator(fs=10000, seed=0)
    sigs = {s: sim.generate(s, n=1024, advance_time=False)
            for s in STATES}
    # 各类信号 RMS 应明显不同（故障类幅值更大）
    rms = {s: float(np.sqrt(np.mean(x ** 2))) for s, x in sigs.items()}
    assert rms["healthy"] < rms["inner_fault"] + 1e-6
    # 至少健康与某种故障 RMS 有区分
    assert abs(rms["healthy"] - rms["outer_fault"]) > 0.05


def test_stream_window_advances():
    sim = DataSimulator(fs=10000, seed=0)
    t0 = sim._t
    w1 = sim.next_window("healthy", 1024)
    w2 = sim.next_window("healthy", 1024)
    assert sim._t - t0 > 0
    assert len(w1) == len(w2) == 1024


def test_degradation_stream():
    sim = DataSimulator(fs=10000, seed=0)
    windows, severities = sim.degradation_stream(window_size=1024, n_windows=10)
    assert windows.shape == (10, 1024)
    assert len(severities) == 10
    # 故障强度单调递增
    assert np.all(np.diff(severities) >= 0)


def test_characteristic_frequencies_match():
    sim = DataSimulator(fs=10000, rpm=1500)
    theo = bearing_characteristic_frequencies(1500, **BEARING_PARAMS)
    assert abs(sim.freqs["BPFI"] - theo["BPFI"]) < 1e-9
