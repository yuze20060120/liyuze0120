# -*- coding: utf-8 -*-
"""特征工程模块单元测试。"""
import numpy as np
import pytest

from core.feature_engineering import (FEATURE_COLUMNS,
                                      bearing_characteristic_frequencies,
                                      butter_bandpass, detrend_signal,
                                      extract_features, extract_freq_features,
                                      extract_time_features, fft_spectrum,
                                      normalize_signal, preprocess_signal,
                                      sliding_window_features)


def test_bearing_characteristic_frequencies():
    freqs = bearing_characteristic_frequencies(rpm=1500)
    assert abs(freqs["fr"] - 25.0) < 1e-6
    # 6205 轴承理论值（近似）：BPFI≈135.4, BPFO≈89.6, BSF≈58.9, FTF≈10.0
    assert abs(freqs["BPFI"] - 135.38) < 1.0
    assert abs(freqs["BPFO"] - 89.62) < 1.0
    assert abs(freqs["BSF"] - 58.92) < 1.0
    assert abs(freqs["FTF"] - 9.96) < 1.0


def test_time_features_shape_and_values():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(1024)
    feats = extract_time_features(sig)
    assert len(feats) == 8
    assert abs(feats["mean"]) < 0.3
    assert abs(feats["std"] - 1.0) < 0.2
    assert feats["peak"] > 0
    assert feats["crest_factor"] > 1.0


def test_freq_features():
    rng = np.random.default_rng(1)
    # 构造 300Hz 正弦，频谱质心应接近 300
    t = np.linspace(0, 1, 10000, endpoint=False)
    sig = np.sin(2 * np.pi * 300 * t)
    feats = extract_freq_features(sig, fs=10000)
    assert 250 < feats["spectral_centroid"] < 350
    assert feats["spectral_spread"] > 0


def test_butter_bandpass_shape():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(2048)
    out = butter_bandpass(sig, 50, 500, fs=10000)
    assert out.shape == sig.shape
    assert np.isfinite(out).all()


def test_preprocess_and_normalize():
    rng = np.random.default_rng(3)
    sig = rng.standard_normal(2048) * 5 + 2  # 带偏置
    det = detrend_signal(sig)
    assert abs(np.mean(det)) < 1e-8
    norm = normalize_signal(sig)
    assert abs(np.mean(norm)) < 1e-8
    assert abs(np.std(norm) - 1.0) < 1e-8


def test_extract_features_returns_10_dims():
    rng = np.random.default_rng(4)
    sig = rng.standard_normal(1024)
    feats = extract_features(sig, fs=10000)
    assert len(feats) == 10
    assert set(feats.keys()) == set(FEATURE_COLUMNS)


def test_fft_spectrum():
    rng = np.random.default_rng(5)
    sig = rng.standard_normal(1024)
    freq, mag = fft_spectrum(sig, 10000)
    assert len(freq) == len(mag) == 513
    assert freq[0] == 0


def test_sliding_window_features():
    rng = np.random.default_rng(6)
    sig = rng.standard_normal(4096)
    feats_list = sliding_window_features(sig, fs=10000,
                                         window_size=1024, step=512)
    # range(0, 4096-1024, 512) = 0..2560 → 6 个窗口
    assert len(feats_list) == 6
    assert set(feats_list[0].keys()) == set(FEATURE_COLUMNS)
