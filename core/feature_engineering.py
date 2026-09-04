# -*- coding: utf-8 -*-
"""
core/feature_engineering.py
信号处理与特征提取模块（《制造智能技术》——智能信号处理与特征提取方向）

功能：
- 信号预处理：去趋势、巴特沃斯带通滤波、归一化
- 轴承特征频率计算：BPFI / BPFO / BSF / FTF
- 时域特征提取（8 维）：均值、标准差、RMS、峰值、峰峰值、波峰因数、偏度、峭度
- 频域特征提取（2 维）：频谱质心、频谱散布
- 对原始振动信号进行滑窗切片并提取完整特征向量
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal


# ============================================================
# 1. 信号预处理
# ============================================================

def detrend_signal(data: np.ndarray) -> np.ndarray:
    """去除信号线性趋势，消除传感器低频漂移影响。

    参数:
        data: 一维振动信号 (N,)

    返回:
        去趋势后的信号 (N,)
    """
    return scipy_signal.detrend(np.asarray(data, dtype=float))


def butter_bandpass(data: np.ndarray, low: float, high: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """巴特沃斯带通滤波。

    参数:
        data: 一维振动信号 (N,)
        low: 通带下限频率 (Hz)
        high: 通带上限频率 (Hz)
        fs: 采样频率 (Hz)
        order: 滤波器阶数

    返回:
        滤波后信号 (N,)
    """
    if high <= low:
        raise ValueError("high 必须大于 low")
    nyquist = fs / 2.0
    if high >= nyquist:
        high = nyquist * 0.99
    b, a = scipy_signal.butter(order, [low / nyquist, high / nyquist], btype="band")
    return scipy_signal.filtfilt(b, a, np.asarray(data, dtype=float))


def normalize_signal(data: np.ndarray) -> np.ndarray:
    """Z-Score 归一化（去均值、除标准差）。

    参数:
        data: 一维振动信号 (N,)

    返回:
        归一化信号 (N,)
    """
    arr = np.asarray(data, dtype=float)
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def preprocess_signal(data: np.ndarray, fs: float,
                      filter_band: Optional[Tuple[float, float]] = None,
                      do_detrend: bool = True,
                      do_normalize: bool = False) -> np.ndarray:
    """信号预处理流水线：去趋势 → 带通滤波 → （可选）归一化。

    参数:
        data: 一维振动信号 (N,)
        fs: 采样频率 (Hz)
        filter_band: (low, high) 带通滤波频带；None 表示不过滤
        do_detrend: 是否去趋势
        do_normalize: 是否归一化

    返回:
        预处理后信号 (N,)
    """
    x = np.asarray(data, dtype=float)
    if do_detrend:
        x = detrend_signal(x)
    if filter_band is not None:
        x = butter_bandpass(x, filter_band[0], filter_band[1], fs)
    if do_normalize:
        x = normalize_signal(x)
    return x


# ============================================================
# 2. 轴承特征频率计算
# ============================================================

def bearing_characteristic_frequencies(rpm: float, n_balls: int = 9,
                                       ball_diameter: float = 7.94e-3,
                                       pitch_diameter: float = 39.04e-3,
                                       contact_angle_deg: float = 0.0
                                       ) -> Dict[str, float]:
    """计算滚动轴承四大特征频率（以 6205 轴承参数为默认）。

    参数:
        rpm: 转速 (rev/min)
        n_balls: 滚动体个数 n
        ball_diameter: 滚动体直径 d (m)
        pitch_diameter: 节径 D (m)
        contact_angle_deg: 接触角 α (度)

    返回:
        {"fr": 转频, "BPFI": 内圈, "BPFO": 外圈, "BSF": 滚动体, "FTF": 保持架}
    """
    fr = rpm / 60.0
    cos_a = np.cos(np.radians(contact_angle_deg))
    ratio = ball_diameter / pitch_diameter * cos_a
    bsf_base = pitch_diameter / (2 * ball_diameter) * fr * (1 - ratio ** 2)
    return {
        "fr": fr,
        "BPFI": n_balls / 2 * fr * (1 + ratio),
        "BPFO": n_balls / 2 * fr * (1 - ratio),
        "BSF": bsf_base,
        "FTF": fr / 2 * (1 - ratio),
    }


# ============================================================
# 3. 频域分析
# ============================================================

def fft_spectrum(data: np.ndarray, fs: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """计算单边幅度谱。

    参数:
        data: 一维振动信号 (N,)
        fs: 采样频率 (Hz)

    返回:
        (freq, magnitude)：频率轴与幅度谱（单边）
    """
    n = len(data)
    fft_vals = np.fft.rfft(data)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = np.abs(fft_vals) / n * 2
    magnitude[0] /= 2  # DC 分量不做 2 倍
    return freq, magnitude


# ============================================================
# 4. 时域 / 频域统计特征提取
# ============================================================

def extract_time_features(data: np.ndarray) -> Dict[str, float]:
    """提取 8 维时域统计特征。"""
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    peak_to_peak = float(np.max(arr) - np.min(arr))
    crest = float(peak / (rms + 1e-10))
    skew = float(np.mean(((arr - mean) / (std + 1e-10)) ** 3)) if n > 1 else 0.0
    kurt = float(np.mean(((arr - mean) / (std + 1e-10)) ** 4)) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest,
        "skewness": skew,
        "kurtosis": kurt,
    }


def extract_freq_features(data: np.ndarray, fs: float) -> Dict[str, float]:
    """提取 2 维频域统计特征（频谱质心、频谱散布）。"""
    freq, magnitude = fft_spectrum(data, fs)
    total = np.sum(magnitude) + 1e-10
    centroid = float(np.sum(freq * magnitude) / total)
    spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * magnitude) / total))
    return {"spectral_centroid": centroid, "spectral_spread": spread}


def extract_features(data: np.ndarray, fs: float,
                     preprocessed: bool = False,
                     filter_band: Optional[Tuple[float, float]] = None
                     ) -> Dict[str, float]:
    """完整特征提取：8 维时域 + 2 维频域 = 10 维特征向量。

    参数:
        data: 一维振动信号 (N,)
        fs: 采样频率 (Hz)
        preprocessed: 是否已做过预处理（True 则跳过预处理）
        filter_band: 预处理时的带通频带

    返回:
        10 维特征字典（键与训练阶段一致）
    """
    if preprocessed:
        x = np.asarray(data, dtype=float)
    else:
        x = preprocess_signal(data, fs, filter_band=filter_band)
    feats = {}
    feats.update(extract_time_features(x))
    feats.update(extract_freq_features(x, fs))
    return feats


# 特征列顺序（与训练 / 推理保持一致）
FEATURE_COLUMNS: List[str] = [
    "mean", "std", "rms", "peak", "peak_to_peak",
    "crest_factor", "skewness", "kurtosis",
    "spectral_centroid", "spectral_spread",
]


def sliding_window_features(data: np.ndarray, fs: float,
                            window_size: int = 1024, step: int = 512,
                            filter_band: Optional[Tuple[float, float]] = None
                            ) -> List[Dict[str, float]]:
    """对整段信号滑窗切片并逐窗提取特征。

    参数:
        data: 一维振动信号 (N,)
        fs: 采样频率 (Hz)
        window_size: 窗口长度（点数）
        step: 窗口步长（点数）
        filter_band: 带通频带

    返回:
        特征字典列表，每个字典对应一个窗口
    """
    x = np.asarray(data, dtype=float)
    feats_list = []
    for start in range(0, len(x) - window_size, step):
        window = x[start:start + window_size]
        feats_list.append(extract_features(window, fs, filter_band=filter_band))
    return feats_list
