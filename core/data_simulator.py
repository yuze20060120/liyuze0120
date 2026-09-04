# -*- coding: utf-8 -*-
"""
core/data_simulator.py
数据采集模拟器（《制造智能技术》——智能感知与信号检测方向）

功能：
- 模拟加速度传感器振动信号采集：按采样率、窗口长度输出信号流
- 支持 4 种轴承状态（正常 / 内圈 / 外圈 / 滚动体故障）
- 支持实时流式回放（每次返回一个窗口），模拟在线监测场景
- 支持退化过程模拟：随"运行时间"逐步加重故障强度，供 RUL 演示
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.feature_engineering import bearing_characteristic_frequencies


# 轴承参数（6205 深沟球轴承）
BEARING_PARAMS = dict(n_balls=9, ball_diameter=7.94e-3,
                      pitch_diameter=39.04e-3, contact_angle_deg=0.0)

# 状态枚举与中文/英文映射（顺序与 data/artifacts/label_encoder.pkl 一致）
STATES = ["ball_fault", "healthy", "inner_fault", "outer_fault"]
STATE_LABELS_CN = {
    "healthy": "正常",
    "inner_fault": "内圈故障",
    "outer_fault": "外圈故障",
    "ball_fault": "滚动体故障",
}
STATE_ENCODER = {s: i for i, s in enumerate(STATES)}  # 与 LabelEncoder 顺序一致


class DataSimulator:
    """轴承振动信号采集模拟器。

    通过叠加转频谐波、故障特征频率、冲击成分与高斯噪声，
    生成与真实加速度传感器采集特性相近的振动信号流。
    """

    def __init__(self, fs: int = 10000, rpm: float = 1500.0,
                 seed: Optional[int] = None):
        self.fs = fs
        self.rpm = rpm
        self.fr = rpm / 60.0
        self.freqs = bearing_characteristic_frequencies(rpm, **BEARING_PARAMS)
        if seed is not None:
            np.random.seed(seed)
        self._t = 0.0  # 模拟器内部时间游标（秒）

    # ---------------- 信号合成 ----------------
    def _base_signal(self, n: int) -> np.ndarray:
        t = np.linspace(self._t, self._t + n / self.fs, n, endpoint=False)
        x = np.zeros(n)
        harmonics = [1, 2, 3, 4, 5]
        amps = [0.5, 0.3, 0.15, 0.08, 0.04]
        for h, amp in zip(harmonics, amps):
            x += amp * np.sin(2 * np.pi * h * self.fr * t + np.random.rand() * 2 * np.pi)
        return x, t

    def _fault_component(self, fault: str, t: np.ndarray,
                         severity: float = 1.0) -> np.ndarray:
        x = np.zeros_like(t)
        if fault == "inner_fault":
            f = self.freqs["BPFI"]
            for k in (-1, 0, 1):
                x += 0.4 * severity * np.sin(2 * np.pi * (f + k * self.fr) * t
                                             + np.random.rand() * 2 * np.pi)
        elif fault == "outer_fault":
            f = self.freqs["BPFO"]
            x += 1.0 * severity * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi)
        elif fault == "ball_fault":
            f = self.freqs["BSF"]
            x += 0.6 * severity * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi)
        return x

    def _impact_component(self, fault: str, t: np.ndarray,
                          severity: float = 1.0) -> np.ndarray:
        n = len(t)
        x = np.zeros(n)
        if fault == "healthy":
            return x
        f = self.freqs.get({"inner_fault": "BPFI", "outer_fault": "BPFO",
                            "ball_fault": "BSF"}[fault], 100.0)
        interval = 1.0 / f if f > 0 else 0.1
        dt = 1.0 / self.fs
        for i in range(int((t[-1] - t[0]) / interval)):
            idx = int(i * interval / dt)
            if idx < n:
                k = min(200, n - idx)
                decay = np.exp(-20 * np.arange(k) * dt)
                x[idx:idx + k] += 0.3 * severity * decay * np.random.randn(k)
        return x

    def generate(self, fault: str = "healthy", n: Optional[int] = None,
                 severity: float = 1.0, snr_db: Optional[float] = None,
                 advance_time: bool = True) -> np.ndarray:
        """生成一段振动信号。

        参数:
            fault: 状态（healthy / inner_fault / outer_fault / ball_fault）
            n: 信号长度（点数），默认 fs（1 秒）
            severity: 故障严重程度（退化模拟用）
            snr_db: 信噪比；None 时按训练集配置自动取值（healthy=25，故障=20）
            advance_time: 是否推进内部时间游标（模拟连续采集）

        返回:
            振动信号 (n,)
        """
        n = n or self.fs
        base, t = self._base_signal(n)
        x = base + self._fault_component(fault, t, severity)
        x += self._impact_component(fault, t, severity)
        if snr_db is None:
            snr_db = 25.0 if fault == "healthy" else 20.0
        if snr_db is not None:
            power = np.mean(x ** 2)
            noise = power / (10 ** (snr_db / 10))
            x += np.sqrt(noise) * np.random.randn(n)
        if advance_time:
            self._t += n / self.fs
        return x

    # ---------------- 流式接口 ----------------
    def next_window(self, fault: str = "healthy", window_size: int = 1024,
                    severity: float = 1.0) -> np.ndarray:
        """返回下一个采集窗口（模拟实时数据流）。"""
        return self.generate(fault, n=window_size, severity=severity)

    def degradation_stream(self, window_size: int = 1024, n_windows: int = 200,
                           start_severity: float = 0.0,
                           end_severity: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """模拟退化过程数据流：故障强度随时间线性加重。

        返回:
            (windows, severity) 形状分别为 (n_windows, window_size) 与 (n_windows,)
        """
        severities = np.linspace(start_severity, end_severity, n_windows)
        windows = []
        for s in severities:
            windows.append(self.next_window("inner_fault", window_size, severity=s))
        return np.array(windows), severities

    def reset(self) -> None:
        """重置内部时间游标。"""
        self._t = 0.0
