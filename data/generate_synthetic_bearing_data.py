# -*- coding: utf-8 -*-
"""
轴承振动信号数据集生成器

数据来源：自建（仿真生成）
生成方式：基于轴承故障机理模型，计算特征频率（BPFO/BPFI/BSF/FTF），
         叠加转频谐波、故障冲击成分及高斯白噪声模拟真实振动信号。

包含状态：healthy（正常）、inner_fault（内圈故障）、
          outer_fault（外圈故障）、ball_fault（滚动体故障）

输出格式：CSV（含 10+ 维时频域统计特征及标签）
输出路径：D:\DXB\liyuze0120\data\bearing_dataset.csv

"""

import os
import numpy as np
import pandas as pd
from scipy import signal

# ==================== 固定路径配置 ====================
DATA_DIR = r"D:\DXB\liyuze0120\data"
os.makedirs(DATA_DIR, exist_ok=True)  # 如果目录不存在则自动创建

# ==================== 信号参数配置 ====================
RPM = 1500          # 转速
fs = 10000          # 采样频率 (Hz)
duration = 5        # 采样时长 (秒)
n_samples = fs * duration
t = np.linspace(0, duration, n_samples)

# 轴承参数 (以6205轴承为例)
n = 9               # 滚动体个数
d = 7.94e-3         # 滚动体直径 (m)
D = 39.04e-3        # 节径 (m)
alpha = 0           # 接触角 (度)

# ==================== 计算特征频率 ====================
fr = RPM / 60
BPFI = n/2 * fr * (1 + d/D * np.cos(np.radians(alpha)))
BPFO = n/2 * fr * (1 - d/D * np.cos(np.radians(alpha)))
BSF  = D/(2*d) * fr * (1 - (d/D * np.cos(np.radians(alpha)))**2)
FTF  = 1/2 * fr * (1 - d/D * np.cos(np.radians(alpha)))

print(f"转频: {fr:.2f} Hz")
print(f"内圈故障(BPFI): {BPFI:.2f} Hz")
print(f"外圈故障(BPFO): {BPFO:.2f} Hz")
print(f"滚动体故障(BSF): {BSF:.2f} Hz")

# ==================== 生成各类信号 ====================
def generate_signal(fault_type, snr_db=20):
    """生成轴承振动信号"""
    # 1. 基频成分 (转频及其谐波)
    harmonics = [1, 2, 3, 4, 5]
    harmonic_amps = [0.5, 0.3, 0.15, 0.08, 0.04]
    x = np.zeros(n_samples)
    for h, amp in zip(harmonics, harmonic_amps):
        x += amp * np.sin(2 * np.pi * h * fr * t + np.random.rand() * 2 * np.pi)
    
    # 2. 故障特征频率成分 (含边带)
    if fault_type == 'inner':
        f_fault = BPFI
        amp_fault = 0.8
        for k in [-1, 0, 1]:
            x += amp_fault * 0.5 * np.sin(2 * np.pi * (f_fault + k*fr) * t + np.random.rand() * 2 * np.pi)
    elif fault_type == 'outer':
        f_fault = BPFO
        amp_fault = 1.0
        x += amp_fault * np.sin(2 * np.pi * f_fault * t + np.random.rand() * 2 * np.pi)
    elif fault_type == 'ball':
        f_fault = BSF
        amp_fault = 0.6
        x += amp_fault * np.sin(2 * np.pi * f_fault * t + np.random.rand() * 2 * np.pi)
    else:  # healthy
        pass
    
    # 3. 添加冲击成分 (模拟故障冲击)
    if fault_type != 'healthy':
        impact_interval = 1.0 / f_fault if f_fault > 0 else 0.1
        for i in range(int(duration / impact_interval)):
            idx = int(i * impact_interval * fs)
            if idx < n_samples:
                decay = np.exp(-20 * (t[idx:min(idx+200, n_samples)] - t[idx]))
                x[idx:min(idx+200, n_samples)] += 0.3 * decay * np.random.randn(min(200, n_samples-idx))
    
    # 4. 添加噪声
    signal_power = np.mean(x**2)
    noise_power = signal_power / (10**(snr_db/10))
    x += np.sqrt(noise_power) * np.random.randn(n_samples)
    
    return x

# ==================== 提取时域/频域特征 ====================
def extract_features(data, fs):
    """提取统计特征"""
    features = {}
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['rms'] = np.sqrt(np.mean(data**2))
    features['peak'] = np.max(np.abs(data))
    features['peak_to_peak'] = np.max(data) - np.min(data)
    features['crest_factor'] = features['peak'] / (features['rms'] + 1e-10)
    features['skewness'] = np.mean(((data - features['mean'])/features['std'])**3)
    features['kurtosis'] = np.mean(((data - features['mean'])/features['std'])**4)
    
    # 频域特征 (FFT)
    fft_vals = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(len(data), 1/fs)
    magnitude = np.abs(fft_vals[:len(fft_vals)//2])
    freq = fft_freq[:len(fft_freq)//2]
    features['spectral_centroid'] = np.sum(freq * magnitude) / (np.sum(magnitude) + 1e-10)
    features['spectral_spread'] = np.sqrt(np.sum(((freq - features['spectral_centroid'])**2) * magnitude) / (np.sum(magnitude) + 1e-10))
    
    return features

# ==================== 构建数据集 ====================
data_records = []
states = {
    'healthy': generate_signal('healthy', snr_db=25),
    'inner_fault': generate_signal('inner', snr_db=20),
    'outer_fault': generate_signal('outer', snr_db=20),
    'ball_fault': generate_signal('ball', snr_db=20),
}

for state_name, signal_data in states.items():
    window_size = 1024
    step = 512
    for start in range(0, len(signal_data) - window_size, step):
        window = signal_data[start:start + window_size]
        features = extract_features(window, fs)
        features['label'] = state_name
        features['rpm'] = RPM
        features['window_id'] = start // step
        data_records.append(features)

df = pd.DataFrame(data_records)

# ==================== 保存到指定路径 ====================
output_path = os.path.join(DATA_DIR, 'bearing_dataset.csv')
df.to_csv(output_path, index=False)
print(f"\n✅ 数据集已生成至: {output_path}")
print(f"共 {len(df)} 条样本")
print("\n各类别样本数:")
print(df['label'].value_counts())