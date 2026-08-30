# -*- coding: utf-8 -*-
"""
数据预处理流水线 - 适配绝对路径
输入: D:\DXB\liyuze0120\data\bearing_dataset.csv
输出: D:\DXB\liyuze0120\data\ (train_processed.csv, test_processed.csv, artifacts/)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# ==================== 固定路径配置 ====================
DATA_DIR = r"D:\DXB\liyuze0120\data"
os.makedirs(DATA_DIR, exist_ok=True)

# 子目录用于存放模型配件（标准化器、编码器）
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ==================== 1. 加载原始数据 ====================
def load_raw_data():
    file_path = os.path.join(DATA_DIR, 'bearing_dataset.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 未找到原始数据文件: {file_path}\n请先运行 generate_bearing_dataset.py 生成数据。")
    
    df = pd.read_csv(file_path)
    print(f"✅ 原始数据加载成功，形状: {df.shape}")
    print(f"标签分布:\n{df['label'].value_counts()}")
    return df

# ==================== 2. 数据清洗 ====================
def clean_data(df):
    if 'window_id' in df.columns:
        df.drop('window_id', axis=1, inplace=True)
    
    # 如果转速恒定，剔除转速列（方差为0，对模型无贡献）
    if 'rpm' in df.columns and df['rpm'].nunique() == 1:
        print(f"⚠️ 'rpm' 为常数 {df['rpm'].iloc[0]}，已自动剔除。")
        df.drop('rpm', axis=1, inplace=True)
    
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)
        print("⚠️ 缺失值已填充")
    else:
        print("✅ 无缺失值")
    
    return df

# ==================== 3. 标签编码 ====================
def encode_labels(df):
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"✅ 标签编码映射: {label_mapping}")
    return df, le

# ==================== 4. 特征与标签分离 ====================
def split_features_labels(df):
    X = df.drop(['label', 'label_encoded'], axis=1)
    y = df['label_encoded'].values
    
    # 剔除方差为0的特征（安全冗余检查）
    zero_var_cols = X.columns[X.var() == 0].tolist()
    if zero_var_cols:
        print(f"⚠️ 剔除零方差特征: {zero_var_cols}")
        X.drop(zero_var_cols, axis=1, inplace=True)
    
    print(f"✅ 特征矩阵维度: {X.shape}")
    return X, y

# ==================== 5. 划分训练集与测试集 ====================
def split_train_test(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✅ 划分完成: 训练集 {X_train.shape[0]} 条, 测试集 {X_test.shape[0]} 条")
    return X_train, X_test, y_train, y_test

# ==================== 6. 特征标准化 ====================
def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print("✅ 标准化完成（均值≈0，标准差≈1）")
    return X_train_scaled, X_test_scaled, scaler

# ==================== 7. 保存结果 ====================
def save_processed_data(X_train, X_test, y_train, y_test, scaler, label_encoder):
    # 保存CSV
    train_df = X_train.copy()
    train_df['label'] = y_train
    train_path = os.path.join(DATA_DIR, 'train_processed.csv')
    train_df.to_csv(train_path, index=False)
    
    test_df = X_test.copy()
    test_df['label'] = y_test
    test_path = os.path.join(DATA_DIR, 'test_processed.csv')
    test_df.to_csv(test_path, index=False)
    
    # 保存标准化器和编码器（供后续预测使用）
    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
    encoder_path = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"✅ 训练集已保存: {train_path}")
    print(f"✅ 测试集已保存: {test_path}")
    print(f"✅ 标准化器已保存: {scaler_path}")
    print(f"✅ 标签编码器已保存: {encoder_path}")

# ==================== 主执行流 ====================
if __name__ == "__main__":
    print("="*50)
    print("开始执行数据预处理流水线")
    print(f"数据根目录: {DATA_DIR}")
    print("="*50)
    
    try:
        df = load_raw_data()
        df = clean_data(df)
        df, label_encoder = encode_labels(df)
        X, y = split_features_labels(df)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
        save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder)
        
        print("\n📊 预处理后的训练集预览（前5行）:")
        print(X_train_scaled.head())
        print("\n🎉 数据预处理全部完成！")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")