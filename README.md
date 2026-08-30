《制造智能技术》课程设计 — 旋转机械轴承状态监测与预测性维护

### 一、项目简介
本项目是《制造智能技术》课程设计作品，采用 Vibe Coding（AI 辅助编程） 方法开发，面向制造场景中的旋转机械（以电机轴承为典型对象）设备状态监测与预测性维护问题。

系统覆盖《制造智能技术》课程中智能感知与信号检测、智能信号处理与特征提取、机器学习与智能决策三大核心技术方向，当前已完成数据集的构建与预处理工作。

### 二、项目结构
text
predictive-maintenance-system/
│
├── data/                                    # 数据目录
│   ├── bearing_dataset.csv                  # ✅ 自建轴承数据集（已生成）
│   ├── train_processed.csv                  # ✅ 预处理后训练集（已生成）
│   ├── test_processed.csv                   # ✅ 预处理后测试集（已生成）
│   └── artifacts/                           # ✅ 模型配件（已生成）
│       ├── scaler.pkl                       # 标准化器
│       └── label_encoder.pkl                # 标签编码器
│
├── src/
│   ├── data/                                # 数据层
│   │   ├── generate_bearing_dataset.py      # ✅ 数据集生成脚本（已完成）
│   │   └── preprocess_bearing_data.py       # ✅ 数据预处理流水线（已完成）
│   ├── features/                            # 特征工程模块（待开发）
│   │   ├── signal_processor.py              # ⏳ 信号预处理
│   │   └── feature_extractor.py             # ⏳ 特征提取
│   ├── models/                              # 算法模型层（待开发）
│   │   ├── random_forest.py                 # ⏳ 随机森林
│   │   ├── xgboost_model.py                 # ⏳ XGBoost
│   │   ├── cnn_model.py                     # ⏳ 1D-CNN
│   │   └── lstm_rul.py                      # ⏳ LSTM RUL
│   ├── backend/                             # 后端 API（待开发）
│   │   └── app.py                           # ⏳ FastAPI 主入口
│   └── frontend/                            # 前端仪表盘（待开发）
│       └── dashboard.py                     # ⏳ Streamlit 主入口
│
├── docs/                                    # 文档目录
│   ├── 选题说明.txt                          # ✅ 已提交
│   └── 方案设计.txt                          # ✅ 已提交
│
├── requirements.txt                         # ⏳ 待整理
└── README.md                                # ✅ 本文件
图例：✅ 已完成 | ⏳ 待开发

### 三、当前进度
### 3.1 已完成工作
序号	任务	状态	说明
1	选题申报	✅	已提交群内接龙
2	选题说明撰写	✅	包含题目、目标、技术方向、选题意义
3	方案设计撰写	✅	包含功能需求、方案论证、技术路线、进度安排
4	自建数据集生成	✅	生成 4 种轴承状态的振动信号及特征，保存至 data/bearing_dataset.csv
5	数据预处理	✅	清洗、编码、划分、标准化，保存训练集/测试集及模型配件

### 3.2 待开发工作
序号	任务	优先级	说明
1	信号处理与特征工程模块	高	滤波、去趋势、时域/频域特征提取
2	故障诊断模型（3 种）	高	随机森林 / XGBoost / 1D-CNN
3	RUL 预测模型（LSTM）	中	基于 NASA C-MAPSS 数据集
4	后端 API 服务（FastAPI）	高	诊断接口、RUL 预测接口
5	前端仪表盘（Streamlit）	高	5 个页面：概览/监控/诊断/RUL/历史
6	系统集成与测试	高	前后端联调、单元测试
7	演示视频与答辩 PPT	高	3 分钟演示视频、答辩 PPT

### 四、数据集说明
### 4.1 自建轴承数据集
通过 generate_bearing_dataset.py 仿真生成，数据保存在 data/bearing_dataset.csv：

项目	说明
信号类型	模拟加速度传感器振动信号
采样频率	10 kHz
采样时长	5 秒/样本
窗口长度	1024 点
窗口步长	512 点
4 种设备状态：

标签	说明
healthy	正常运行
inner_fault	内圈故障
outer_fault	外圈故障
ball_fault	滚动体故障
提取特征（共 10+ 维）：

类别	特征
时域	均值、标准差、RMS、峰值、峰峰值、波峰因数、偏度、峭度
频域	频谱质心、频谱散布

### 数据集来源说明

**本系统当前使用的数据集为自建数据集**，通过 `generate_bearing_dataset.py` 脚本仿真生成，非公开数据集。数据包含 4 种轴承健康状态，基于轴承故障机理模型（特征频率 BPFO/BPFI/BSF/FTF）构造振动信号，并添加转频谐波、故障冲击及背景噪声以模拟真实传感器采集环境。

### 4.2 数据预处理结果
运行 preprocess_bearing_data.py 后生成：

文件	说明
train_processed.csv	标准化后的训练集（75%）
test_processed.csv	标准化后的测试集（25%）
artifacts/scaler.pkl	标准化器（推理时加载）
artifacts/label_encoder.pkl	标签编码器（推理时加载）

### 4.3 计划使用的公开数据集
数据集	用途	状态
CWRU 轴承数据集	故障诊断模型训练与评估	待下载
NASA C-MAPSS	RUL 预测模型训练	待下载
### 五、环境配置

### 5.1 克隆仓库
git clone <仓库地址>
cd predictive-maintenance-system

### 5.2 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或
venv\Scripts\activate      # Windows

### 5.3 安装依赖
bash
pip install -r requirements.txt   # 待整理
当前已使用的库：

numpy

pandas

scipy

scikit-learn

### 六、运行当前已完成模块

### 6.1 生成数据集
bash
cd src/data
python generate_bearing_dataset.py
输出：D:\DXB\liyuze0120\data\bearing_dataset.csv

### 6.2 数据预处理
cd src/data
python preprocess_bearing_data.py
输出：
D:\DXB\liyuze0120\data\train_processed.csv
D:\DXB\liyuze0120\data\test_processed.csv
D:\DXB\liyuze0120\data\artifacts\scaler.pkl
D:\DXB\liyuze0120\data\artifacts\label_encoder.pkl

### 七、计划安排
阶段	周次	任务	状态
第一阶段	第 1-2 天	环境搭建、vibe coding 学习、选题调研	✅
第二阶段	第 3-4 天	选题申报、方案设计	✅
第三阶段	第 5-7 天	数据资源整理（生成 + 预处理）	✅
第四阶段	第 8-9 天	详细开发（特征工程 + 模型训练）	⏳
第五阶段	第 9 天	集成调试	⏳
第六阶段	第 10 天	报告撰写	⏳

### 八、已提交文档
文档	位置	说明
选题说明	docs/选题说明.txt	题目、目标、技术方向、选题意义
方案设计	docs/方案设计.txt	功能需求、方案论证、技术路线、进度安排