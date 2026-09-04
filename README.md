# 基于 Vibe Coding 的设备状态监测与预测性维护系统

《制造智能技术》课程设计 —— 面向旋转机械（电机轴承）的状态监测与预测性维护 B/S 系统

答辩人：李宇泽 · 智能制造工程专业

---

## 一、项目简介

本课程设计运用《制造智能技术》课程的**三大核心技术方向**，采用 **Vibe Coding（AI 辅助编程）** 方法，开发了一套完整的 **B/S 架构可运行 Demo**，面向制造/工业环节中的设备运维与生产管理场景：

| 技术方向 | 对应课程模块 | 系统实现 |
|---------|-------------|---------|
| 智能感知与信号检测 | 智能感知与检测模块 | 模拟加速度传感器数据采集、采样率配置、实时数据流回放 |
| 智能信号处理与特征提取 | 信号处理与特征工程模块 | 去趋势、带通滤波、FFT 频谱分析、轴承特征频率计算、时域8维+频域2维特征提取 |
| 机器学习与智能决策 | 机器学习与智能决策模块 | 随机森林 / XGBoost / 1D-CNN 故障诊断、LSTM-RUL 剩余寿命预测、模型部署 |

**系统功能闭环**：数据采集 → 信号处理与特征提取 → 故障诊断（三种模型）→ RUL 寿命预测 → 结果入库 → 前端仪表盘可视化 → 告警管理。

> **数据集来源声明**：本系统当前使用的核心数据集为**自建数据集**，通过 Python 仿真生成（基于轴承故障机理模型，计算 BPFO/BPFI/BSF/FTF 特征频率，叠加转频谐波、故障冲击及高斯白噪声），**非公开数据集**。后续计划引入 CWRU 轴承数据集、NASA C-MAPSS 数据集做迁移验证与对比。

---

## 二、系统架构（四层 B/S 架构）

```
┌──────────────────────────────────────────────────────────────┐
│              前端展示层（Streamlit，端口 8501）                 │
│      设备概览 │ 实时监控 │ 故障诊断 │ RUL 预测 │ 历史分析        │
└───────────────────────────┬──────────────────────────────────┘
                            │  HTTP REST API
┌───────────────────────────┴──────────────────────────────────┐
│              后端服务层（FastAPI，端口 8000）                   │
│   /api/diagnose │ /api/diagnose/all │ /api/predict-rul        │
│   /api/devices  │ /api/records/*    │ /api/stats              │
└───────┬───────────────┬───────────────┬──────────────────────┘
        │               │               │
┌───────┴──────┐ ┌──────┴───────┐ ┌─────┴──────────┐
│  算法模块层   │ │  算法模块层   │ │   算法模块层    │
│ 数据采集模拟  │ │ 特征工程      │ │ RF/XGB/CNN诊断 │
│ (DataSimulator)│ │(滤波/FFT/特征)│ │ + LSTM-RUL    │
└──────────────┘ └──────────────┘ └────────────────┘
┌──────────────────────────────────────────────────────────────┐
│          数据存储层（SQLite + 文件系统）                        │
│  bearing_monitor.db │ CSV 数据集 │ 模型文件(.joblib/.pt)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、项目结构

```
liyuze0120/
├── core/                          # 算法模块层
│   ├── data_simulator.py          # 数据采集模拟器（智能感知）
│   ├── feature_engineering.py     # 信号处理与特征提取（10 维特征）
│   ├── diagnosis_models.py        # 故障诊断：RF / XGBoost / 1D-CNN
│   └── rul_model.py               # LSTM-RUL 剩余寿命预测
├── backend/                       # 后端服务层
│   ├── main.py                    # FastAPI 应用（REST API + Swagger）
│   ├── service.py                 # 推理服务（模型加载 / 诊断 / RUL / 落库）
│   └── database.py                # SQLite 数据持久化
├── frontend/                      # 前端展示层
│   └── app.py                     # Streamlit 仪表盘（5 页面）
├── data/                          # 数据与模型
│   ├── bearing_dataset.csv        # 自建特征数据集（4 状态 × 96 样本）
│   ├── raw_signal_dataset.csv     # 原始信号窗口数据集（1D-CNN 用，1200×1024）
│   ├── degradation_dataset.csv    # 退化数据集（LSTM-RUL 用，60 台设备×120 步）
│   ├── train_processed.csv        # 预处理后训练集（288 条）
│   ├── test_processed.csv         # 预处理后测试集（96 条）
│   ├── artifacts/                 # 标准化器 / 标签编码器
│   ├── models/                    # 训练好的模型与评估结果
│   └── database/bearing_monitor.db# SQLite 数据库
├── scripts/
│   ├── train_models.py            # 一键训练全部模型
│   ├── init_db.py                 # 初始化数据库
│   └── run_all.py                 # 一键启动前后端
├── tests/                         # pytest 自动化测试（37 项）
├── requirements.txt
└── README.md
```

---

## 四、环境配置

### 4.1 要求
- Python 3.10+（本机验证 Python 3.14）
- Windows 10/11 / Ubuntu 20.04+ / macOS 12+

### 4.2 安装依赖
```bash
pip install -r requirements.txt
# PyTorch 若默认源较慢，可用 CPU 版：
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4.3 一键启动（Windows 双击，推荐）
项目根目录提供两个批处理文件，双击即可，无需在终端敲命令：
- **`启动系统.bat`**：自动完成「定位 Python → 创建项目虚拟环境 .venv → 安装缺失依赖 → 训练缺失模型 → 初始化数据库 → 启动前后端 → 打开浏览器」。
  脚本会自动跳过 Windows 应用商店的 Python 占位程序，找到真实可用的 Python；依赖统一安装到项目自带的 `.venv` 中，不影响系统全局环境。
  首次双击会安装依赖（约几分钟），之后启动约 15 秒。
- **`停止系统.bat`**：按端口（8000/8501）结束前后端服务。

> 附加：在 `cmd` 中运行 `启动系统.bat check` 可仅做环境自检（不启动服务）。

### 4.4 手动运行流程（4 步）
> 手动运行时建议先激活虚拟环境：`.venv\Scripts\activate`（或 `source .venv/bin/activate`）
```bash
# ① 生成与预处理数据（已包含在仓库，可跳过）
python data/generate_synthetic_bearing_data.py
python data/preprocess_bearing_data.py
python data/generate_raw_signal_dataset.py
python data/generate_degradation_dataset.py

# ② 训练全部模型（RF / XGBoost / 1D-CNN / LSTM-RUL）
python scripts/train_models.py

# ③ 初始化数据库
python scripts/init_db.py

# ④ 一键启动前后端
python scripts/run_all.py
```

启动后访问：
- 前端仪表盘：http://localhost:8501
- 后端 API 文档（Swagger）：http://localhost:8000/docs
- 后端健康检查：http://localhost:8000/api/health

---

## 五、功能说明（前端 5 页面）

| 页面 | 功能 |
|------|------|
| 设备概览 | 设备列表、健康度仪表盘、告警与记录统计 |
| 实时监控 | 模拟实时采集：波形 / 频谱展示 + 在线诊断 |
| 故障诊断 | 三种模型诊断（或集成表决），特征展示与置信度分布 |
| RUL 预测 | 退化过程模拟，LSTM 预测剩余寿命与维护建议 |
| 历史分析 | 诊断记录、告警记录、三模型性能对比 |

---

## 六、后端 API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 系统健康检查 |
| POST | `/api/diagnose` | 振动信号故障诊断（指定模型） |
| POST | `/api/diagnose/all` | 三种模型联合诊断（集成表决） |
| POST | `/api/predict-rul` | LSTM-RUL 剩余寿命预测 |
| GET | `/api/devices` | 设备列表 |
| GET | `/api/devices/{id}` | 设备详情 |
| GET | `/api/records/diagnosis` | 诊断历史 |
| GET | `/api/records/alarms` | 告警历史 |
| GET | `/api/records/rul` | RUL 预测历史 |
| GET | `/api/stats` | 数据库统计 |

---

## 七、模型性能（测试集评估）

| 模型 | 输入 | 准确率 | F1 |
|------|------|--------|-----|
| 随机森林 | 10 维时频域特征 | 1.0000 | 1.0000 |
| XGBoost | 10 维时频域特征 | 0.9896 | 0.9896 |
| 1D-CNN | 原始振动信号（1024 点） | 1.0000 | 1.0000 |
| LSTM-RUL | 最近 20 步健康指标 | RMSE=0.60 / MAE=0.47 | — |

> 详细评估结果见 `data/models/training_summary.json`。

---

## 八、测试

```bash
python -m pytest tests -v
```
覆盖：特征工程、数据模拟器、三种诊断模型、LSTM-RUL、SQLite 数据库、后端 API 集成测试（共 37 项）。

---

## 九、Git 版本控制

本项目全程使用 Git 小步提交，提交历史见 `git log`，包含：
- 数据资源构建与预处理
- 特征工程与算法模块开发
- 后端 API 与前端仪表盘
- 自动化测试与系统集成

---

## 十、Vibe Coding 实践

本项目全程采用 AI 辅助编程（Vibe Coding）方法开发：
- **工具**：Cursor、Aider、DeepSeek API、豆包
- **方法论**：意图驱动、迭代生成、上下文管理、人工审查、测试先行
- **关键 Prompt 策略**：任务拆解、规格驱动（先定义函数签名与类型注解）、约束说明（如"先划分后标准化防数据泄露"）、上下文工程（让 AI 先读现有代码再生成）

AI 使用披露与典型纠错案例详见 `设计报告.md`（第六章）。
