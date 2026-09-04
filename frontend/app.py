# -*- coding: utf-8 -*-
"""
frontend/app.py
Streamlit 前端仪表盘（前端展示层）

五个页面：设备概览 / 实时监控 / 故障诊断 / RUL预测 / 历史分析
通过 REST API 调用 FastAPI 后端（默认 http://localhost:8000）。

启动：
    streamlit run frontend/app.py --server.port 8501
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_simulator import DataSimulator, STATE_LABELS_CN

# ==================== 配置 ====================
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").strip().rstrip("/")
FS = 10000
WINDOW_SIZE = 1024

st.set_page_config(page_title="设备状态监测与预测性维护系统",
                   page_icon="⚙️", layout="wide")

# ==================== 后端调用封装 ====================
@st.cache_data(ttl=3)
def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ==================== 侧边栏导航 ====================
st.sidebar.title("⚙️ 设备状态监测")
st.sidebar.caption("基于 Vibe Coding 的预测性维护系统")
page = st.sidebar.radio(
    "功能导航",
    ["设备概览", "实时监控", "故障诊断", "RUL 预测", "历史分析"],
)
st.sidebar.divider()
st.sidebar.caption(f"后端服务：{API_BASE}")


# ==================== 工具函数 ====================
def plot_waveform(sig: np.ndarray, title: str = "振动波形"):
    t = np.arange(len(sig)) / FS * 1000
    fig = go.Figure(go.Scatter(x=t, y=sig, mode="lines",
                               line=dict(color="#1E5AA8", width=1)))
    fig.update_layout(title=title, xaxis_title="时间 (ms)",
                      yaxis_title="加速度 (g)", height=280, margin=dict(t=40))
    return fig


def plot_spectrum(sig: np.ndarray, title: str = "频谱"):
    from scipy.fft import rfft, rfftfreq
    n = len(sig)
    mag = np.abs(rfft(sig)) / n * 2
    freq = rfftfreq(n, 1 / FS)
    fig = go.Figure(go.Scatter(x=freq, y=mag, mode="lines",
                               line=dict(color="#2E8B57", width=1)))
    fig.update_layout(title=title, xaxis_title="频率 (Hz)",
                      yaxis_title="幅值", height=280, margin=dict(t=40))
    return fig


def health_gauge(health: float, title: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health,
        title={"text": title},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#1E5AA8"},
               "steps": [{"range": [0, 40], "color": "#ffcccc"},
                         {"range": [40, 70], "color": "#ffe6b3"},
                         {"range": [70, 100], "color": "#d6f0d6"}]}))
    fig.update_layout(height=220, margin=dict(t=50, b=10))
    return fig


# ==================== 页面 1：设备概览 ====================
def page_overview():
    st.title("设备概览")
    st.caption("设备状态总览 · 健康度 · 告警统计")

    stats = api_get("/api/stats")
    devices = api_get("/api/devices")

    if isinstance(stats, dict) and "error" not in stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("在线设备", stats.get("devices", 0))
        c2.metric("诊断记录", stats.get("diagnosis_records", 0))
        c3.metric("告警总数", stats.get("alarm_records", 0))
        c4.metric("未处理告警", stats.get("unhandled_alarms", 0),
                  delta=-stats.get("unhandled_alarms", 0))
    else:
        st.error(f"后端未连接：{stats}")

    if isinstance(devices, list) and devices:
        col = st.columns(2)
        for i, dev in enumerate(devices):
            with col[i % 2]:
                st.subheader(dev["device_id"] + "  " + dev["name"])
                st.caption(f"{dev['device_type']} · {dev['location']}")
                st.plotly_chart(health_gauge(dev["health"], "健康度"),
                                use_container_width=True)
                st.info(f"状态：**{dev['status']}**")
        with st.expander("设备明细表"):
            st.dataframe(pd.DataFrame(devices), use_container_width=True)
    else:
        st.warning("暂无设备数据")


# ==================== 页面 2：实时监控 ====================
def page_realtime():
    st.title("实时监控")
    st.caption("模拟实时振动信号采集 · 波形 / 频谱 / 在线诊断")

    device_id = st.selectbox("监控设备", ["DEV-001", "DEV-002", "DEV-003"])
    status = st.selectbox("模拟状态", list(STATE_LABELS_CN.values()))
    # 反查英文状态名
    fault = [k for k, v in STATE_LABELS_CN.items() if v == status][0]
    model = st.selectbox("诊断模型", ["random_forest", "xgboost", "cnn1d"],
                         format_func=lambda m: {"random_forest": "随机森林",
                                                "xgboost": "XGBoost",
                                                "cnn1d": "1D-CNN"}[m])

    if st.button("开始采集（生成 1 秒数据）", type="primary"):
        sim = DataSimulator(fs=FS, seed=int(np.random.rand() * 10000))
        sig = sim.generate(fault=fault, n=FS, snr_db=20.0)
        st.plotly_chart(plot_waveform(sig), use_container_width=True)
        st.plotly_chart(plot_spectrum(sig), use_container_width=True)

        result = api_post("/api/diagnose", {
            "device_id": device_id, "waveform": sig.tolist(), "model": model})
        if "error" in result:
            st.error(f"诊断失败：{result['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("诊断结果", result["fault_cn"])
            c2.metric("置信度", f"{result['confidence']:.2%}")
            c3.metric("模型", result["model"])
            probs = result.get("probabilities", {})
            st.plotly_chart(
                px.bar(x=list(probs.keys()), y=list(probs.values()),
                       labels={"x": "状态", "y": "概率"},
                       title="各类别概率分布",
                       color=list(probs.keys())),
                use_container_width=True)


# ==================== 页面 3：故障诊断 ====================
def page_diagnose():
    st.title("故障诊断")
    st.caption("振动信号上传 → 特征提取 → 诊断结果与置信度")

    device_id = st.selectbox("诊断设备", ["DEV-001", "DEV-002", "DEV-003"])
    mode = st.radio("输入方式", ["模拟生成信号", "手动输入特征向量"])
    mode_all = st.checkbox("三种模型联合诊断（集成表决）")
    model = st.selectbox(
        "诊断模型（联合诊断时自动使用全部模型）",
        ["random_forest", "xgboost", "cnn1d"],
        format_func=lambda m: {"random_forest": "随机森林",
                               "xgboost": "XGBoost",
                               "cnn1d": "1D-CNN"}[m],
        disabled=mode_all)

    if "diag_sig" not in st.session_state:
        st.session_state.diag_sig = None
    if "diag_feats" not in st.session_state:
        st.session_state.diag_feats = None

    if mode == "模拟生成信号":
        status = st.selectbox("实际状态（用于演示）",
                              list(STATE_LABELS_CN.values()))
        fault = [k for k, v in STATE_LABELS_CN.items() if v == status][0]
        if st.button("生成信号", type="primary"):
            sim = DataSimulator(fs=FS, seed=int(np.random.rand() * 10000))
            st.session_state.diag_sig = sim.generate(fault=fault,
                                                     n=WINDOW_SIZE, snr_db=20.0)
            st.session_state.diag_feats = None
        if st.session_state.diag_sig is not None:
            st.success("信号已生成，可点击下方按钮执行诊断")
            st.plotly_chart(plot_waveform(st.session_state.diag_sig),
                            use_container_width=True)
            st.plotly_chart(plot_spectrum(st.session_state.diag_sig),
                            use_container_width=True)
    else:
        st.info("输入 10 维特征（与训练一致）：mean, std, rms, peak, peak_to_peak, "
                "crest_factor, skewness, kurtosis, spectral_centroid, spectral_spread")
        feat_input = st.text_area("特征向量（逗号分隔）",
                                  "0.0,0.5,0.5,1.5,3.0,3.0,0.1,3.0,300,400")
        try:
            st.session_state.diag_feats = [float(x) for x in
                                           feat_input.replace("，", ",").split(",")]
            if len(st.session_state.diag_feats) != 10:
                st.warning("特征数量应为 10 个")
        except ValueError:
            st.error("特征格式错误，请用逗号分隔的 10 个数值")

    if st.button("执行诊断", type="primary",
                 disabled=(mode == "模拟生成信号"
                           and st.session_state.diag_sig is None)):
        if mode == "模拟生成信号":
            sig = st.session_state.diag_sig
            if sig is None:
                st.stop()
            if mode_all:
                result = api_post("/api/diagnose/all",
                                  {"device_id": device_id,
                                   "waveform": sig.tolist()})
                if "error" in result:
                    st.error(f"诊断失败：{result['error']}")
                else:
                    st.success(f"**最终结论（多数表决）：{result['final_fault_cn']}**")
                    rows = []
                    for m, res in result["model_results"].items():
                        rows.append({"模型": m,
                                     "结论": res.get("fault_cn", res.get("error", "—")),
                                     "置信度": round(res.get("confidence", 0), 4)})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                result = api_post("/api/diagnose", {
                    "device_id": device_id, "waveform": sig.tolist(),
                    "model": model})
                if "error" in result:
                    st.error(f"诊断失败：{result['error']}")
                else:
                    c1, c2 = st.columns(2)
                    c1.metric("诊断结果", result["fault_cn"])
                    c2.metric("置信度", f"{result['confidence']:.2%}")
                    st.plotly_chart(
                        px.bar(x=list(result["probabilities"].keys()),
                               y=list(result["probabilities"].values()),
                               labels={"x": "状态", "y": "概率"},
                               color=list(result["probabilities"].keys())),
                        use_container_width=True)
                    if "features" in result:
                        with st.expander("提取的 10 维特征"):
                            st.json(result["features"])
        else:
            feats = st.session_state.diag_feats
            if feats is None:
                st.stop()
            # 特征向量直接调用后端诊断（需先用特征构造请求）
            # 后端 diagnose 接口接收波形；此处直接调用本地服务层完成特征诊断
            from backend.service import get_service
            try:
                result = get_service().diagnose_features(feats, model)
                c1, c2 = st.columns(2)
                c1.metric("诊断结果", result["fault_cn"])
                c2.metric("置信度", f"{result['confidence']:.2%}")
                st.plotly_chart(
                    px.bar(x=list(result["probabilities"].keys()),
                           y=list(result["probabilities"].values()),
                           labels={"x": "状态", "y": "概率"},
                           color=list(result["probabilities"].keys())),
                    use_container_width=True)
            except Exception as e:
                st.error(f"诊断失败：{e}")


# ==================== 页面 4：RUL 预测 ====================
def page_rul():
    st.title("RUL 预测（剩余使用寿命）")
    st.caption("基于 LSTM 的退化趋势分析与剩余寿命预测")

    device_id = st.selectbox("预测设备", ["DEV-001", "DEV-002", "DEV-003"])
    n_windows = st.slider("退化观测段数（输入序列长度）", 6, 30, 12)
    st.caption("模拟设备从轻微故障逐步退化的过程，基于最近若干段信号预测剩余寿命")

    if st.button("运行退化模拟并预测 RUL", type="primary"):
        sim = DataSimulator(fs=FS, seed=int(np.random.rand() * 10000))
        # 生成退化波形序列：故障强度从 0 线性到 2.5
        severities = np.linspace(0.0, 2.5, n_windows)
        waves = []
        rms_series = []
        for sev in severities:
            w = sim.generate(fault="inner_fault", n=WINDOW_SIZE,
                             severity=sev, snr_db=20.0)
            waves.append(w.tolist())
            rms_series.append(float(np.sqrt(np.mean(w ** 2))))

        with st.spinner("正在预测剩余寿命..."):
            result = api_post("/api/predict-rul", {
                "device_id": device_id, "waveforms": waves})

        if "error" in result:
            st.error(f"预测失败：{result['error']}")
        else:
            pred_rul = result["pred_rul"]
            health = result["health"]
            rul_max = result.get("rul_max", 120)
            c1, c2, c3 = st.columns(3)
            c1.metric("预测剩余寿命", f"{pred_rul:.1f} 步")
            c2.metric("健康度", f"{health:.1f} / 100")
            c3.metric("维护建议",
                      "立即安排维护" if pred_rul <= 20 else
                      ("近期关注" if pred_rul <= 50 else "状态良好"))

            # 退化趋势
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, n_windows + 1)),
                                     y=rms_series, mode="lines+markers",
                                     name="RMS 趋势", line=dict(color="#c0392b")))
            fig.update_layout(title="退化趋势（RMS 健康指标）",
                              xaxis_title="观测序号", yaxis_title="RMS",
                              height=320, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(health_gauge(health, "设备健康度"),
                            use_container_width=True)


# ==================== 页面 5：历史分析 ====================
def page_history():
    st.title("历史分析")
    st.caption("诊断记录 · 告警记录 · 模型性能对比")

    tab1, tab2, tab3 = st.tabs(["诊断记录", "告警记录", "模型性能"])

    with tab1:
        recs = api_get("/api/records/diagnosis", {"limit": 200})
        if isinstance(recs, list) and recs:
            df = pd.DataFrame(recs)
            st.dataframe(df, use_container_width=True)
            st.caption(f"共 {len(df)} 条诊断记录")
            # 故障分布
            dist = df["fault_cn"].value_counts().reset_index()
            dist.columns = ["故障类型", "数量"]
            st.plotly_chart(px.pie(dist, names="故障类型", values="数量",
                                   title="诊断结果分布"), use_container_width=True)
        else:
            st.info("暂无诊断记录")

    with tab2:
        alarms = api_get("/api/records/alarms")
        if isinstance(alarms, list) and alarms:
            st.dataframe(pd.DataFrame(alarms), use_container_width=True)
        else:
            st.info("暂无告警记录")

    with tab3:
        summary_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "models", "training_summary.json")
        import json
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            rows = []
            for name, m in summary.get("diagnosis", {}).items():
                rows.append({"模型": {"random_forest": "随机森林",
                                      "xgboost": "XGBoost",
                                      "cnn1d": "1D-CNN"}[name],
                             "准确率": round(m["accuracy"], 4),
                             "精确率": round(m["precision_macro"], 4),
                             "召回率": round(m["recall_macro"], 4),
                             "F1": round(m["f1_macro"], 4)})
            dfm = pd.DataFrame(rows)
            st.dataframe(dfm, use_container_width=True)
            fig = px.bar(dfm, x="模型", y=["准确率", "F1"], barmode="group",
                         title="三种故障诊断模型测试集性能对比")
            st.plotly_chart(fig, use_container_width=True)
            rul = summary.get("rul", {})
            st.metric("LSTM-RUL 预测 RMSE", f"{rul.get('rmse', '—')}")
            st.metric("LSTM-RUL 预测 MAE", f"{rul.get('mae', '—')}")
        else:
            st.info("未找到模型评估汇总文件")


# ==================== 路由 ====================
if page == "设备概览":
    page_overview()
elif page == "实时监控":
    page_realtime()
elif page == "故障诊断":
    page_diagnose()
elif page == "RUL 预测":
    page_rul()
elif page == "历史分析":
    page_history()
