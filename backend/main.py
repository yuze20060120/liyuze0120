# -*- coding: utf-8 -*-
"""
backend/main.py
FastAPI 后端服务（后端服务层）

提供 RESTful API（自动生成 Swagger 文档 /docs）：
- GET  /api/health                    系统健康检查
- POST /api/diagnose                  振动信号故障诊断（可指定模型）
- POST /api/diagnose/all              三种模型联合诊断
- POST /api/predict-rul               剩余寿命预测
- GET  /api/devices                   设备列表
- GET  /api/devices/{device_id}       设备详情
- GET  /api/records/diagnosis         诊断历史
- GET  /api/records/alarms            告警历史
- GET  /api/records/rul               RUL 历史
- GET  /api/stats                     数据库统计

启动：uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend import database as db
from backend.service import (get_service, run_diagnosis_and_record,
                             run_rul_and_record)

app = FastAPI(
    title="设备状态监测与预测性维护系统 API",
    description="基于Vibe Coding的设备状态监测与预测性维护系统后端服务。"
                "覆盖故障诊断（随机森林/XGBoost/1D-CNN）与 LSTM-RUL 寿命预测。",
    version="1.0.0",
)


# ==================== 请求 / 响应模型 ====================
class DiagnoseRequest(BaseModel):
    device_id: str = Field(default="DEV-001", description="设备ID")
    waveform: List[float] = Field(..., description="振动信号波形（一维数组）")
    model: str = Field(default="random_forest",
                       description="模型：random_forest / xgboost / cnn1d")


class DiagnoseAllRequest(BaseModel):
    device_id: str = Field(default="DEV-001")
    waveform: List[float] = Field(...)


class RULRequest(BaseModel):
    device_id: str = Field(default="DEV-001")
    waveforms: List[List[float]] = Field(..., description="最近若干段振动信号")


# ==================== 基础接口 ====================
@app.get("/")
def root():
    return {"app": "设备状态监测与预测性维护系统",
            "docs": "/docs", "version": "1.0.0"}


@app.get("/api/health")
def health():
    svc = get_service()
    return {
        "status": "ok",
        "loaded_models": ["random_forest", "xgboost", "cnn1d", "lstm_rul"],
        "database": "sqlite",
    }


# ==================== 故障诊断接口 ====================
@app.post("/api/diagnose")
def diagnose(req: DiagnoseRequest):
    if not req.waveform or len(req.waveform) < 256:
        raise HTTPException(status_code=422, detail="波形长度过短（至少 256 点）")
    if req.model not in ("random_forest", "xgboost", "cnn1d"):
        raise HTTPException(status_code=422,
                            detail="model 可选 random_forest / xgboost / cnn1d")
    try:
        return run_diagnosis_and_record(req.device_id, req.waveform, req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/diagnose/all")
def diagnose_all(req: DiagnoseAllRequest):
    if not req.waveform or len(req.waveform) < 256:
        raise HTTPException(status_code=422, detail="波形长度过短（至少 256 点）")
    svc = get_service()
    result = svc.diagnose_all_models(req.waveform)
    # 落库（以多数表决结论记录）
    db.insert_diagnosis_record(
        device_id=req.device_id,
        signal_len=len(req.waveform),
        fault_label=result["final_fault_label"],
        fault_cn=result["final_fault_cn"],
        confidence=1.0,
        model="ensemble",
    )
    return result


# ==================== RUL 预测接口 ====================
@app.post("/api/predict-rul")
def predict_rul(req: RULRequest):
    if not req.waveforms:
        raise HTTPException(status_code=422, detail="waveforms 不能为空")
    try:
        waves = [list(map(float, w)) for w in req.waveforms]
        return run_rul_and_record(req.device_id, waves)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 设备与记录接口 ====================
@app.get("/api/devices")
def devices():
    return db.list_devices()


@app.get("/api/devices/{device_id}")
def device_detail(device_id: str):
    dev = db.get_device(device_id)
    if dev is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    return dev


@app.get("/api/records/diagnosis")
def diagnosis_records(limit: int = 100, device_id: Optional[str] = None):
    return db.query_diagnosis_records(limit=limit, device_id=device_id)


@app.get("/api/records/alarms")
def alarm_records(limit: int = 50):
    return db.query_alarms(limit=limit)


@app.get("/api/records/rul")
def rul_records(limit: int = 100):
    return db.query_rul_records(limit=limit)


@app.get("/api/stats")
def stats():
    return db.db_stats()
