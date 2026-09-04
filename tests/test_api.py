# -*- coding: utf-8 -*-
"""后端 API 集成测试（FastAPI TestClient）。"""
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app
from core.data_simulator import DataSimulator

client = TestClient(app)


@pytest.fixture(scope="module")
def waveform():
    sim = DataSimulator(fs=10000, seed=3)
    return sim.generate("inner_fault", n=1024, snr_db=20.0).tolist()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "cnn1d" in r.json()["loaded_models"]


def test_devices_and_stats():
    r = client.get("/api/devices")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert client.get("/api/stats").json()["devices"] >= 3


def test_device_detail_not_found():
    r = client.get("/api/devices/DEV-999")
    assert r.status_code == 404


def test_diagnose_random_forest(waveform):
    r = client.post("/api/diagnose", json={
        "device_id": "DEV-001", "waveform": waveform,
        "model": "random_forest"})
    assert r.status_code == 200
    data = r.json()
    assert "fault_cn" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_diagnose_cnn(waveform):
    r = client.post("/api/diagnose", json={
        "device_id": "DEV-001", "waveform": waveform, "model": "cnn1d"})
    assert r.status_code == 200
    assert "fault_cn" in r.json()


def test_diagnose_invalid_model(waveform):
    r = client.post("/api/diagnose", json={
        "device_id": "DEV-001", "waveform": waveform, "model": "svm"})
    assert r.status_code == 422


def test_diagnose_short_waveform():
    r = client.post("/api/diagnose", json={
        "device_id": "DEV-001", "waveform": [0.1, 0.2], "model": "xgboost"})
    assert r.status_code == 422


def test_diagnose_all(waveform):
    r = client.post("/api/diagnose/all", json={
        "device_id": "DEV-002", "waveform": waveform})
    assert r.status_code == 200
    data = r.json()
    assert data["final_fault_cn"]
    assert set(data["model_results"].keys()) == {
        "random_forest", "xgboost", "cnn1d"}


def test_predict_rul():
    sim = DataSimulator(fs=10000, seed=4)
    waves = [sim.generate("inner_fault", n=1024, severity=s,
                          snr_db=20.0).tolist()
             for s in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]]
    r = client.post("/api/predict-rul", json={
        "device_id": "DEV-003", "waveforms": waves})
    assert r.status_code == 200
    data = r.json()
    assert "pred_rul" in data
    assert 0 <= data["pred_rul"] <= data.get("rul_max", 120) + 5


def test_records_after_diagnosis(waveform):
    before = len(client.get("/api/records/diagnosis").json())
    client.post("/api/diagnose", json={
        "device_id": "DEV-001", "waveform": waveform, "model": "xgboost"})
    after = len(client.get("/api/records/diagnosis").json())
    assert after == before + 1
