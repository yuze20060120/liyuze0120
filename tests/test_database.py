# -*- coding: utf-8 -*-
"""数据库层单元测试（使用临时数据库文件）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import database as db


@pytest.fixture()
def tmp_db(tmp_path):
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


def test_init_creates_default_devices(tmp_db):
    devices = db.list_devices(tmp_db)
    assert len(devices) >= 3
    ids = [d["device_id"] for d in devices]
    assert "DEV-001" in ids


def test_insert_and_query_diagnosis(tmp_db):
    rid = db.insert_diagnosis_record("DEV-001", 1024, "inner_fault",
                                     "内圈故障", 0.95, "random_forest", tmp_db)
    assert rid > 0
    recs = db.query_diagnosis_records(device_id="DEV-001", db_path=tmp_db)
    assert len(recs) == 1
    assert recs[0]["fault_cn"] == "内圈故障"
    assert recs[0]["confidence"] == pytest.approx(0.95)


def test_insert_and_query_alarm(tmp_db):
    db.insert_alarm("DEV-001", "warning", "内圈故障", tmp_db)
    db.insert_alarm("DEV-002", "critical", "需维护", tmp_db)
    alarms = db.query_alarms(limit=10, db_path=tmp_db)
    assert len(alarms) == 2
    # 时间戳同为秒级，顺序不保证，只校验两级告警均存在
    assert {a["level"] for a in alarms} == {"warning", "critical"}


def test_insert_and_query_rul(tmp_db):
    db.insert_rul_record("DEV-001", 30.5, 55.0, tmp_db)
    recs = db.query_rul_records(db_path=tmp_db)
    assert len(recs) == 1
    assert recs[0]["pred_rul"] == pytest.approx(30.5)


def test_update_device_health(tmp_db):
    db.update_device_health("DEV-001", 60.0, "异常", tmp_db)
    dev = db.get_device("DEV-001", tmp_db)
    assert dev["health"] == pytest.approx(60.0)
    assert dev["status"] == "异常"


def test_db_stats(tmp_db):
    db.insert_diagnosis_record("DEV-001", 1024, "healthy", "正常",
                               1.0, "random_forest", tmp_db)
    db.insert_alarm("DEV-001", "warning", "测试", tmp_db)
    stats = db.db_stats(tmp_db)
    assert stats["diagnosis_records"] == 1
    assert stats["alarm_records"] == 1
    assert stats["unhandled_alarms"] == 1
