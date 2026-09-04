# -*- coding: utf-8 -*-
"""
backend/database.py
SQLite 数据持久化层（数据存储层）

表结构：
- devices            设备信息（设备ID、名称、类型、状态、安装位置、健康度）
- diagnosis_records  诊断记录（时间、设备、信号、诊断结果、置信度、模型）
- alarm_records      告警记录（时间、设备、告警级别、告警内容、处理状态）
- rul_records        寿命预测记录（时间、设备、预测RUL、健康度）
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "database", "bearing_monitor.db")


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """初始化数据库表结构与默认设备数据。"""
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            device_type TEXT,
            location    TEXT,
            status      TEXT DEFAULT '正常',
            health      REAL DEFAULT 100.0,
            created_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS diagnosis_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id   TEXT,
            timestamp   TEXT,
            signal_len  INTEGER,
            fault_label TEXT,
            fault_cn    TEXT,
            confidence  REAL,
            model       TEXT,
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
        );
        CREATE TABLE IF NOT EXISTS alarm_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id   TEXT,
            timestamp   TEXT,
            level       TEXT,
            message     TEXT,
            handled     INTEGER DEFAULT 0,
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
        );
        CREATE TABLE IF NOT EXISTS rul_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id   TEXT,
            timestamp   TEXT,
            pred_rul    REAL,
            health      REAL,
            FOREIGN KEY (device_id) REFERENCES devices(device_id)
        );
    """)
    # 默认设备（幂等）
    default_devices = [
        ("DEV-001", "1# 主电机驱动轴承", "电机轴承", "一号车间 A 工位", "正常", 99.2, None),
        ("DEV-002", "2# 离心风机驱动轴承", "风机轴承", "二号车间 B 工位", "正常", 97.8, None),
        ("DEV-003", "3# 压缩机传动轴承", "压缩机轴承", "三号车间 C 工位", "正常", 95.5, None),
    ]
    for dev in default_devices:
        cur.execute(
            "INSERT OR IGNORE INTO devices (device_id, name, device_type, location, "
            "status, health, created_at) VALUES (?,?,?,?,?,?,?)",
            (*dev[:6], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


# ---------------- 设备管理 ----------------
def list_devices(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_device(device_id: str, db_path: str = DB_PATH
               ) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM devices WHERE device_id=?", (device_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_device_health(device_id: str, health: float, status: str,
                         db_path: str = DB_PATH) -> None:
    conn = _connect(db_path)
    conn.execute("UPDATE devices SET health=?, status=? WHERE device_id=?",
                 (float(health), status, device_id))
    conn.commit()
    conn.close()


# ---------------- 诊断记录 ----------------
def insert_diagnosis_record(device_id: str, signal_len: int,
                            fault_label: str, fault_cn: str,
                            confidence: float, model: str,
                            db_path: str = DB_PATH) -> int:
    conn = _connect(db_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        "INSERT INTO diagnosis_records (device_id, timestamp, signal_len, "
        "fault_label, fault_cn, confidence, model) VALUES (?,?,?,?,?,?,?)",
        (device_id, ts, int(signal_len), fault_label, fault_cn,
         float(confidence), model))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def query_diagnosis_records(limit: int = 100, device_id: Optional[str] = None,
                            db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    if device_id:
        rows = conn.execute(
            "SELECT * FROM diagnosis_records WHERE device_id=? "
            "ORDER BY timestamp DESC LIMIT ?", (device_id, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM diagnosis_records ORDER BY timestamp DESC LIMIT ?",
            (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- 告警记录 ----------------
def insert_alarm(device_id: str, level: str, message: str,
                 db_path: str = DB_PATH) -> int:
    conn = _connect(db_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        "INSERT INTO alarm_records (device_id, timestamp, level, message) "
        "VALUES (?,?,?,?)", (device_id, ts, level, message))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def query_alarms(limit: int = 50, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM alarm_records ORDER BY timestamp DESC "
                        "LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- RUL 记录 ----------------
def insert_rul_record(device_id: str, pred_rul: float, health: float,
                      db_path: str = DB_PATH) -> int:
    conn = _connect(db_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.execute(
        "INSERT INTO rul_records (device_id, timestamp, pred_rul, health) "
        "VALUES (?,?,?,?)", (device_id, ts, float(pred_rul), float(health)))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def query_rul_records(limit: int = 100, db_path: str = DB_PATH
                      ) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM rul_records ORDER BY timestamp DESC "
                        "LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_stats(db_path: str = DB_PATH) -> Dict[str, int]:
    """统计各表记录数，用于仪表盘概览。"""
    conn = _connect(db_path)
    stats = {}
    for table in ("devices", "diagnosis_records", "alarm_records", "rul_records"):
        row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
        stats[table] = row["c"]
    unhandled = conn.execute(
        "SELECT COUNT(*) AS c FROM alarm_records WHERE handled=0").fetchone()
    stats["unhandled_alarms"] = unhandled["c"]
    conn.close()
    return stats
