# -*- coding: utf-8 -*-
"""
scripts/init_db.py
初始化 SQLite 数据库（创建表结构 + 写入默认设备）。

用法：
    python scripts/init_db.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import database as db


def main() -> None:
    db.init_db()
    devices = db.list_devices()
    stats = db.db_stats()
    print("=" * 50)
    print("数据库初始化完成")
    print(f"数据库路径: {db.DB_PATH}")
    print(f"设备数量: {len(devices)}")
    for dev in devices:
        print(f"  - {dev['device_id']} | {dev['name']} | 状态: {dev['status']} "
              f"| 健康度: {dev['health']}")
    print(f"表记录数: {stats}")


if __name__ == "__main__":
    main()
