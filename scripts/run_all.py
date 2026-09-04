# -*- coding: utf-8 -*-
"""
scripts/run_all.py
一键启动系统（后端 FastAPI + 前端 Streamlit）。

用法：
    python scripts/run_all.py            # 默认端口 8000 / 8501
    python scripts/run_all.py --port 8000 --fe-port 8501
    python scripts/run_all.py --only-backend   # 只启动后端

启动后：
    - 后端 API 文档:  http://localhost:8000/docs
    - 前端仪表盘:     http://localhost:8501
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键启动设备状态监测系统")
    parser.add_argument("--port", type=int, default=8000, help="后端端口")
    parser.add_argument("--fe-port", type=int, default=8501, help="前端端口")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--only-backend", action="store_true", help="仅启动后端")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    # 确保数据库已初始化
    from backend import database as db
    db.init_db()
    print("[run_all] 数据库已就绪:", db.DB_PATH)

    procs = []

    def _start_backend():
        cmd = [sys.executable, "-m", "uvicorn", "backend.main:app",
               "--host", args.host, "--port", str(args.port)]
        return subprocess.Popen(cmd, cwd=ROOT,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)

    def _start_frontend():
        env = dict(os.environ)
        env["API_BASE"] = f"http://{args.host}:{args.port}"
        cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
               "--server.port", str(args.fe_port), "--server.headless", "true"]
        return subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT)

    try:
        procs.append(_start_backend())
        print(f"[run_all] 后端已启动: http://{args.host}:{args.port}  "
              f"(API 文档: /docs)")
        if not args.only_backend:
            time.sleep(2)
            procs.append(_start_frontend())
            print(f"[run_all] 前端已启动: http://{args.host}:{args.fe_port}")

        if not args.no_browser and not args.only_backend:
            time.sleep(3)
            webbrowser.open(f"http://{args.host}:{args.fe_port}")

        print("\n按 Ctrl+C 停止服务...")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("服务已停止。")


if __name__ == "__main__":
    main()
