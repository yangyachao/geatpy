#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geatpy 跨平台打包与构建脚本 (Windows / macOS / Linux 通用)

用法:
    python scripts/build.py                # 编译并生成 wheel 至 dist/
    python scripts/build.py --install      # 编译、打包并安装到当前 Python 环境
    python scripts/build.py --test         # 编译、安装并运行 pytest 单元测试
    python scripts/build.py --sdist        # 额外生成源码包 (.tar.gz)
    python scripts/build.py --release      # 发布优化模式 (默认开启)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


def run_cmd(cmd, check=True):
    print(f"\033[34m[执行命令]\033[0m {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if check and res.returncode != 0:
        print(f"\033[31m[错误] 命令执行失败 (退出代码 {res.returncode})\033[0m")
        sys.exit(res.returncode)
    return res.returncode


def ensure_toolchain():
    # 检查 cargo
    cargo_bin = shutil.which("cargo")
    if not cargo_bin:
        cargo_home = Path.home() / ".cargo" / "bin" / ("cargo.exe" if os.name == "nt" else "cargo")
        if cargo_home.exists():
            os.environ["PATH"] = str(cargo_home.parent) + os.pathsep + os.environ.get("PATH", "")
            cargo_bin = str(cargo_home)
        else:
            print("\033[31m[错误] 未检测到 Rust 编译器 (cargo)。请先访问 https://rustup.rs/ 安装 Rust。\033[0m")
            sys.exit(1)

    print(f"\033[32m[检测] Rust 工具链:\033[0m {cargo_bin}")

    # 检查 maturin
    try:
        import maturin  # noqa: F401
    except ImportError:
        print("\033[33m[提示] 正在安装构建工具 maturin...\033[0m")
        run_cmd([sys.executable, "-m", "pip", "install", "maturin>=1.5"])


def main():
    parser = argparse.ArgumentParser(description="Geatpy 跨平台自动化打包构建工具")
    parser.add_argument("--install", "-i", action="store_true", help="构建后直接安装至当前 Python 环境")
    parser.add_argument("--test", "-t", action="store_true", help="构建并安装后运行 pytest 测试")
    parser.add_argument("--sdist", "-s", action="store_true", help="同时构建源码包 (sdist)")
    parser.add_argument("--debug", "-d", action="store_true", help="以 Debug 模式构建 (默认 Release)")
    parser.add_argument("--target", help="指定编译的目标平台架构 (如 x86_64-pc-windows-msvc, aarch64-apple-darwin)")
    args = parser.parse_args()

    print("=" * 60)
    print("      Geatpy (Rust Core) 跨平台打包与构建工具")
    print(f"      Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"      工作目录: {REPO_ROOT}")
    print("=" * 60)

    ensure_toolchain()

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    build_cmd = [
        sys.executable,
        "-m",
        "maturin",
        "build",
        "-o",
        str(DIST_DIR),
    ]

    if not args.debug:
        build_cmd.append("--release")

    if args.sdist:
        build_cmd.append("--sdist")

    if args.target:
        build_cmd.extend(["--target", args.target])

    print("\n\033[32m[1/3] 开始构建 Geatpy Wheel...\033[0m")
    run_cmd(build_cmd)

    # 查找最新生成的 wheel
    wheels = sorted(DIST_DIR.glob("geatpy-*.whl"), key=os.path.getmtime, reverse=True)
    if not wheels:
        print("\033[31m[错误] 未在 dist 目录找到生成的 wheel 文件\033[0m")
        sys.exit(1)

    latest_wheel = wheels[0]
    print(f"\n\033[32m[2/3] 构建成功! 生成 Wheel:\033[0m {latest_wheel} ({latest_wheel.stat().st_size / 1024:.1f} KB)")

    # 安装
    if args.install or args.test:
        print(f"\n\033[32m[3/3] 安装 Wheel 至当前环境...\033[0m")
        uv_bin = shutil.which("uv")
        if uv_bin:
            run_cmd([uv_bin, "pip", "install", "--reinstall", str(latest_wheel), "--python", sys.executable])
        else:
            run_cmd([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(latest_wheel)])
        print("\033[32m[完成] 安装完成!\033[0m")

    # 测试
    if args.test:
        print("\n\033[32m[测试] 执行 pytest 单元测试套件...\033[0m")
        run_cmd([sys.executable, "-m", "pytest", str(REPO_ROOT / "tests"), "-v"])
        print("\033[32m[完成] 测试全部通过!\033[0m")


if __name__ == "__main__":
    main()
