#!/usr/bin/env bash
# ==============================================================================
# Geatpy (Rust Core) 一键本地构建与打包脚本
# 支持 Linux / macOS
# ==============================================================================

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"

cd "${REPO_ROOT}"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}        Geatpy (Rust Core) 本地构建与打包工具          ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 参数解析
INSTALL_FLAG=false
TEST_FLAG=false
SDIST_FLAG=false
RELEASE_FLAG=true

for arg in "$@"; do
    case "$arg" in
        --install|-i)
            INSTALL_FLAG=true
            ;;
        --test|-t)
            TEST_FLAG=true
            INSTALL_FLAG=true
            ;;
        --sdist|-s)
            SDIST_FLAG=true
            ;;
        --debug|-d)
            RELEASE_FLAG=false
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --install, -i   构建完成后安装至当前 Python 环境"
            echo "  --test, -t      构建、安装并运行 pytest 单元测试"
            echo "  --sdist, -s     额外构建 Python 源码分发包 (sdist)"
            echo "  --debug, -d     使用 Debug 模式构建 (默认使用 Release 优化)"
            echo "  --help, -h      显示本帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $arg${NC}"
            exit 1
            ;;
    esac
done

# 1. 检查 Rust 环境
if ! command -v cargo &> /dev/null; then
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    else
        echo -e "${RED}[错误] 未检测到 Rust/cargo，请先安装 Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}[1/5] Rust 工具链检测成功:${NC} $(cargo --version)"

# 2. 检查 Python 与 maturin
PYTHON_BIN="${PYTHON:-python3}"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

echo -e "${GREEN}[2/5] 使用 Python 环境:${NC} $("${PYTHON_BIN}" --version) (${PYTHON_BIN})"

if ! command -v maturin &> /dev/null && ! "${PYTHON_BIN}" -m maturin --version &> /dev/null; then
    echo -e "${YELLOW}[提示] 未检测到 maturin，正在通过 pip 自动安装...${NC}"
    "${PYTHON_BIN}" -m pip install "maturin>=1.5"
fi

MATURIN_CMD="maturin"
if ! command -v maturin &> /dev/null; then
    MATURIN_CMD="${PYTHON_BIN} -m maturin"
fi

# 3. 准备输出目录
mkdir -p "${DIST_DIR}"

# 4. 执行打包构建
BUILD_OPTS=("-o" "${DIST_DIR}")
if [ "${RELEASE_FLAG}" = true ]; then
    BUILD_OPTS+=("--release")
fi
if [ "${SDIST_FLAG}" = true ]; then
    BUILD_OPTS+=("--sdist")
fi

echo -e "${GREEN}[3/5] 开始构建 Geatpy (包含 Rust 原生核心)...${NC}"
${MATURIN_CMD} build "${BUILD_OPTS[@]}"

LATEST_WHEEL=$(ls -t "${DIST_DIR}"/geatpy*.whl 2>/dev/null | head -n 1)
echo -e "${GREEN}[4/5] 构建成功! 生成 Wheel 文件:${NC}"
ls -lh "${LATEST_WHEEL}"

# 5. 可选安装与测试
if [ "${INSTALL_FLAG}" = true ]; then
    echo -e "${BLUE}--> 正在安装最新构建的 Wheel 至当前 Python 环境...${NC}"
    if command -v uv &> /dev/null; then
        uv pip install --reinstall "${LATEST_WHEEL}" --python "${PYTHON_BIN}"
    else
        "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps "${LATEST_WHEEL}"
    fi
    echo -e "${GREEN}--> 安装完成!${NC}"
fi

if [ "${TEST_FLAG}" = true ]; then
    echo -e "${BLUE}--> 运行 pytest 单元测试套件...${NC}"
    "${PYTHON_BIN}" -m pytest "${REPO_ROOT}/tests" -v
    echo -e "${GREEN}--> 单元测试全部通过!${NC}"
fi

echo -e "${GREEN}[5/5] 打包流程顺利完成! 输出目录: ${DIST_DIR}${NC}"
