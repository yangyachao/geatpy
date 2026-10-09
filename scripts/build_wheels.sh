#!/usr/bin/env bash
# ==============================================================================
# Geatpy 多平台 / 发布级 Wheel 生产打包脚本
# 支持生成 manylinux、macOS、Windows 的 abi3 通用二进制 Wheel 及 sdist 源码包
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"

cd "${REPO_ROOT}"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}        Geatpy 发布级多架构 Wheel 打包工具            ${NC}"
echo -e "${BLUE}======================================================${NC}"

# 环境准备
if ! command -v cargo &> /dev/null && [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

PYTHON_BIN="${PYTHON:-python3}"
if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

mkdir -p "${DIST_DIR}"

# 1. 构建标准当前平台 Release Wheel
echo -e "${GREEN}[1/3] 构建宿主系统原生 ABI3 Release Wheel...${NC}"
"${PYTHON_BIN}" -m maturin build --release -o "${DIST_DIR}"

# 2. 构建源码分发包 (sdist)
echo -e "${GREEN}[2/3] 构建源码分发包 (sdist .tar.gz)...${NC}"
"${PYTHON_BIN}" -m maturin sdist -o "${DIST_DIR}"

# 3. 检查与汇总已生成分发文件
echo -e "${GREEN}[3/3] 打包汇总与完整性校验:${NC}"
echo -e "${BLUE}产物输出目录: ${DIST_DIR}${NC}"
ls -lh "${DIST_DIR}"

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}打包全部完成!${NC}"
echo -e "可使用以下命令上传至 PyPI / 私有仓库:"
echo -e "  twine upload dist/*"
echo -e "${GREEN}======================================================${NC}"
