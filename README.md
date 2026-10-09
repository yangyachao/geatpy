# **Geatpy (Rust Core Edition)**
The High-Performance Genetic and Evolutionary Algorithm Toolbox for Python, Powered by Rust.

[![CI](https://github.com/yangyachao/geatpy/actions/workflows/ci.yml/badge.svg)](https://github.com/yangyachao/geatpy/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/yangyachao/geatpy?color=blue&logo=github)](https://github.com/yangyachao/geatpy/releases)
[![Python Version](https://img.shields.io/badge/python-3.8_~_3.14+-blue.svg?logo=python&logoColor=white)](https://pypi.org/project/geatpy/)
[![Rust](https://img.shields.io/badge/rust-PyO3_abi3-orange.svg?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-LGPL--3.0-green.svg)](LICENSE)

---

## 📌 项目渊源与致敬 (Upstream Heritage & Fork Notice)

本项目是经典高性能进化算法工具箱 **[geatpy-dev/geatpy](https://github.com/geatpy-dev/geatpy)** 的现代化高性能 Rust 重构版本。

- **原官方项目主页**: [https://github.com/geatpy-dev/geatpy](https://github.com/geatpy-dev/geatpy)
- **官方文档与教程**: [http://www.geatpy.com](http://www.geatpy.com) / [Geatpy Documentation](https://github.com/geatpy-dev/geatpy/tree/master/docs)
- **开源致谢**: 衷心感谢原作者团队在进化计算领域的杰出贡献与算法设计奠基。

### 为什么发起这个重构版本？(Motivation)
原版 Geatpy 长期基于特定版本的 C/C++ 动态链接库，在现代开发环境中面临诸多痛点：
1. **高版本 Python 无法安装**: 官方二进制包仅支持至 Python 3.10，在 Python 3.11、3.12、3.13 及最新 3.14+ 上由于 C API 变动无法编译或运行。
2. **跨平台支持弱**: 缺乏对现代化 macOS (Apple Silicon M1/M2/M3/M4 ARM64) 和 Linux ARM64 的预编译 Wheel 支持。
3. **C 内核维护门槛高**: C 扩展缺乏现代包管理机制，在各操作系统本地编译时容易遇到编译链缺失与字符集编码异常。

针对以上行业痛点，**本项目使用 Rust (PyO3 + abi3) 对底层核心库进行了 100% 全量重写**，实现了全平台预编译二进制轮子与全版本 Python 开箱即用。

---

## ⚡ 核心改进与特性对比 (Key Improvements)

| 特性维度 | 原官方版本 (Geatpy) | 本版本 (Geatpy Rust Core Edition) |
| :--- | :--- | :--- |
| **底层核心实现** | 早期 C/C++ 动态库 (部分二进制分发) | **100% 现代纯 Rust 原生实现** (`crates/geatpy_core`) |
| **Python 版本支持** | 仅支持 Python 3.5 ~ 3.10 | **Python 3.8 ~ 3.14+ 全覆盖支持** |
| **ABI 兼容性** | 每个 Python 版本需分别单独编译 | **PyO3 `abi3` 单一构建兼容所有 Python 3.8+ 版本** |
| **跨平台预编译** | Windows x64, Linux x64 (Mac 支持有限) | **全平台官方预编译 Wheel (Linux x86_64/ARM64, macOS Universal2, Windows x64)** |
| **代码与内存安全** | 存在原生裸指针与越界崩溃风险 | **Rust 强类型、严格边界检查与所有权保障，零段错误 (Zero Segfault)** |
| **多核并行与性能** | C OpenMP 并行 | **Rayon 多核工作窃取调度与极致向量化加速** |
| **API 兼容度** | 原版基准 | **100% 完全兼容原有 Python API、数据结构与算法模板，代码无缝平替** |

---

## 📦 安装指南 (Installation)

### 方式一：通过 pip 一键安装 (PyPI 推荐)

```bash
pip install geatpy-rs
```

> 💡 **提示**：分发包名为 `geatpy-rs`，在 Python 代码中完全无需改动，继续沿用熟悉的 **`import geatpy as ea`** 即可无缝调用！

### 方式二：从 GitHub Releases 下载预编译 Wheel

前往 [Releases 页面](https://github.com/yangyachao/geatpy/releases) 下载对应系统的 Wheel 文件，或通过链接一键安装：

```bash
# Linux (x86_64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.0/geatpy_rs-2.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Linux (ARM64 / aarch64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.0/geatpy_rs-2.7.0-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# macOS (Universal2 - 支持 Apple Silicon M1/M2/M3/M4 以及 Intel Mac)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.0/geatpy_rs-2.7.0-cp38-abi3-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl

# Windows (x86_64)
pip install https://github.com/yangyachao/geatpy/releases/download/v2.7.0/geatpy_rs-2.7.0-cp38-abi3-win_amd64.whl
```

### 方式三：从源码本地构建

克隆仓库后，仅需 Rust 工具链与 `maturin` 即可一键构建安装：

```bash
git clone https://github.com/yangyachao/geatpy.git
cd geatpy

# 使用内置跨平台构建脚本
python scripts/build.py --install

# 或使用 maturin 直接安装至当前环境
pip install maturin
maturin develop --release
```

详细本地编译、跨平台交叉编译与单元测试说明，请参阅 [BUILD.md](BUILD.md)。

---

## 🚀 快速上手 (Quick Start)

原有代码和案例库无需进行任何修改，直接沿用原有接口即可：

### 示例：使用 NSGA-III 求解经典 DTLZ1 三目标优化问题

```python
import numpy as np
import geatpy as ea

class DTLZ1(ea.Problem):
    def __init__(self, M=3):
        name = 'DTLZ1'
        maxormins = [1] * M
        Dim = M + 4
        varTypes = [0] * Dim
        lb = [0] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        Vars = pop.Phen
        XM = Vars[:, (self.M - 1):]
        g = np.array([100 * (self.Dim - self.M + 1 + np.sum(((XM - 0.5)**2 - np.cos(20 * np.pi * (XM - 0.5))), 1))]).T
        ones_metrix = np.ones((Vars.shape[0], 1))
        pop.ObjV = 0.5 * np.fliplr(np.cumprod(np.hstack([ones_metrix, Vars[:, :self.M - 1]]), 1)) * \
                   np.hstack([ones_metrix, 1 - Vars[:, range(self.M - 2, -1, -1)]]) * np.tile(1 + g, (1, self.M))

if __name__ == '__main__':
    problem = DTLZ1(M=3)
    algorithm = ea.moea_NSGA3_templet(
        problem,
        ea.Population(Encoding='RI', NIND=100),
        MAXGEN=500,
        logTras=1
    )
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=True)
```

---

## 📊 架构设计与底层算子覆盖

Rust 核心库位于 `crates/geatpy_core/`，涵盖了进化算法全套 70+ 个底层算子：

- **编码映射 (`encoding.rs`)**: `bin2dec`, `bin2gray`, `gray2bin`, `dec2bin`, `crtip`, `crtrp`, `crtbp`, `crtup`, `crtsr`...
- **选择算子 (`selection.rs`)**: `tour`, `rws`, `sus`, `etour`, `urs`, `recsrr`, `rwGA`, `awGA`...
- **交叉与重组 (`recombination.rs`)**: `xovsp`, `xovdp`, `xovmp`, `xovsh`, `xovdp_templet`, `recdis`, `recint`, `reclin`, `recmut`, `rechex`...
- **变异算子 (`mutation.rs`)**: `mutbin`, `mutbga`, `mutgau`, `mutmar`, `mutpol`, `mutuni`, `mutswap`, `mutmove`, `mutinv`...
- **多目标排序与小生境 (`multiobjective.rs`)**: 非支配快速排序 (`NDSort`), 拥挤距离计算 (`crowddis`), 基于参考点的临界前沿关联选择 (`refselect`)...
- **性能评估与指标 (`indicator.rs`)**: 世代距离 (`gd`), 反世代距离 (`igd`), 超体积指标 (`hv`), 距离测度矩阵 (`cdist`)...
- **子种群迁移 (`utils.rs`)**: 环形、中心、全联通拓扑结构种群迁移算子 (`migrate`)。

---

## 📄 引用 (Citation) & 许可证 (License)

如果您在学术研究中使用了本项目，请按照原项目规范引用：

```bibtex
@misc{geatpy2020,
  author = {Jazzbin, J.},
  title = {Geatpy: The Genetic and Evolutionary Algorithm Toolbox with High Performance in Python},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/geatpy-dev/geatpy}}
}
```

本项目继承原项目协议，采用 **[GNU Lesser General Public License v3.0 (LGPL-3.0)](LICENSE)** 开源。
