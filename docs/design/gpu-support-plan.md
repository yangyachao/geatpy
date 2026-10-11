# GPU 支持方案（搁置中）

- 状态：**已搁置**，待排期。提出日期 2026-10-11（用户需求：支持 GPU）。
- 适用版本：geatpy-rs 2.7.1 之后。

## 1. 耗时剖析

在 Rust 内核（2.7.1 + master）上用 cProfile 统计典型算法的耗时分布。测试函数是廉价的
numpy 实现（DTLZ2 M=3 D=12、Rastrigin D=30），因此内核占比偏高；真实问题的目标函数往往是仿真
或模型计算，求值通常占 90% 以上。

| 场景 | 总耗时 | 目标函数求值 | Rust 内核 | Python 其它 | 主要热点 |
| --- | --- | --- | --- | --- | --- |
| NSGA2, N=100, 300 代 | 0.08 s | 12% | 52% | 36% | ndsortESS 24% |
| NSGA2, N=1000, 60 代 | 0.26 s | 2% | 81% | 17% | ndsortESS 66% |
| NSGA2, N=5000, 12 代 | 0.58 s | 1% | 82% | 17% | ndsortESS 74% |
| NSGA3, N=105, 300 代 | 0.10 s | 9% | 70% | 21% | refselect 35% |
| NSGA3, N=1035, 40 代 | 0.48 s | 1% | 92% | 7% | refselect 66% |
| DE/rand/1/bin, N=100, 500 代 | 0.12 s | 14% | 50% | 36% | mutde 32% |
| DE/rand/1/bin, N=10000, 20 代 | 2.74 s | 2% | 96% | 2% | **mutde 93%** |

结论：

- 常规规模（N ≤ 1000）下，每代内核耗时只有毫秒级，GPU 的内核启动与 PCIe 传输开销会抵消甚至超过收益。
- 用户“想要 GPU”的真实诉求，多数是**目标函数本身很慢**，应该让求值跑在 GPU 上。
- 大种群下的热点是 O(N²) 算子（非支配排序、参考点关联），以及 `mutde` 的一个实现问题（见第二步）。

## 2. 方案：三步走

### 第一步：GPU 求值支持（收益最大、成本最低，只改 Python 层）

现状障碍：`Algorithm.call_aimFunc` 强制要求 `pop.ObjV` / `pop.CV` 是 `np.ndarray`，
用户在 `evalVars` 里返回 CuPy / PyTorch 数组会直接报错。

改动：

- `Problem` 增加可选参数 `device`（如 `'cuda'`、`'cuda:1'`、`'mps'`）。设置后，传给 `evalVars`
  的 `Vars` 先转成对应设备上的张量（PyTorch 优先，CuPy 兼容），用户可直接写 GPU 代码。
- `ObjV` / `CV` 若为 GPU 数组，自动经 DLPack（`__dlpack__`）或 `.cpu().numpy()` / `.get()`
  转回 numpy，再交给 Rust 内核。
- PyTorch / CuPy 作为可选依赖（`pip install geatpy-rs[gpu]`），未安装时不启用、行为不变。
- Rust 轮子不动，保留 abi3 单一轮子。
- 每代只传输 N×D 决策变量与 N×M 目标值，开销可忽略。

需要的测试：CPU 回退、返回 torch/cupy 张量时的自动转换、`device` 设置后 `Vars` 的类型与设备、
`aimFunc` 路径不受影响。

### 第二步：先把 CPU 内核做快（不涉及 GPU，对大种群最实际）

- **mutde 的 O(N²) 问题**：当前实现为每个个体生成长度为 N 的候选池
  （`(0..n).filter(...).collect()` + `swap_remove`），整体 O(N²)，N=10000 时占总耗时 93%。
  改为拒绝采样：每个个体只抽 3～5 个互不相同、且不等于自身的下标，O(1)。分布不变
  （均匀、无放回、排除自身），预计提速一个数量级以上。
- **恢复多核并行**：对齐原版时移除了 rayon，各算子的 `Parallel` 参数目前不生效。
  可为 ndsortESS、refselect、cdist、逐行变异/重组加回 rayon 并行，并让 `Parallel=True` 生效；
  随机算子需改为每线程独立 RNG。多核 CPU 可提速数倍，没有任何打包负担。

### 第三步：可选的 GPU 内核后端（仅超大种群时值得做）

- 适用：N 上万、目标数较多时，O(N²) 的非支配排序（支配矩阵可广播计算）、参考点关联、cdist。
- 做法：用 PyTorch 写独立的 `geatpy.gpu` 后端，向量化实现 ndsort、crowdis、refselect、SBX、
  多项式变异、DE 等；种群常驻 GPU，不在每代往返传输。PyTorch 同时覆盖 CUDA / ROCm / Apple MPS。
- 与 Rust 内核做差分测试（同 tests/test_core_parity.py 的统计口径），保证语义一致。
- 参考：EvoX（JAX/PyTorch）、EvoTorch 等 GPU 进化计算库的向量化写法。

**不建议**在 Rust 内核里直接写 CUDA（cudarc / wgpu）：需要按 CUDA 版本和平台分别出轮子，
CI 与发布复杂度成倍增加，而该场景收益有限。

## 3. 建议顺序

1. 第二步中的 mutde 修复（改动小、收益确定）。
2. 第一步 GPU 求值支持。
3. 第二步的 rayon 并行。
4. 第三步视真实的超大种群需求再定。

## 4. 复现剖析

剖析脚本思路：对 `ea.optimize` 包一层 `cProfile`，按函数名把耗时归入
“`evalVars`/`aimFunc`”“`_geatpy_core` 内置函数”和“其它”三类；
规模与代数见上表（`logTras=0`，关闭绘图与输出）。
