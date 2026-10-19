# geatpy-rs 2.8.0 性能与 GPU 支持方案

- 状态：**已评审，按本方案实施**。开发分支 `2.8.0-gpu`。
- 来源：2026-10-11 提出、当时搁置的 GPU 方案。评审结论是 GPU 整体收益有限，2.8.0 改为**先把 CPU 内核做快，
  GPU 只做轻量的求值支持**，GPU 算子卸载暂缓（设计保留在附录 A）。
- 基线：geatpy-rs 2.7.3（`3f1b534`）。

## 0. 目标与边界

目标：

1. 修掉 `mutde` 的 O(N²) 实现问题。
2. 恢复多核并行（rayon），覆盖大种群下的热点算子。
3. `evalVars` / `aimFunc` 可以直接返回 PyTorch / CuPy 张量，方便用户把目标函数放到 GPU 上。
4. 与官方 2.7.0 二进制的一致性约定不变：确定性算子并行前后**逐位一致**，随机算子**分布一致**；
   `tests/test_core_parity*.py` 全部通过。
5. 发布形态不变：单一 `cp38-abi3` 轮子，不引入 CUDA 或 torch 依赖。

不做：

- Rust 内核里写 CUDA / wgpu。
- 2.8.0 内的 GPU 算子卸载（附录 A）与种群常驻 GPU 的算法模板（附录 B）。

## 1. 评审依据

### 1.1 端到端剖析（2.7.1 数据，2.7.3 未改变这些热点）

用 cProfile 统计典型算法的耗时分布。测试函数是廉价的 numpy 实现（DTLZ2 M=3 D=12、Rastrigin D=30），
因此内核占比偏高；真实问题的目标函数往往是仿真或模型计算，求值通常占 90% 以上。

| 场景 | 总耗时 | 目标函数求值 | Rust 内核 | Python 其它 | 主要热点 |
| --- | --- | --- | --- | --- | --- |
| NSGA2, N=100, 300 代 | 0.08 s | 12% | 52% | 36% | ndsortESS 24% |
| NSGA2, N=1000, 60 代 | 0.26 s | 2% | 81% | 17% | ndsortESS 66% |
| NSGA2, N=5000, 12 代 | 0.58 s | 1% | 82% | 17% | ndsortESS 74% |
| NSGA3, N=105, 300 代 | 0.10 s | 9% | 70% | 21% | refselect 35% |
| NSGA3, N=1035, 40 代 | 0.48 s | 1% | 92% | 7% | refselect 66% |
| DE/rand/1/bin, N=100, 500 代 | 0.12 s | 14% | 50% | 36% | mutde 32% |
| DE/rand/1/bin, N=10000, 20 代 | 2.74 s | 2% | 96% | 2% | **mutde 93%** |

### 1.2 单算子耗时（2.7.3，单线程，2026-10-19 实测）

ObjV 为 N×3 的近似球面前沿，Chrom 为 N×10 的 RI 染色体：

| N | `ndsortESS` | `mutde` | `cdist(ObjV, ObjV)` |
| --- | --- | --- | --- |
| 1000 | 2.3 ms | 2.4 ms | 8.1 ms |
| 5000 | 46 ms | 33 ms | 142 ms |
| 20000 | 674 ms | 572 ms | 12.9 s |

`cdist` 的 N×N 结果在 N=20000 时有 3.2 GB，耗时主要花在内存上；模板里 `cdist` 多用于 N×R（参考点）的规模，
上表最后一列只说明 N×N 用法本身不可扩展，GPU 也无济于事（结果仍需回传主机）。

### 1.3 结论

- 常规规模（N ≤ 1000）每代内核只有几毫秒，GPU 的启动与传输开销会抵消收益。
- N=5000 时 GPU 卸载 ndsort 估计能让端到端快 3～4 倍；但真实问题中求值占 90% 以上，内核再快，整体最多快约 10%，
  且 Python 层的 17%～36% 开销 GPU 无能为力。
- GPU 求值对性能本身没有贡献：用户今天就可以在 `evalVars` 里自行转换张量；geatpy 能做的是省掉这几行样板代码。
- 确定的收益在 CPU 侧：`mutde` 修复（大种群预计约 100 倍）与多核并行（8～16 核上数倍），对所有用户生效，
  不增加依赖，不需要维护两套实现。

## 2. 工作包

### WP1 `mutde` 的 O(N²) 修复

- 现状：每个个体都构造长度 N 的候选池（`(0..n).filter(..).collect()` + `swap_remove`），整体 O(N²)。
- 改为对 3～5 个下标做拒绝采样：均匀、互不相同、`rows == n` 时排除自身；候选不足（N 很小）时退化为与现在相同的
  有放回抽样（`draw` 在候选池耗尽后的行为）。分布与现实现完全相同，只是随机序列不同。
- 验证：`scripts/parity/fuzz/` 对官方二进制做随机差分；`scripts/parity/classic.py` 中 DE 系列用例 |z| ≤ 3。
- 验收：DE/rand/1/bin N=10000 端到端提速 ≥ 10 倍。

### WP2 rayon 确定性并行

原则：**结果与线程数无关**。确定性算子并行前后逐位一致；随机算子在相同 `np.random.seed` 下，
无论几个线程都得到同样结果。

- 依赖：重新引入 `rayon`；计算期间用 `py.allow_threads` 释放 GIL（种子在释放前取好，`crate::utils::rng()` 需要 GIL）。
- 线程数：默认使用全部核心，可用环境变量 `GEATPY_NUM_THREADS` 或 `ea.set_num_threads(n)` 设置；
  `ea.set_num_threads(1)` 完全回到串行。
- **确定性算子：超过规模阈值时自动并行**，不依赖 `Parallel` 参数（模板调用这些算子时从不传 `Parallel`，
  只认参数的话模板用户拿不到收益；由于结果逐位一致，自动开启不改变任何行为）：

  | 算子 | 并行方式 | 备注 |
  | --- | --- | --- |
  | `refselect` | 个体到参考向量距离与关联按行并行；小生境选择（随机）保持串行 | NSGA-III 热点 |
  | `cdist` | 按输出行并行 | |
  | `ndsortESS` / `ndsortTNS` | ENS-SS 的逐个插入本身是串行的；候选做法：插入时与前沿成员的支配检查按块并行，或 N 很大时改用并行支配计数 | **需实验**，取更快者，结果必须逐位一致 |
  | `crowdis`、`GD` / `IGD` / `Spacing` | 按目标或按行并行 | 视实测收益决定是否做 |

- **随机算子：仅在 `Parallel=True` 时并行**（沿用原版参数语义，默认 False，序列与 2.7.3 相同）。
  并行时按固定大小的行块切分，每块从 `rng()` 依次派生子种子，块大小与线程数无关，保证可复现。
  候选：`mutpolyn`、`mutgau`、`mutbga`、`mutde`、`recsbx`、`recdis`、`mutbin` 等逐行独立的算子；
  这类算子是 O(N·D)，收益有限，只做实测有收益的几个。
- 阈值：每个算子的自动并行阈值由基准确定（小规模时线程调度开销大于收益），写在 Rust 常量里并注释来源。
- 验证：确定性算子在 1 / 2 / 全部线程下与串行逐位比较；随机算子跑 fuzz 与 `classic.py`；
  `AGENTS.md` 中“`Parallel` 目前不生效”一条改为新的语义。

### WP3 GPU 求值支持（轻量，只改 Python 层）

- 新增 `geatpy/_device.py` 的 `to_numpy(x)`：按类型鸭子识别，不主动 import torch / cupy。
  torch 用 `.detach().cpu().numpy()`，cupy 用 `.get()`，CPU 上的 DLPack 对象用 `np.from_dlpack`，其余 `np.asarray`。
- `Problem.evaluation` 在调用 `evalVars` / `aimFunc` 之后，把 `pop.ObjV`、`pop.CV` 中的张量转回 numpy；
  `calReferObjV` 的返回值同样处理。这样 `Algorithm.call_aimFunc` 的格式检查无需改动。
  属于“有意更宽松”（原版报 `ObjV is illegal`），写入 AGENTS.md 差异清单。
- 修正 `Problem.evaluation` 的异常吞没：它用 `except RuntimeError` 判断“子类没有重写 aimFunc”，
  而 torch 的 CUDA 错误（显存不足、设备端断言）也是 `RuntimeError`，会被吞掉并误报
  “aimFunc 和 evalVars 必须至少重写一个”。改为检查方法是否被重写（构造参数传入或子类方法），真实异常原样抛出；
  两者都未重写时，报错信息保持不变。
- 2.8.0 **不加** `Problem(device=...)` 参数：用户在 `evalVars` 开头自行 `torch.as_tensor(Vars, device=...)`，
  一行即可，且 dtype、设备、多卡分配都由用户掌控。是否加该参数视反馈再定。
- 示例：`demo/gpu/`，一个用 PyTorch 写目标函数的单目标问题和一个多目标问题，`device='cpu'` 也能跑。
- 测试 `tests/test_device.py`（无需 GPU）：返回 torch(cpu) 张量、带 `__dlpack__` / `__array__` 的伪数组时的自动转换；
  `aimFunc` 写回张量；`evaluation` 不再吞掉真实 `RuntimeError`；未安装 torch 时 import geatpy 不受影响。
  CI 增加一个可选 job（ubuntu、Python 3.12，安装 CPU 版 torch）；没有 torch 的 job 跳过相关用例。

### WP4 基准、文档与发布

- `scripts/bench/bench_core.py`：复现第 1 节两张表，并对 N ∈ {1k, 5k, 20k} 比较 1 线程与全部线程；
  结果回填本文档，用于定阈值和写 README。
- README / README.en 同步更新：性能数字、多核并行与线程数设置、“目标函数放到 GPU 上”的用法；`history.txt`；
  版本号 2.8.0（commitizen 管理的三处）。
- AGENTS.md：目录加入 `geatpy/_device.py`；更新“`Parallel` 不生效”与“有意更宽松”两条；
  验证表加入“改了并行路径：多线程逐位比较”。

## 3. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 并行改变确定性算子结果 | 只用不改变运算顺序的并行（按行 / 按块，每个输出元素的计算仍串行）；测试逐位比较 |
| 并行破坏随机可复现性 | 种子按固定块派生，与线程数无关；默认 `Parallel=False` 时与 2.7.3 序列相同 |
| 小规模反而变慢 | 每个算子单独设阈值；`ea.set_num_threads(1)` 可完全关闭 |
| 与用户自己的多进程 / 多线程争抢核心 | 提供线程数设置，README 说明 |
| `mutde` 随机序列变化 | 分布不变，fuzz 与 classic 验证；在 history 中说明同种子结果与 2.7.x 不同 |

## 4. 里程碑

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| M1 | WP1 `mutde` 修复 + WP3 GPU 求值 | 全部测试通过；fuzz / classic 无异常；DE N=10000 提速 ≥ 10 倍 |
| M2 | WP2 并行框架 + `refselect` / `cdist` 并行 + 基准脚本 | 多线程逐位一致；NSGA-III 大种群提速数据回填 |
| M3 | WP2 `ndsort` 并行实验 + 随机算子并行；WP4 文档，发布 2.8.0 | 同上；README 双语同步 |

## 附录 A：GPU 算子卸载（暂缓）

重启条件：有用户在 N 上万的规模下使用，且 WP2 并行后内核仍是瓶颈；先在 CUDA 机器上跑基准确认收益。

设计要点（评审前的方案，供重启时参考）：

- `ea.gpu.enable(device='cuda', min_size=None)` 全局开启，默认关闭；在 `geatpy/core/__init__.py` 的分派层，
  只对“规整调用”（二维 float ndarray、参数合法、N ≥ 阈值）走 torch 实现，其余交给 Rust，保证报错和返回类型不变。
- 只卸载确定性的 O(N²) 算子：`ndsortESS` / `ndsortTNS`（分块支配矩阵 + 逐层剥离，fp64 下逐位一致）、
  `refselect` 的关联步骤（随机部分留在 Rust）。`cdist` 的 N×N 结果需回传主机，不卸载。
- MPS 不支持 fp64，只支持 CUDA / ROCm；正确性测试用 torch 的 CPU 设备与 Rust 对比，无卡 CI 也能覆盖。

## 附录 B：种群常驻 GPU 的算法模板（远期预研）

在 `geatpy.gpu` 下提供 NSGA-II / NSGA-III / DE（RI 编码）的 torch 版模板，`evalVars` 直接接收与返回张量，
整代不回主机；参考 EvoX、EvoTorch 的向量化写法。只有 N 上万、目标函数也在 GPU 上时才有明显收益，视需求再定。
