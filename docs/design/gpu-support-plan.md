# geatpy-rs 2.8.0 性能与 GPU 支持方案

- 状态：**实施中**：WP1、WP2 已完成并验证；WP3、WP4 待做。开发分支 `2.8.0-gpu`。
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

### WP1 `mutde` 的 O(N²) 修复（已完成）

- 原实现：每个个体都构造长度 N 的候选池（`(0..n).filter(..).collect()` + `swap_remove`），整体 O(N²)。
- 现实现：从 0..N 拒绝采样 3～5 个下标，均匀、互不相同、`rows == n` 时排除自身；候选不足（N 很小）时
  与原实现一样退化为有放回抽样。分布与原实现完全相同，只是随机序列不同（同种子结果与 2.7.x 不同）。
  `Mask_N` 矩阵路径同样受益。
- 结果：单次调用 N=5000 从 33 ms 降到 1.5 ms，N=20000 从 572 ms 降到 6.7 ms。
- 验证：`tests/test_core_perf.py` 用“第 k 行取 2^k”的输入精确还原供体下标，检查互不相同、不含自身、均匀分布；
  fuzz（fz.py / fz2.py）对官方二进制的差异报告与 2.7.3 完全相同；`classic.py` 见下文“WP1 + WP2 端到端结果”。

### WP2 rayon 确定性并行（已完成）

原则：**结果与线程数无关**。只拆分与切分方式无关的工作（每个任务算一行输出或一个候选），
归约（求和）在并行算完各项后按原顺序串行完成，所以任何线程数下都与串行逐位一致。

- 基础设施：`crates/geatpy_core/src/parallel.rs`。自建 rayon 线程池（不用全局池，便于改线程数）；
  线程数默认取 `GEATPY_NUM_THREADS`，否则为逻辑 CPU 数；`ea.set_num_threads(n)` / `ea.get_num_threads()`，
  `set_num_threads(1)` 完全串行，`None` / 0 恢复默认。`pool_for(work, min_work)` 按工作量决定是否并行。
- fork 安全：线程池记录创建它的进程号。Linux 上 `multiprocessing` fork 出的子进程继承了线程池却没有工作线程，
  直接使用会永久卡死；子进程发现进程号不同时会新建线程池（继承来的那个直接泄漏）。有测试覆盖，
  且已确认去掉该检查时测试会卡住超时。
- 各算子（实测于 4 核 Xeon 2.1 GHz，与 2.7.3 对比）：

  | 算子 | 做法 | 2.7.3 | 1 线程 | 4 线程 |
  | --- | --- | --- | --- | --- |
  | `ndsortESS` / `TNS` / `DED`（N=20000, M=3） | 见下 | 655 ms | 172 ms | 64 ms |
  | 同上（N=50000, M=3） | | 4.2 s | 0.70 s | 0.28 s |
  | `refselect`（N=1035, M=3） | 关联按行并行；小生境选择改用有序集合与按参考点分组 | 28 ms | 6.9 ms | 2.1 ms |
  | `refselect`（N=10000, M=5） | | 2.7 s | 0.70 s | 0.18 s |
  | `refgselect`（N=10000, R=4950） | 余弦矩阵按行并行；按参考向量分组由每组扫描全体改为一次分桶 | 0.77 s | 0.77 s | 0.42 s |
  | `cdist`（2N×R, N=5050） | 按输出行并行 | 271 ms | 295 ms | 111 ms |
  | `IGD`（N=2000, PF≈10000） | 每行最近距离并行，再按原顺序求和（`GD`、`Spacing` 同） | 78 ms | 78 ms | 21 ms |

  - ndsort：原实现每层对每个未分层个体都扫描它之前的全部个体（O(层数·N²)）。现在每层维护前沿列表，只与前沿成员比较
    （单线程即快 3～6 倍）。并行时按块处理：块内每个个体与“块之前已确定的前沿”及“块内排在它前面的未分层个体”比较；
    由支配关系的传递性，这与串行得到的前沿完全相同（证明写在 `front_levels` 的注释里）。前沿小的层（例如 M=2）
    并行反而更慢，所以每层先串行处理前若干个体估计前沿大小，“预计前沿大小 × 比较的目标数 ≥ 1024”才转为并行。
    计算期间释放 GIL。
  - `cdist` 的 N×N 用法主要耗在分配和写入结果矩阵（内存带宽），并行只有约 2 倍。
- **随机算子保持串行，`Parallel` 参数仍不生效**（与原计划不同）：N=10000、D=30 时 `recsbx`、`mutpolyn`、`mutgau`、
  `recdis`、`mutde` 等单次只要 2～8 ms，并行最多省几毫秒，却要改变随机序列、增加按块派生种子的复杂度，不值得。
- 顺带修复：`to_f64_array2` / `to_i32_array2` 改为总返回行优先数组。此前 Fortran 顺序的输入（例如转置）会让依赖
  连续内存的代码 panic（2.7.3 的 `mutbin` 即如此），新代码的 `cdist` 也依赖这一点。
- 验证：
  - 用 2.7.3 与新版本（1 线程、4 线程）对同一批输入逐位比较，全部一致：ndsort 三个函数 3312 例（含重复行、NaN、inf、
    约束、maxormins、needNum / needLevel，最大 N=20000；其中 18 例含 NaN 的输入两边同样 panic，见下）、
    `refselect` 与 `cdist` 866 例、`GD` / `IGD` / `Spacing` 240 例、`refgselect` 150 例，以及参考点列数与目标数不同的 `refselect`。
  - `tests/test_core_perf.py` 在仓库测试中比较 1 / 3 / 4 线程逐位一致，并覆盖 fork 子进程与 Fortran 顺序输入。
  - ObjV 含 NaN 时 ndsort 会 panic（2.7.3 即有）的问题已在 master 修复（按原版走 numpy unique 与 T-ENS）。合并后含 NaN 的
    输入沿用 master 的串行实现，不含 NaN 的输入走 `front_levels`；`refselect` / `refgselect` 的关联改用 master 复现原版
    舍入的算术（行归一化 + FMA 点积），仍按行并行。合并后与 master 构建逐位比较（1 / 4 线程）全部一致：NaN、整数并列、
    重复极点、重复参考向量 720 例，`refselect` 与 `cdist` 866 例，指标 240 例，`refgselect` 150 例。
  - master 为复现原版舍入改用 FMA 点积（`mul_add`），而 x86_64 轮子不能假定 CPU 有 FMA，每次 `mul_add` 都是一次库函数调用，
    合并后 `refselect` 单线程一度慢了约 3 倍。`utils::dot_fma` 改为运行时检测 FMA 并使用以 `fma` 特性编译的循环
    （已同时修复到 master）；FMA 软硬件都只舍入一次，结果不变（与改动前的构建逐位比较一致）。上表 `refselect` /
    `refgselect` 的新版本数字均在合并 master 并加上该修复之后测得。

### WP1 + WP2 端到端结果

`scripts/bench/bench_core.py`，4 核 Xeon 2.1 GHz，总耗时在 cProfile 下测得（Python 部分会被放慢），单位秒：

| 场景 | 2.7.3 | 1 线程 | 4 线程 |
| --- | --- | --- | --- |
| NSGA2, N=1000, 60 代 | 0.69 | 0.39 | 0.30 |
| NSGA2, N=5000, 12 代 | 1.08 | 0.55 | 0.48 |
| NSGA3, N=1035, 40 代 | 0.89 | 0.59 | 0.28 |
| NSGA3, N=5050, 10 代 | 3.56 | 1.42 | 0.54 |
| DE/rand/1/bin, N=10000, 20 代 | 3.36 | 0.60 | 0.60 |

N ≈ 100 的场景（NSGA2 / NSGA3 / DE，约 0.2 s）三者持平，差异在计时波动（约 ±30%）之内；这些规模低于所有并行阈值。
NSGA2 N=5000 之后的瓶颈在 Python 层（`other` 约 55%），内核已不是主要耗时。

`scripts/parity/classic.py`（36 个用例，两边各 30 次，2.7.3 对新版本单线程）：没有 |z| > 3 的用例。只用到确定性算子与
未改动随机算子的用例两边结果完全相同（z = 0）；用到 `mutde` 的 DE 类用例随机序列不同，|z| 最大 1.33。

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
