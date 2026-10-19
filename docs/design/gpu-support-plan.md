# geatpy-rs 2.8.0 GPU 加速方案

- 状态：**规划中，待评审**。开发分支 `2.8.0-gpu`。
- 来源：2026-10-11 提出、当时搁置的 GPU 方案（第 1 节剖析数据沿用），本版细化为 2.8.0 的可执行计划。
- 基线：geatpy-rs 2.7.3（`3f1b534`）。

## 0. 目标与边界

目标：

1. 用户可以把**目标函数求值**放到 GPU 上（PyTorch / CuPy），无需自己写转换代码。
2. 大种群（N 数千以上）时，**O(N²) 的确定性算子**可选择卸载到 GPU。
3. **默认行为不变**：不安装 torch、或不开启 GPU 时，输出与 2.7.3 完全一致，
   `tests/test_core_parity*.py` 照旧全部通过；与官方 2.7.0 二进制的一致性约定不受影响。
4. 发布形态不变：仍是单一 `cp38-abi3` 轮子，Rust 内核不引入任何 CUDA 构建依赖。

不做：

- 在 Rust 内核里写 CUDA / wgpu（需按 CUDA 版本和平台分别出轮子，CI 与发布成本成倍增加，收益有限）。
- 全部算子、全部编码的 GPU 版本；变异/重组这类 O(N·D) 的访存型算子在 GPU 上收益抵不过传输。
- 种群常驻 GPU 的新算法模板（列入 2.9 预研，见第 6 节）。

## 1. 耗时剖析（沿用 2.7.1 数据）

用 cProfile 统计典型算法的耗时分布。测试函数是廉价的 numpy 实现（DTLZ2 M=3 D=12、Rastrigin D=30），
因此内核占比偏高；真实问题的目标函数往往是仿真或模型计算，求值通常占 90% 以上。
2.7.3 的性能改动只涉及二进制/排列算子，下表热点未变。

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

- 常规规模（N ≤ 1000）下每代内核耗时只有毫秒级，GPU 的启动与 PCIe 传输开销会抵消收益。
- 用户想要 GPU 的真实诉求多数是**目标函数本身很慢**，应让求值跑在 GPU 上（WP1，收益最大）。
- 大种群下的热点是 O(N²) 算子（非支配排序、参考点关联、距离矩阵）（WP3），
  以及 `mutde` 的一个实现问题（WP2，纯 CPU 修复）。

## 2. 总体架构

```
用户 evalVars(Vars) ── Vars 为 torch/cupy 张量 ──┐
                                                  │  geatpy/_device.py（设备解析、张量 <-> numpy 转换，懒加载 torch/cupy）
Problem.evaluation ──── ObjV/CV 自动转回 numpy ───┘
        │
Algorithm 模板（原样）── ea.ndsortESS / ea.cdist / ea.refselect ...
        │
geatpy/core/__init__.py 分派层：GPU 已开启 且 输入规整 且 N ≥ 阈值 ？ ──是──> geatpy/gpu/（torch 实现）
                                                                └──否──> _geatpy_core（Rust，原样）
```

两个开关相互独立：

- **求值设备**属于问题：`ea.Problem(..., device='cuda')`。
- **算子卸载**是全局配置：`ea.gpu.enable(device='cuda', min_size=None)` / `ea.gpu.disable()` / `ea.gpu.status()`，
  也可用环境变量 `GEATPY_GPU=cuda:0` 开启。默认关闭。

## 3. 工作包

### WP1 设备转换层 + GPU 求值（只改 Python 层）

现状障碍：`Algorithm.call_aimFunc` 要求 `pop.ObjV` / `pop.CV` 是 `np.ndarray`，
`evalVars` 返回 torch / cupy 张量会直接报 `ObjV is illegal`。

改动：

- 新增 `geatpy/_device.py`：
  - `resolve_device(spec)`：解析 `'cuda'`、`'cuda:1'`、`'mps'`、`'cpu'`，以及 `'cupy'` / `'cupy:1'`；
    后端未安装或设备不可用时报清晰的错误。
  - `to_device(arr, device, dtype=None)`：numpy → torch/cupy 张量。
  - `to_numpy(x)`：按类型鸭子识别（不主动 import torch/cupy）：torch 用 `.detach().cpu().numpy()`，
    cupy 用 `.get()`，CPU 上的 DLPack 对象用 `np.from_dlpack`，其余 `np.asarray`；结果统一为 float64 二维数组。
- `Problem.__init__` 末尾增加仅关键字参数 `device=None, dtype=None`（现有子类按位置调用父类构造，不受影响）：
  - 设置 `device` 后，`evaluation` 把 `pop.Phen` 转成该设备上的张量再传给 `evalVars`；
  - `dtype` 默认保持 float64；可设 `'float32'`（消费级显卡 fp64 吞吐只有 fp32 的 1/32～1/64，Apple MPS 不支持 fp64）；
  - `aimFunc(pop)` 路径不改 `pop.Phen` 的类型（避免破坏 Population 语义），用户自行转换；
  - **无论是否设置 `device`**，`evalVars` / `aimFunc` 写回的 `ObjV`、`CV` 若是张量都自动转回 numpy
    （属于“有意更宽松”，写入 AGENTS.md 的差异清单）；`calReferObjV` 的返回值同样处理。
- 顺带修正 `Problem.evaluation` 的异常吞没：它用 `except RuntimeError` 判断“子类没有重写 aimFunc”，
  而 torch 的 CUDA 错误（显存不足、设备端断言）也是 `RuntimeError`，会被吞掉并误报
  “aimFunc 和 evalVars 必须至少重写一个”。改为检查方法是否被重写（实例属性或子类方法），真实异常原样抛出。
  对未重写两者的情况，报错信息保持不变。
- 示例：`demo/gpu/`，一个 PyTorch 写的大规模单目标问题和一个多目标问题，CPU 可跑（`device='cpu'`）。

### WP2 CPU 前置修复（不涉及 GPU）

- **mutde 的 O(N²)**：当前每个个体都构造长度 N 的候选池（`(0..n).filter(..).collect()` + `swap_remove`），
  N=10000 时占总耗时 93%。改为对 3～5 个下标做拒绝采样：均匀、互不相同、`rows == n` 时排除自身；
  候选不足（N 很小）时退化为与现在相同的有放回抽样。分布与现实现完全相同，只是随机序列不同。
  按 AGENTS.md 用 `scripts/parity/fuzz/` 与 `scripts/parity/classic.py` 验证分布与解质量。
- **rayon 确定性并行**（可选，视排期放 2.8.0 或 2.8.x）：让 `Parallel=True` 生效，默认仍为 False。
  随机算子由 `crate::utils::rng()` 为每个固定大小的数据块派生种子，结果与线程数无关、`np.random.seed` 仍可复现；
  确定性算子（ndsort、cdist、refselect 的关联步骤）并行前后逐位一致。

### WP3 GPU 算子卸载（opt-in，torch 实现）

只卸载 O(N²) 且结果确定的算子：

| 算子 | GPU 做法 | 与 Rust 的一致性 |
| --- | --- | --- |
| `ndsortESS` / `ndsortTNS` | 分块广播计算支配矩阵（含 CV 约束支配、maxormins），按入度逐层剥离；支持 `needNum` / `needLevel` 提前停止，未分层个体为 inf | 只有比较运算，fp64 下**逐位一致** |
| `cdist` | `'euclidean'` 用逐差平方分块计算（不用 `torch.cdist` 的 matmul 展开，避免抵消误差）；`'cosine'` 系列直接矩阵乘 | 相对误差 ≤ 1e-12，测试按容差比较 |
| `refselect` | 只把“个体到参考向量的垂直距离与关联”放到 GPU；带随机性的小生境选择仍在 Rust | 关联在距离几乎相等时可能因舍入不同而翻转，按分布一致验收 |

`refselect` 需要 Rust 侧拆出一个内部函数（如 `_refselect_niche`，接收关联结果，不对外文档化），其余算子 Rust 不动。
`ndsortDED`、`crowdis`、变异/重组/选择均不卸载。

分派规则（`geatpy/core/__init__.py`，在 `geatpy/__init__.py` 的 `from geatpy.core.X import X` 之前装好）：

- 只有“规整调用”才卸载：ObjV 为二维 float ndarray、参数类型合法、N ≥ 阈值；
  其它一切情况（list 输入、一维、非法参数等）交给 Rust，保证报错与返回类型和现在完全一致。
- 返回类型与 Rust 相同（levels 为 float64 ndarray，maxLevel 为 int 等）。
- 阈值 `min_size` 默认值由基准测试确定（预估 ndsort 在 2000～5000 之间），用户可覆盖。
- torch 未安装 / 设备不可用：`enable()` 时报错；环境变量开启时警告一次并回退 CPU。
- 支配矩阵按显存上限分块（默认单块 ≤ 256 MB），N 很大时不爆显存。
- MPS 不支持 fp64：算子卸载只支持 CUDA / ROCm；MPS 仅用于 WP1 求值。

### WP4 测试与基准

- `tests/test_device.py`：不需要 GPU。覆盖 CPU 回退、torch(cpu) 张量与带 `__dlpack__` / `__array__`
  的伪数组自动转换、`device` / `dtype` 设置后 `Vars` 的类型与设备、`aimFunc` 路径不变、
  `evaluation` 不再吞掉真实 `RuntimeError`。
- `tests/test_gpu_ops.py`：用 **torch 的 CPU 设备**跑 WP3 实现并与 Rust 对比（ndsort 逐位一致、cdist 容差、
  refselect 统计口径同 `test_core_parity.py`），无 GPU 的 CI 也能覆盖正确性；`cuda` 用例加 `pytest.mark.gpu`，无卡时跳过。
- CI：ci.yml 增加一个可选 job（ubuntu、Python 3.12，安装 CPU 版 torch）跑上述两组测试；原测试矩阵不变。
- `scripts/bench/gpu_bench.py`：复现第 1 节表格，并对 N ∈ {1k, 5k, 20k, 50k} 比较 CPU 与 GPU 的算子耗时与端到端耗时；
  在 CUDA 机器上手动运行，结果回填本文档，用于确定默认阈值。
- 原有 parity 测试、fuzz、`classic.py` 在默认（GPU 关闭）配置下必须不变。

### WP5 文档与发布

- README / README.en 增加“GPU 加速”一节（两者同步）；`history.txt`；版本号 2.8.0（commitizen 管理的三处）。
- 可选依赖：`geatpy-rs[torch]`（带 `python_version >= "3.9"` 标记，torch 已不支持 3.8）；
  CuPy 不设 extra，由用户按 CUDA 版本安装 `cupy-cuda12x` 等。
- AGENTS.md：目录加入 `geatpy/_device.py`、`geatpy/gpu/`；验证表加入“改了 GPU 路径”一行；
  “有意更宽松”加入 ObjV/CV 接受张量。

## 4. 兼容性与风险

| 风险 | 对策 |
| --- | --- |
| GPU 路径改变默认输出 | 默认关闭；关闭时不 import torch，不经过新代码 |
| GPU 与 CPU 结果不一致 | 只卸载确定性算子；ndsort 逐位一致，cdist 设容差，refselect 统计验收 |
| 随机可复现性 | GPU 算子不消耗随机数；refselect 的随机部分留在 Rust，仍由 `np.random.seed` 控制 |
| 小种群反而变慢 | 阈值分派，小于阈值走 Rust |
| 显存不足 | 分块；OOM 时报错（不再被 evaluation 吞掉） |
| fp32 / MPS 精度 | 求值可用 fp32（用户显式选择）；算子卸载只用 fp64 |
| Python 3.8 | GPU 功能要求 3.9+，3.8 下核心功能照旧 |
| torch 导入耗时 | 懒加载，只在设置 `device` 或 `enable()` 时导入 |

## 5. 里程碑

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| M1 | WP2 mutde 修复 + WP1 求值 + 示例 | 全部测试通过；fuzz / classic 无异常；DE N=10000 提速 ≥ 10× |
| M2 | WP3 ndsort、cdist + 分派层 + 基准脚本 | torch(cpu) 对比测试通过；CUDA 机器上的基准数据回填，确定阈值 |
| M3 | WP3 refselect、WP4 CI job、WP5 文档，发布 2.8.0 | 同上；README 双语同步 |
| 可选 | WP2 rayon 并行 | 并行前后确定性算子逐位一致，随机算子 fuzz / classic 通过 |

## 6. 后续（2.9 预研）

- 种群常驻 GPU 的独立模板（`geatpy.gpu` 下的 NSGA-II / NSGA-III / DE，RI 编码），用户的 `evalVars`
  直接接收与返回张量，整代不回主机；参考 EvoX、EvoTorch 的向量化写法。
- 是否做取决于 2.8.0 的使用反馈与基准：只有 N 上万、目标函数也在 GPU 上时才有明显收益。

## 7. 待确认

1. 2.8.0 范围：建议 WP1 + WP2(mutde) + WP3（ndsort、cdist 必做，refselect 视进度）；rayon 是否纳入。
2. 算子后端只用 PyTorch（建议）；CuPy 只支持求值。
3. API 命名：`Problem(device=, dtype=)` 与 `ea.gpu.enable()`。
