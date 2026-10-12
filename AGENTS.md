# geatpy-rs 协作约定

geatpy 2.7.0 的 Python 层原样保留，闭源 Cython 内核 `_core` 用 Rust 重写（`crates/geatpy_core`，PyO3 abi3，
单一 `cp38-abi3` 轮子）。目标是**与官方 2.7.0 二进制行为一致**：同样的输入得到同样的输出、同样的返回类型、
同样的报错；随机算子则分布一致。PyPI 包名 `geatpy-rs`，导入名仍是 `geatpy`。

## 目录

- `crates/geatpy_core/src/`：Rust 内核。`lib.rs` 注册全部导出函数；`encoding.rs`（crt*、bs2*、ri2bs、boundfix）、
  `selection.rs`（选择、ranking/scaling/powing/indexing、otos、ecs）、`recombination.rs`（rec*、xov*、recombin）、`mutation.rs`（mut*、mutate）、
  `multiobjective.rs`（ndsort*、crowdis、cdist、mergecv、tcheby、pbi、awGA、rwGA、crtidp、refselect、refgselect、
  migrate）、`indicator.rs`（GD、IGD、Spacing、HV）、`utils.rs`（参数解析、FixType 修复、int32 输出）。
- `geatpy/`：原版 Python 层（算法模板、Problem、Population 等），一般不改。
- `tests/`：`test_core_parity.py` 与官方二进制的黑盒差分（期望值在 `tests/fixtures/core_parity_orig_2_7_0.json`）；
  `test_core_parity_review.py` 逆向复核得到的回归用例。
- `reverse/`（已 gitignore，只在本机）：官方 `_core` 的反编译结果，仅供审查。
  - `reverse/compact/core/<模块>.c`：该模块的纯 C 核心函数（算法主体，优先读这个）；
  - `reverse/compact/outline/<模块>.txt`：Cython Python 层（`__pyx_pf_`）按顺序出现的名字、常量与运算，
    用于确认参数默认值、校验、返回类型；
  - `reverse/compact/with_pf/<模块>.c`：纯 C 核心函数加 Cython `__pyx_pf_` 包装函数的反编译正文（outline 不够时查参数处理细节）；
  - `reverse/decompiled/<模块>.c`：完整反编译（很大，只在上面几份不够时查具体函数）；
  - `reverse/REVIEW_FINDINGS.md`：上一轮审查报告。

## 已知且有意保留的差异

以下差异已确认，审查时不必再报（除非发现记录本身有误）：

- 原版自身缺陷、Rust 按文档实现：`ndsortDED` 分层错误、`mergecv` 跨行累加、`tcheby`/`pbi` 在 `idealPoint=None`
  时返回理想点、`refselect` 候选矩阵行序错位及无需选择时返回 arange、`migrate` 的 `Replacement=2` 淘汰最优个体、
  `mutgau`/`mutuni` 的 `Middle=True` 以 `(ub-lb)/2` 为中心、`crtip` 边界为小数时越界。
- 原版缺陷（第三轮差分确认）：`migrate` 的 `Replacement=1` 总淘汰最后一个个体、`Structure=0` 迁入来源含自身；
  `pbi` 个体在理想点时为 NaN；`crowdis` 含 inf 时为 NaN；`cdist` 对非连续数组读错；长度为 1 或一维输入时段错误；
  HV 内部 float32；`sus` 二进制要求至少 3 个参数。
- `rws` 先把 FitnV 平移到非负；`crowdis` 接受 `levels=None`；HV 在 M=4 时用精确算法（原版蒙特卡洛）。
- 有意更宽松（原版报错）：`maxormins` 接受 list；'P' 编码的 mutinv/mutmove/mutswap 不强制 FieldDR；一维输入；
  整数参数（Half_N、Nind 等）接受 numpy 整数（原版只认 Python int）。
- 整数输出一律 int32（与原版一致）；例外是原版用 numpy 实现的 `crtbp`、`urs`、`ecs`、Nsel = N 的 `otos`（int64），
  以及长度 < 2 的排列染色体原样返回。
- 并行参数 `Parallel` 目前不生效（为与原版随机序列语义对齐移除了 rayon）。

## 验证

| 改动范围 | 验证要求 |
| --- | --- |
| Rust 内核 | `maturin develop --release` 后 `python -m pytest tests -q`；`cargo fmt --check`；`cargo clippy -- -D warnings` |
| 有意改变输出 | 在 Docker 中用官方二进制重新生成黄金文件（`scripts/parity/run_original_core.sh`），提交说明写明原因 |
| 改了算子行为或校验 | 用 `scripts/parity/fuzz/` 对官方二进制做随机差分（见其 README，官方 macOS arm64 轮子可在本机原生运行） |
| 改了随机算子或性能路径 | 另跑 `scripts/parity/classic.py`（36 个经典用例，两边各 30 次，`cmp` 中 \|z\| > 3 需复查；`rastrigin_multi_SEGA` 因 migrate 缺陷修正而显著更好，属预期） |
| 仅文档 | 检查内容与引用路径 |

- 随机数一律用 `crate::utils::rng()`（每次调用从 numpy 全局 RandomState 取种子，`np.random.seed` 才能复现），不要用 `thread_rng`。
- 性能优化只改实现、不改算法：每个基因/个体的概率与原版相同（例如 `mutbin` 用几何分布跳跃抽取变异位），改后用 fuzz 与
  `scripts/parity/classic.py` 确认分布与解质量不变。
- 提交信息用英文 Conventional Commits；README 与设计文档用中文；Rust 注释用英文。
- `reverse/` 下的反编译内容不得提交或发布。
