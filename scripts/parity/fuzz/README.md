# 随机差分测试

`fz.py`（约 1300 个用例）和 `fz2.py`（约 200 个用例：全局重组模式、默认参数、参数校验）在同样的输入上分别调用
官方 geatpy 2.7.0 内核与 Rust 内核，`cmp.py` 比对结果：

- 确定性用例逐值（相对误差 1e-9）比对数值、dtype、形状、返回结构，以及是否报错；
- 随机算子重复调用 300–2000 次，逐元素均值做 z 检验（默认 |z| > 6 判为差异）；
- 使原解释器崩溃或超时（300 秒）的用例由 `drive.sh` 记为 `crash` 后跳过；已知会让原版段错误的用例
  （长度为 1 的 xovsp/xovsh、一维输入）在原版上不再实际运行、直接记为 `crash`，免得 macOS 弹出大量崩溃报告。

脚本兼容 Python 3.6，输入只用 `np.random.RandomState`，两边得到的输入完全相同。

## 准备官方内核

官方 2.7.0 在 PyPI 上有 macOS arm64 的 cp38/cp39 轮子，可在 Apple Silicon 上原生运行，比 Docker 下模拟 x86 快得多：

```bash
uv venv -p 3.9 orig39
VIRTUAL_ENV=orig39 uv pip install "geatpy==2.7.0" "numpy==1.23.5" "scipy<1.14" "matplotlib<3.10"
```

`numpy` 必须低于 1.24（原版 `mergecv` 用到了 `np.float`）。该轮子与 Docker 中的 Linux x86_64 原版
（`scripts/parity/run_original_core.sh`）在 `tests/fixtures` 的全部用例上结果一致。
Linux 上可以用同一个 Docker 镜像运行 `fz.py`。

## 运行

```bash
scripts/parity/fuzz/drive.sh orig39/bin/python /tmp/orig.jsonl
scripts/parity/fuzz/drive.sh .venv/bin/python /tmp/rust.jsonl
python scripts/parity/fuzz/cmp.py /tmp/orig.jsonl /tmp/rust.jsonl
```

第二批用例：在前面加上 `SCRIPT=fz2.py`。要在仓库目录外运行，或用 `-I` 运行（`drive.sh` 已经这样做），
以免仓库里的 `geatpy/` 遮住已安装的官方包。

## 结果怎么看

已知且有意保留的差异见仓库根目录 `AGENTS.md`（原版缺陷、更宽松的输入处理），比对结果中出现这些属于正常。
只调用一次的随机用例（reps = 1，如 `mutinv_len9`）的“值差异”没有意义，只用来比对是否报错。
