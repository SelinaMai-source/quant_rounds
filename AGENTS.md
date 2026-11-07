# 一进二策略 — Agent 须知

工作范围只在本目录 `/home/mxy/Stock/课题/一进二策略`（实路径 `/data1/stock-mxy/课题/一进二策略`）。本课题目录（含 `core/`、`scripts/`、`result/`、`.cursor/`）**可写**。

仍禁止写入 Stock 其余部分：`/home/mxy/Stock/src`、`/home/mxy/Stock/runs`、Stock 根下除本课题外的任何路径，以及其他课题。可只读 import `first_board_hermes`，可读 `runs/` / 快照 / 预测。

新卖点、notebook、追踪表、回测产物一律写在本课题目录，不要写回 `Stock/src` 或 `Stock/runs`。

## 硬规则：ZyQuant / Hermes / 正式回测

1. **开发框架 = [ZyQuant](https://github.com/xiejinglover/ZyQuant)**：数据快照、PIT、因子、实验编排优先用本机 `zyquant` API（`DataSnapshot` / PIT / Factor / Experiment）。先 `import zyquant; print(zyquant.__version__)`。**禁止** `pip install --upgrade zyquant`，除非现环境完全跑不起来并先停下来记原因。
2. **缺数查 Hermes**：课题快照不够时，只读连 Hermes（`HERMES_MYSQL_*` / 本机已有凭据），用 ZyQuant Hermes 连接器看表；不把采集结果写进 Stock src/runs。
3. **正式回测数字仍走 Stock 引擎**：`local_executor` → `first_board_hermes.fixed_sleeve_backtest`（`run_fixed_dual_sleeve_top3_backtest`）。不允许用 ZyQuant `Strategy`/`Backtest` 的 NAV/Sharpe **替换**正式结果。ZyQuant 账本若跑了必须标 `zyquant_diagnostic`。

对照入口：`/data1/stock-mxy/tools/backtest_first_board_fixed_two_sleeves.py`。课题只读 import。Stock 没有的卖点只写在 `core/local_executor.py`，不能另起执行器当正式结果。

XGBoost 训练关闭 `subsample` / `colsample_bytree`（必须 `1.0`）；不要改 Stock 已训完的 fs00263 / fs01122。

## 旧表锁（fs00263 Top1 线，不要改）

`result/一进二_最优实验组追踪.md` 仍有效。该线默认对照 = **`baseline_v3` = `R4_B05`**（`result/round4/R4_B05/`）。`baseline_v1`（`R1_00_baseline`）与 `baseline_v2`（`R3_03`）保留作参考，不是删档。不自动升 `baseline_v4`。

## 新表（fs01122 Top2 线）

`result/一进二_fs01122_Top2_实验组追踪.md`：契约为 fs01122 预滤集成 / n100_d2 / Top2 / 25%+15% NAV / `open_gap_min=-0.01` / `close_from_d2`。产物写 `result/replay_fs01122/`，不覆盖旧 `result/roundN/`。

## Round7 基线重放（不删旧表）

canonical Round7 在本目录 `core/round7/`、`result/round7/`、`scripts/run_round7.py`。

`result/一进二_Round7基线_实验组追踪.md`：开发基线 = `R7_FS01122_DUAL90_10_CLEAN`（双头 90/10 + clean_risk + Top2 30/10）。产物写 `result/replay_round7/`。不改旧两张表，不覆盖本目录 canonical `result/round7/`。不自动升版。
