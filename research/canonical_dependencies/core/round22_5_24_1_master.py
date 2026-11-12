"""三轮总报告。一进二和二进三分表，不把两条线的 Sharpe 排成一张名次。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.r225241_io import RESULT_ROOT, load_json, write_text


def _rows(path) -> list[dict[str, Any]]:
    frame_path = RESULT_ROOT / path
    if not frame_path.is_file():
        return []
    if frame_path.suffix == ".csv":
        return pd.read_csv(frame_path).to_dict(orient="records")
    return []


def render_master(tests: dict[str, Any]) -> str:
    r225 = load_json(RESULT_ROOT / "round22_5" / "decision.json")
    r231 = load_json(RESULT_ROOT / "round23_1" / "decision.json")
    r241 = load_json(RESULT_ROOT / "round24_1" / "decision.json")
    c1 = load_json(RESULT_ROOT / "round22_5" / "c1_full_train_attribution.json")
    c4 = load_json(RESULT_ROOT / "round22_5" / "c4_full_train_attribution.json")
    fs2 = load_json(RESULT_ROOT / "round22_5" / "fs2_oof.json")
    nulls = load_json(RESULT_ROOT / "round22_5" / "null_recheck.json")
    lineage = load_json(RESULT_ROOT / "round22_5" / "lineage_nav_bridge.json")
    rolling = load_json(RESULT_ROOT / "round22_5" / "c4_rolling_semantics.json")
    lgbm = load_json(RESULT_ROOT / "round23_1" / "lgbm.json")
    weights = load_json(RESULT_ROOT / "round23_1" / "weights.json")
    board = load_json(RESULT_ROOT / "round24_1" / "board_audit.json")
    features = load_json(RESULT_ROOT / "round24_1" / "data" / "feature_status.json")
    hyp225 = _rows("round22_5/hypotheses.csv")
    hyp231 = _rows("round23_1/hypotheses.csv")
    hyp241 = _rows("round24_1/hypotheses.csv")
    hyps = hyp225 + hyp231 + hyp241
    counts = {name: sum(1 for row in hyps if str(row.get("result")) == name) for name in ("PASS", "FAIL", "BLOCKED")}
    exec225 = _rows("round22_5/execution_stress.csv")
    sleeves241 = _rows("round24_1/sleeve_metrics.csv")
    lines = [
        "# Round22.5–24.1 总研究报告",
        "",
        "## 1. 范围与锁定基线",
        "",
        "一进二锁定基线仍是 baseline_v5=R16_O01，全历史 2018-01-02～2026-07-24，Sharpe=1.8756627913361097，CAGR=0.5499812832273525，MaxDD=-0.24056327715394388，NAV=12750089.355724689，笔数=1107，资金=300000。费用 ROUND7_FLAT_5BP。没有改执行器。",
        "二进三是独立研究线。下面的表把两条线分开，不把它们的 Sharpe 排在同一张名次里，也不拿二进三 CAGR 去比一进二。",
        "",
        "## 2. 假设与预注册",
        "",
        "假设写在各轮 frozen_manifest.json，看结果之后没有改口径。每条假设只有 PASS、FAIL 或 BLOCKED。",
        "",
        "## 3. 数据与产物",
        "",
        "历史 result/round21* 到 result/round24 只读。新产物在 result/round22_5、result/round23_1、result/round24_1 和本目录。缺文件记 MISSING_ARTIFACT，没有用报告里的数字编造预测或成交。",
        "",
        "## 4. Round22.5 事实表与多重检验",
        "",
        "experiment_fact_table.parquet 来自 Round21.3–24 的真实 summary。缺 Top1/Top2 为 NOT_IDENTIFIABLE。多重检验登记在 multiple_testing_registry.csv。自助区间不是后验。",
        "",
        "## 5. C1/C4 覆盖与滚动",
        "",
        f"C4 公式未改。n_valid_3={rolling.get('n_valid_3')}，n_valid_4={rolling.get('n_valid_4')}，n_valid_5={rolling.get('n_valid_5')}。滚动按行，不把日历空档插进窗口。",
        "",
        "## 6. 空值、打乱与截面置换",
        "",
        f"ALLNAN 净值与 v5 相同：{nulls.get('allnan_nav_identical')}。SHUFFLE 对齐冻结 Sharpe：{nulls.get('shuffle_matches_frozen')}（读到 {nulls.get('shuffle_sharpe')}）。截面置换种子 17/42/73 不计 alpha。",
        "",
        "## 7. 归因与净值桥",
        "",
        f"C1 裁决={c1.get('verdict')}。PARTIAL 财富比重算={c1.get('partial_wealth_recomputed')}，FULL_TRAIN 财富比重算={c1.get('full_train_wealth_recomputed')}。",
        f"C4 全历史={c4.get('full_history_summary')}。净值桥={lineage.get('status')}，残差/净值={lineage.get('residual_over_nav')}。",
        "",
        "## 8. FS2 与执行压力",
        "",
        f"FS2={fs2.get('verdict')}。公式是经营现金流/平均资产。执行压力 {len(exec225)} 行，主敏感性没有当日替补。",
        "",
        "## 9. Round22.5 决策",
        "",
        f"{r225.get('promote_recommendation')}。C1={r225.get('c1')}。C4={r225.get('c4')}。FS2={r225.get('fs2')}。",
        "",
        "## 10. Round23.1 目标与嵌套权重",
        "",
        "正式主目标是 O4。O1 的日 Spearman 只作为失败对照，Ten=0 不是正式候选。内折不足 120 个信号日则 Ten=0.90。没有在外折 Sharpe 上重调。",
        f"选出的十头权重={weights.get('weights')}。与 0.90 相同={weights.get('identical_to_0.90')}。",
        "",
        "## 11. LightGBM、多样性与规则法医",
        "",
        f"多样性 Spearman={lgbm.get('daily_spearman')}，Top1 不一致={lgbm.get('top1_disagreement')}，结论={lgbm.get('verdict')}。未过门则没有融合袖套。规则没有重调阈值。",
        "",
        "## 12. Round23.1 决策",
        "",
        f"{r231.get('final')}。经济门、统计、Top 变化和执行同时成立才是 V6_RESEARCH_CANDIDATE。",
        "",
        "## 13. Round24.1 二板状态、标签与特征",
        "",
        f"板块审计 board_ok={board.get('board_ok')}，检查行={board.get('checked_main_st_rows')}，不符={board.get('mismatches')}。特征集={features.get('feature_set')}，阻断={features.get('blocked')}。",
        "label_2to3_net 的 0.001 留在诊断标签里。净值由执行器按 ROUND7_FLAT_5BP 从价格计算，不把标签成本再扣一次。",
        "",
        "## 14. 阶段模型、弃权与执行",
        "",
        "阶段 A 是可交易且已成熟样本上的涨跌分类。阶段 B 用 P 排序，或用同一树参数的回归预测已实现净收益。弃权阈值只在内折按已选可交易机会的平均已实现净收益选择。不足 5 个名字的日子不做 RankIC。",
        "",
        "## 15. Round24.1 决策",
        "",
        f"{r241.get('final')}。门={r241.get('gate')}。没有替换 baseline_2to3_v0。",
        "",
        "## 16. 分线对照、测试与生产配置",
        "",
        "### 一进二（只在这条线内比较，对照 v5）",
        "",
        "| 项 | 结果 |",
        "| --- | --- |",
        f"| Round22.5 | {r225.get('promote_recommendation')} |",
        f"| Round23.1 | {r231.get('final')} |",
        f"| v5 Sharpe | 1.8756627913361097 |",
        "",
        "### 二进三（独立，不和上一张表的 Sharpe 排名）",
        "",
        "| 项 | 结果 |",
        "| --- | --- |",
        f"| Round24.1 | {r241.get('final')} |",
        f"| 特征集 | {features.get('feature_set')} |",
        "",
        "二进三袖套指标见下表，只在二进三内部阅读。",
        "",
        "```",
        pd.DataFrame(sleeves241).to_string(index=False) if sleeves241 else "MISSING_ARTIFACT",
        "```",
        "",
        f"测试：通过 {tests.get('n_passed')}，失败 {tests.get('n_failed')}，合计 {tests.get('n_tests')}。",
        f"假设：PASS={counts['PASS']}，FAIL={counts['FAIL']}，BLOCKED={counts['BLOCKED']}。",
        "",
        "CURRENT 基线未改。PRODUCTION_CONFIG_CHANGED=false。AUTO_PROMOTE=false。LIVE_READY=false。没有独立 Forward，因此没有 LIVE_READY。",
        "",
    ]
    text = "\n".join(lines)
    write_text(RESULT_ROOT / "round22_5_24_1_master" / "MASTER_RESEARCH_REPORT.md", text)
    return text
