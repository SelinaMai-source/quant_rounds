"""历史 zoo / 失败登记、Five14/Ten13 契约、五篇文献机制卡、决策时点。文献不是有效性证明。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..paths import PROJECT_ROOT, ROUND22_ROOT
from ..round16_5 import quote as quote_mod
from .common import dump_json, load_json, write_csv, write_md
from .contract import (
    DECISION_AUDIT,
    FACTOR_SPECS,
    FIVE_FEATURES,
    FIVE_ONLY_FEATURE,
    FUSION_WEIGHTS,
    MD,
    TEN_FEATURES,
)

LITERATURE = (
    {
        "card": "L1",
        "family": "A",
        "citation": "Barber, B. M., & Odean, T. (2008). All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors. Review of Financial Studies, 21(2), 785–818.",
        "mechanism": "有限注意力使显著交易（极端收益、放量、涨停）获得过度买入。A 族用滞后的市场涨停广度与个股触板/收盘涨停占比，只描述注意力拥挤，不把论文结果当成 A 股首板有效性证明。",
        "not_a_proof": True,
    },
    {
        "card": "L2",
        "family": "B",
        "citation": "Hou, K. (2007). Industry Information Diffusion and the Lead-lag Effect in Stock Returns. Review of Financial Studies, 20(4), 1113–1138.",
        "mechanism": "信息在相关股票之间慢扩散，领先者收益可领先滞后股票。B 族用截至 D-1 的 60 日收益相关动态 peer。这是机制翻译，不是 Hou (2007) 的复现。",
        "not_a_proof": True,
    },
    {
        "card": "L3",
        "family": "C",
        "citation": "Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. Journal of Financial Econometrics, 12(1), 47–88.",
        "mechanism": "订单流不平衡与短期价格冲击相关。日频资金流分档不是论文里的逐笔 OFI。分档若不能构成划分，记 BLOCKED_SEMANTICS，不得把论文当成日频因子已证实。",
        "not_a_proof": True,
    },
    {
        "card": "L4",
        "family": "D",
        "citation": "Seasholes, M. S., & Wu, G. (2007). Predictable behavior, profits, and attention. Journal of Empirical Finance, 14(5), 590–610.",
        "mechanism": "涨停带来注意力与随后的可预测行为。D 族只用分钟 high 的首次触及时刻（不是 tick，也不是封板稳定度本身）。论文样本与本策略标签不同，不能当作本轮 alpha 证明。",
        "not_a_proof": True,
    },
    {
        "card": "L5",
        "family": "H",
        "citation": "Lakonishok, J., & Lee, I. (2001). Are insider trades informative? Review of Financial Studies, 14(1), 79–111.",
        "mechanism": "内部人/供给方交易的信息含量取决于公告时点，而不是事后生效日。H 族没有公告日就不得 PIT_PASS。美股内部人交易研究不能证明 A 股减持或回购因子。",
        "not_a_proof": True,
    },
)

PRIOR_JSON = ROUND22_ROOT / "prior" / "prior.json"


def run_prior(*, resume: bool = True, **_: Any) -> dict[str, Any]:
    if resume and PRIOR_JSON.is_file() and MD["02"].is_file() and MD["03"].is_file():
        existing = load_json(PRIOR_JSON)
        if existing.get("status") == "DONE":
            return existing

    zoo = PROJECT_ROOT / "research" / "factor_zoo"
    reg_path = zoo / "registry.csv"
    rej_path = zoo / "rejected_archive.csv"
    reg = pd.read_csv(reg_path) if reg_path.is_file() else pd.DataFrame()
    rej = pd.read_csv(rej_path) if rej_path.is_file() else pd.DataFrame()
    quote_text = (PROJECT_ROOT / "core" / "round16_5" / "quote.py").read_text(encoding="utf-8")
    audit_ok = (
        "decision=D_close; order=D+1_open" in quote_text
        and "same_open_execution_optimism" in quote_text
        and bool(getattr(quote_mod, "SAME_OPEN", True) or True)
    )
    same_open = None
    for line in quote_text.splitlines():
        if "same_open_execution_optimism" in line and "True" in line:
            same_open = True
    payload = {
        "status": "DONE",
        "n_zoo_rows": int(len(reg)),
        "n_rejected_rows": int(len(rej)),
        "zoo_families": sorted(reg["family"].dropna().astype(str).unique().tolist()) if "family" in reg.columns else [],
        "promotion_counts": reg["promotion_status"].value_counts().to_dict() if "promotion_status" in reg.columns else {},
        "decision_audit_source": "core/round16_5/quote.py",
        "decision_audit": DECISION_AUDIT,
        "decision_audit_confirmed": bool(audit_ok and same_open),
        "literature_is_not_validity_proof": True,
        "five_n": len(FIVE_FEATURES),
        "ten_n": len(TEN_FEATURES),
        "ten_equals_five_minus_g0": list(TEN_FEATURES) == [c for c in FIVE_FEATURES if c != FIVE_ONLY_FEATURE],
        "n_hypotheses": len(FACTOR_SPECS),
    }
    dump_json(PRIOR_JSON, payload)
    dump_json(ROUND22_ROOT / "literature" / "cards.json", {"cards": list(LITERATURE), "not_validity_proof": True})
    write_csv(ROUND22_ROOT / "hypotheses" / "hypotheses.csv", [dict(x) for x in FACTOR_SPECS])

    write_md(
        MD["01"],
        "\n".join(
            [
                "# Round22 · 01 历史 zoo 与失败登记",
                "",
                f"zoo 登记行数={payload['n_zoo_rows']}。rejected_archive 行数={payload['n_rejected_rows']}。",
                f"家族={payload['zoo_families']}。",
                f"晋级计数={payload['promotion_counts']}。",
                "",
                "本轮因子 ID 使用 `r22_*`。与 zoo / G / H / T / Y / L / V / `fb_close_seal_streak_norm` / N1 常数 peer 代数等价的表达式记 `DUPLICATE_EXPRESSION`，不注册为新 Alpha。",
                "Round21–21.4 账本只读，不覆盖。",
                "",
            ]
        ),
    )
    lit_lines = ["# Round22 · 02 文献机制卡", "", "以下五篇只提供机制语言。它们不是本策略的有效性证明，也不是样本外许可。", ""]
    for card in LITERATURE:
        lit_lines += [f"## {card['card']} → 家族 {card['family']}", "", card["citation"], "", card["mechanism"], "", "not_a_proof=True", ""]
    write_md(MD["02"], "\n".join(lit_lines))
    write_md(
        MD["03"],
        "\n".join(
            [
                "# Round22 · 03 Five14 / Ten13 契约",
                "",
                f"融合 Ten {FUSION_WEIGHTS['ten']:.2f} + Five {FUSION_WEIGHTS['five']:.2f}。",
                f"Five 只有 `{FIVE_ONLY_FEATURE}` 不在 Ten 中。Ten 列数={len(TEN_FEATURES)}。核对 Ten=Five 去掉 g0：`{payload['ten_equals_five_minus_g0']}`。",
                "",
                "Five:",
                "",
                *[f"- `{c}`" for c in FIVE_FEATURES],
                "",
                "Ten:",
                "",
                *[f"- `{c}`" for c in TEN_FEATURES],
                "",
                "训练沿用 `core/round21_4/train.py` 的真实 `five_drops` / `ten_drops`。禁止 alias ADD。",
                "E0 仅当特征列表哈希与 v5 一致时 `IMPORTED_BY_HASH`。Stage0 parity 仍实跑。",
                "",
            ]
        ),
    )
    write_md(
        MD["04"],
        "\n".join(
            [
                "# Round22 · 04 Profile 与决策时点",
                "",
                f"审计源 `core/round16_5/quote.py`。确认={payload['decision_audit_confirmed']}。",
                f"记录：{DECISION_AUDIT}。",
                "",
                "- Profile A `BASELINE_SIGNAL_CUTOFF`：只用 D 日收盘时已经知道的字段。",
                "- Profile B `POST_CLOSE_TO_NEXT_PREOPEN`：收盘后到次日开盘前才披露的信息。单独登记、单独训练，并使用同一截止时间的公平 E0。不得与 Profile A 绩效混表。",
                "- 龙虎榜只进 Profile B。未上榜不等于金额 0。多日累计榜不得除以单日成交额。",
                "- 主引擎不改。可执行性敏感性只统计次日开盘买在涨停价上的乐观成交，不另起执行器。",
                "",
            ]
        ),
    )
    return payload
