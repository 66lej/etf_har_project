from __future__ import annotations

from pathlib import Path
from textwrap import fill

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


OUT_PDF = Path("output/pdf/professor_three_strategy_summary.pdf")
PREVIEW_DIR = Path("tmp/pdfs/professor_three_strategy_summary")


def wrap(text: str, width: int = 100) -> str:
    return fill(text, width=width)


def add_page(ax, title: str, lines: list[str], footer: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.06, 0.95, title, fontsize=20, fontweight="bold", ha="left", va="top")

    y = 0.90
    for line in lines:
        if line == "":
            y -= 0.022
            continue
        if line.startswith("HEADER:"):
            y -= 0.01
            ax.text(0.06, y, line.replace("HEADER:", ""), fontsize=12.5, fontweight="bold", ha="left", va="top")
            y -= 0.032
            continue
        if line.startswith("MATH:"):
            y -= 0.004
            ax.text(0.08, y, line.replace("MATH:", ""), fontsize=12.0, ha="left", va="top")
            y -= 0.050
            continue
        text = line
        x = 0.08
        if line.startswith("- "):
            text = wrap(line, width=92)
        else:
            text = wrap(line, width=98)
            x = 0.06
        ax.text(x, y, text, fontsize=10.5, ha="left", va="top", linespacing=1.35)
        n_lines = text.count("\n") + 1
        y -= 0.030 * n_lines + 0.008

    ax.text(0.06, 0.035, footer, fontsize=8.5, color="#555555", ha="left", va="bottom")


def add_table_page(ax, title: str, df: pd.DataFrame, footer: str, benchmark_note: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.06, 0.95, title, fontsize=20, fontweight="bold", ha="left", va="top")

    headers = [
        "Strategy",
        "Gross Sharpe\n(0 bp)",
        "Net\n10 bp",
        "Net\n20 bp",
        "Net\n50 bp",
        "Main message",
    ]
    widths = [0.22, 0.18, 0.12, 0.12, 0.12, 0.24]
    left = 0.06
    total_w = 0.88
    top = 0.84
    header_h = 0.10
    row_h = 0.13

    x = left
    for header, frac in zip(headers, widths):
        w = total_w * frac
        ax.add_patch(Rectangle((x, top - header_h), w, header_h, facecolor="#E9EEF5", edgecolor="#B8B8B8", linewidth=1))
        ax.text(x + 0.01, top - 0.06, wrap(header, width=14), fontsize=8.2, fontweight="bold", ha="left", va="center")
        x += w

    for r, row in enumerate(df.itertuples(index=False), start=1):
        y_top = top - header_h - (r - 1) * row_h
        bg = "#FBFBFB" if r % 2 else "#F2F5F8"
        values = [
            wrap(str(row[0]), width=22),
            wrap(str(row[1]), width=16),
            wrap(str(row[2]), width=10),
            wrap(str(row[3]), width=10),
            wrap(str(row[4]), width=10),
            wrap(str(row[5]), width=26),
        ]
        x = left
        for val, frac in zip(values, widths):
            w = total_w * frac
            ax.add_patch(Rectangle((x, y_top - row_h), w, row_h, facecolor=bg, edgecolor="#B8B8B8", linewidth=1))
            ax.text(x + 0.01, y_top - 0.07, val, fontsize=8.8, ha="left", va="center", linespacing=1.25)
            x += w

    ax.text(
        0.06,
        0.20,
        wrap(
            "Gross AlphaMark Sharpe is a 0 bp metric. Net Sharpe numbers come from the separate tradability review "
            "with explicit transaction-cost assumptions at 10, 20, and 50 bps. The strongest overall profile remains "
            + benchmark_note,
            width=105,
        ),
        fontsize=10.5,
        ha="left",
        va="top",
        linespacing=1.35,
    )
    ax.text(0.06, 0.035, footer, fontsize=8.5, color="#555555", ha="left", va="bottom")


def main() -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    alphamark = pd.read_csv("outputs/alphamark_eval/alphamark_low_turnover_three_model_summary.csv")
    netcost = pd.read_csv("outputs/model_exploration/final_strategy_ranking_v2.csv")
    baseline_tc = pd.read_csv("outputs/model_exploration/low_turnover_baseline_q100_v1/best_by_scenario.csv")
    bucket_diag = pd.read_csv("outputs/analysis/normalized_dollar_volume_buckets_v1/bucket_summary.csv")

    def sharpe_pair(family: str) -> str:
        sub = alphamark[alphamark["family"].eq(family)].copy()
        cap = float(sub.loc[sub["bet_size_col"].eq("betsize_cap250k"), "sharpe"].iloc[0])
        mkt = float(sub.loc[sub["bet_size_col"].eq("betsize_mktcap_lag"), "sharpe"].iloc[0])
        return f"{cap:.3f} / {mkt:.3f}"

    def net_sharpe(family: str, scenario: str) -> str:
        sub = netcost[(netcost["family"].eq(family)) & (netcost["scenario"].eq(scenario))].copy()
        return f"{float(sub['sharpe'].iloc[0]):.3f}"

    def baseline_gross_pair() -> str:
        sub = pd.read_pickle("outputs/alphamark_eval/2006_2023_baseline/SUMMARY_STATS/summary_stats_20060103_20231229.pkl")
        sub = sub[
            sub["signal"].eq("pret_baseline")
            & sub["qrank"].eq("qr_100")
            & sub["stat_type"].eq("sharpe")
        ].copy()
        cap = float(sub.loc[sub["bet_size_col"].eq("betsize_cap250k"), "value"].iloc[0])
        mkt = float(sub.loc[sub["bet_size_col"].eq("betsize_mktcap_lag"), "value"].iloc[0])
        return f"{cap:.3f} / {mkt:.3f}"

    def baseline_net_pair(scenario: str) -> str:
        sub = baseline_tc[baseline_tc["scenario"].eq(scenario)].copy()
        return f"{float(sub['sharpe'].iloc[0]):.3f}"

    def baseline_best_spec(scenario: str) -> str:
        sub = baseline_tc[baseline_tc["scenario"].eq(scenario)].iloc[0]
        bet = "mktcap" if sub["bet_col"] == "bet_size_mktcap_lag" else "equal"
        return f"{bet}, hold={int(sub['hold_days'])}, thr={sub['rebalance_threshold']:.1f}"

    def bucket_value(family: str, bucket: str, col: str) -> str:
        sub = bucket_diag[(bucket_diag["family"].eq(family)) & (bucket_diag["bucket"].eq(bucket))].copy()
        return f"{float(sub[col].iloc[0]):.3f}"

    summary_df = pd.DataFrame(
        [
            {
                "Strategy": "Low-turnover consensus majority ETF",
                "Gross AlphaMark Sharpe\n(cap250k / mktcap)": sharpe_pair("low_turnover_consensus_majority_etf"),
                "Net Sharpe\n10 bps": net_sharpe("low_turnover_consensus_majority_etf", "net_10bps"),
                "Net Sharpe\n20 bps": net_sharpe("low_turnover_consensus_majority_etf", "net_20bps"),
                "Net Sharpe\n50 bps": net_sharpe("low_turnover_consensus_majority_etf", "net_50bps"),
                "Main message": "ETF information works best as confirmation.",
            },
            {
                "Strategy": "Low-turnover blend top3",
                "Gross AlphaMark Sharpe\n(cap250k / mktcap)": sharpe_pair("low_turnover_blend_top3"),
                "Net Sharpe\n10 bps": net_sharpe("low_turnover_blend_top3", "net_10bps"),
                "Net Sharpe\n20 bps": net_sharpe("low_turnover_blend_top3", "net_20bps"),
                "Net Sharpe\n50 bps": net_sharpe("low_turnover_blend_top3", "net_50bps"),
                "Main message": "ETF information is useful as a soft adjustment.",
            },
            {
                "Strategy": "Low-turnover regime",
                "Gross AlphaMark Sharpe\n(cap250k / mktcap)": sharpe_pair("low_turnover_regime"),
                "Net Sharpe\n10 bps": net_sharpe("low_turnover_regime", "net_10bps"),
                "Net Sharpe\n20 bps": net_sharpe("low_turnover_regime", "net_20bps"),
                "Net Sharpe\n50 bps": net_sharpe("low_turnover_regime", "net_50bps"),
                "Main message": "ETF information is most useful in weak regimes.",
            },
        ]
    )

    footer = "Data window: 2006-01-03 to 2023-12-29. Rolling setup: 10-year train, 1-year test, dynamic universe, Ridge regression."
    benchmark_note = (
        f"the consensus strategy. Optimized low-turnover baseline benchmark at q=1.0: gross {baseline_gross_pair()}, "
        f"net 10 bps {baseline_net_pair('net_10bps')} [{baseline_best_spec('net_10bps')}], "
        f"net 20 bps {baseline_net_pair('net_20bps')} [{baseline_best_spec('net_20bps')}], "
        f"net 50 bps {baseline_net_pair('net_50bps')} [{baseline_best_spec('net_50bps')}]."
    )

    pages: list[tuple[str, object]] = []
    pages.append(
        (
            "Three Strategy Summary",
            [
                "This note summarizes the three final low-turnover strategies that survived both the exploratory screening and the AlphaMark evaluation.",
                "",
                "HEADER:Project framing",
                "- Predictive layer: stock-level daily return forecasts from rolling-window Ridge regressions.",
                "- Execution layer: quantile-based long-short portfolios with explicit turnover controls.",
                "- Main empirical lesson: ETF-linked information does not help much when used as a large raw feature block, but it becomes useful when used as a confirmation, conditioning, or soft-adjustment device.",
                "",
                "HEADER:Final ranking by AlphaMark Sharpe",
                f"- Consensus majority ETF: {sharpe_pair('low_turnover_consensus_majority_etf')} (cap250k / mktcap).",
                f"- Blend top3: {sharpe_pair('low_turnover_blend_top3')} (cap250k / mktcap).",
                f"- Regime: {sharpe_pair('low_turnover_regime')} (cap250k / mktcap).",
                "",
                "HEADER:Cost-adjusted Sharpe from tradability review",
                f"- Consensus majority ETF: 10 bps {net_sharpe('low_turnover_consensus_majority_etf', 'net_10bps')}, 20 bps {net_sharpe('low_turnover_consensus_majority_etf', 'net_20bps')}, 50 bps {net_sharpe('low_turnover_consensus_majority_etf', 'net_50bps')}.",
                f"- Blend top3: 10 bps {net_sharpe('low_turnover_blend_top3', 'net_10bps')}, 20 bps {net_sharpe('low_turnover_blend_top3', 'net_20bps')}, 50 bps {net_sharpe('low_turnover_blend_top3', 'net_50bps')}.",
                f"- Regime: 10 bps {net_sharpe('low_turnover_regime', 'net_10bps')}, 20 bps {net_sharpe('low_turnover_regime', 'net_20bps')}, 50 bps {net_sharpe('low_turnover_regime', 'net_50bps')}.",
                "",
                "HEADER:Baseline benchmark reference (q=1.0)",
                f"- Baseline gross Sharpe: {baseline_gross_pair()} (cap250k / mktcap).",
                f"- Optimized low-turnover baseline net Sharpe: 10 bps {baseline_net_pair('net_10bps')} [{baseline_best_spec('net_10bps')}], 20 bps {baseline_net_pair('net_20bps')} [{baseline_best_spec('net_20bps')}], 50 bps {baseline_net_pair('net_50bps')} [{baseline_best_spec('net_50bps')}].",
                "",
                "The strongest strategy is the consensus version. This is economically meaningful because it suggests that ETF-side information is most valuable when it confirms the stock-only forecast, not when it replaces it.",
            ],
        )
    )
    pages.append(
        (
            "Shared Framework",
            [
                "HEADER:Base predictive models",
                "Baseline model:",
                r"MATH:$\hat r^{\mathrm{base}}_{i,t+1}=\beta_0+\beta_d r^{(d)}_{i,t}+\beta_w r^{(w)}_{i,t}+\beta_m r^{(m)}_{i,t}$",
                "Top1 ETF model: baseline terms plus the daily return of the single ETF that is most correlated with the stock inside the current training window.",
                r"MATH:$\hat r^{\mathrm{top1}}_{i,t+1}=\hat r^{\mathrm{base}}_{i,t+1}+\gamma \, ETF^{\mathrm{top1}(i)}_{d,t}$",
                "Top3 ETF model: baseline terms plus the average daily return of the three ETFs that are most correlated with the stock inside the current training window.",
                r"MATH:$\hat r^{\mathrm{top3}}_{i,t+1}=\hat r^{\mathrm{base}}_{i,t+1}+\gamma \cdot \frac{1}{3}\sum_{e\in \mathrm{Top3}(i)} ETF_{e,d,t}$",
                "",
                "HEADER:Common trading logic",
                "- Signal sign determines long versus short.",
                "- Position size is based on either equal weights or lagged market-cap weights.",
                "- The portfolio is normalized to one unit of gross exposure each day.",
                r"MATH:$w_{i,t}\propto \mathrm{sign}(s_{i,t})\cdot |\mathrm{bet\_size}_{i,t}|,\qquad \sum_i |w_{i,t}| = 1$",
                "",
                "HEADER:Turnover controls",
                "- hold_days: minimum time before another rebalance is allowed.",
                "- rebalance_threshold: minimum gap between current and target portfolios before a rebalance is executed.",
                "- These two hyperparameters are what made the strategies economically viable after transaction costs.",
            ],
        )
    )
    pages.append(
        (
            "Strategy 1: Consensus Majority ETF",
            [
                "HEADER:Model design",
                "This strategy combines three forecast signals: baseline, baseline plus top1 ETF daily term, and baseline plus top3 ETF daily mean.",
                r"MATH:$s^{\mathrm{cons}}_{i,t}=\mathrm{MajorityVote}\!\left(s^{\mathrm{base}}_{i,t},\,s^{\mathrm{top1}}_{i,t},\,s^{\mathrm{top3}}_{i,t}\right)$",
                "The final signal takes the average of the positive signals if at least two of the three models are positive, the average of the negative signals if at least two are negative, and zero otherwise.",
                "",
                "HEADER:Trading rule",
                "- Best execution setting: bet_size_mktcap_lag, q=1.0, hold_days=20, rebalance_threshold=0.3.",
                "- Interpretation: trade only when at least two related models agree on direction, then rebalance infrequently.",
                "- Gross AlphaMark Sharpe: 0.492 / 0.537 (cap250k / mktcap).",
                "- Best net Sharpe across the low-turnover grid: 0.482 at 10 bps, 0.417 at 20 bps, 0.221 at 50 bps.",
                "",
                "HEADER:Why it matters",
                "This is the best-performing strategy in AlphaMark. The result supports a clear interpretation: ETF-linked information is most useful as a confirmation filter. The strategy removes many marginal names, lowers unnecessary turnover, and keeps exposure focused on names where multiple views agree.",
            ],
        )
    )
    pages.append(
        (
            "Strategy 2: Regime",
            [
                "HEADER:Model design",
                "First define a blended signal:",
                r"MATH:$s^{\mathrm{blend}}_{i,t}=0.7\, s^{\mathrm{base}}_{i,t}+0.3\, s^{\mathrm{top3}}_{i,t}$",
                "Then define a simple bull regime using SPY_d > 0 and SPY_w > 0. In bull regimes, use the baseline signal. Outside bull regimes, use the blended signal only when the baseline and top3 ETF signals have the same sign; otherwise set the signal to zero.",
                r"MATH:$s^{\mathrm{reg}}_{i,t}=s^{\mathrm{base}}_{i,t}\quad \mathrm{if}\quad SPY_{d,t}>0\ \mathrm{and}\ SPY_{w,t}>0$",
                r"MATH:$s^{\mathrm{reg}}_{i,t}=s^{\mathrm{blend}}_{i,t}\quad \mathrm{if}\quad \mathrm{sign}(s^{\mathrm{base}}_{i,t})=\mathrm{sign}(s^{\mathrm{top3}}_{i,t})$",
                r"MATH:$s^{\mathrm{reg}}_{i,t}=0\quad \mathrm{otherwise}$",
                "",
                "HEADER:Trading rule",
                "- Best execution setting: bet_size_equal, q=1.0, hold_days=20, rebalance_threshold=0.1.",
                "- Interpretation: trust stock-only momentum when the broad market is strong; require ETF confirmation when the environment is weaker.",
                "- Gross AlphaMark Sharpe: 0.419 / 0.421 (cap250k / mktcap).",
                "- Best net Sharpe across the low-turnover grid: 0.452 at 10 bps, 0.404 at 20 bps, 0.279 at 50 bps.",
                "",
                "HEADER:Why it matters",
                "This strategy has the clearest economic story. It shows that ETF information is conditional information: it is not uniformly useful every day, but it becomes useful when market conditions are less supportive and a second layer of confirmation is valuable.",
            ],
        )
    )
    pages.append(
        (
            "Strategy 3: Blend Top3",
            [
                "HEADER:Model design",
                "This strategy uses a soft combination rather than a hard vote:",
                r"MATH:$s^{\mathrm{blend}}_{i,t}=0.7\, s^{\mathrm{base}}_{i,t}+0.3\, s^{\mathrm{top3}}_{i,t}$",
                "The idea is deliberately conservative. Stock-specific HAR information remains the core signal, and ETF-linked information is only allowed to make a small correction.",
                "",
                "HEADER:Trading rule",
                "- Best execution setting: bet_size_equal, q=1.0, hold_days=20, rebalance_threshold=0.1.",
                "- Interpretation: keep broad cross-sectional coverage, but rebalance only when the desired portfolio has moved enough.",
                "- Gross AlphaMark Sharpe: 0.449 / 0.406 (cap250k / mktcap).",
                "- Best net Sharpe across the low-turnover grid: 0.419 at 10 bps, 0.373 at 20 bps, 0.259 at 50 bps.",
                "",
                "HEADER:Why it matters",
                "This is the simplest of the three strategies and remains competitive in AlphaMark. It is useful for presentation because it shows that the ETF side does not need to dominate the model to add value; a modest adjustment can already improve the economic profile relative to a naive high-turnover implementation.",
            ],
        )
    )
    pages.append(
        (
            "Liquidity Bucket Diagnostic",
            [
                "HEADER:Experiment design",
                "We computed normalized dollar volume for each stock and day using current dollar volume divided by its 20-day lagged moving average. Each day, the cross-section was sorted into five buckets Q1 to Q5, from relatively quiet names to relatively active names.",
                r"MATH:$NDVOL_{i,t}=\frac{Volume_{i,t}\cdot |Close_{i,t}|}{MA_{20}(Volume\cdot |Close|)_{i,t-1}}$",
                "We then re-ran each low-turnover backtest separately inside each bucket, using the same execution parameters as the final strategy versions.",
                "",
                "HEADER:Main empirical findings",
                f"- At 20 bps, the full-universe portfolios remain positive: baseline {bucket_value('baseline_lowturnover', 'ALL', 'net_sharpe')}, consensus {bucket_value('consensus_majority_etf', 'ALL', 'net_sharpe')}, blend {bucket_value('blend_top3', 'ALL', 'net_sharpe')}, regime {bucket_value('regime', 'ALL', 'net_sharpe')}.",
                "- Once the portfolio is restricted to a single liquidity bucket, net Sharpe becomes negative for all four strategies. This means the alpha is not coming from one isolated liquidity segment.",
                f"- On a gross basis, the ETF-linked strategies look cleanest in the middle-to-upper liquidity buckets, especially consensus in Q3 {bucket_value('consensus_majority_etf', 'Q3', 'gross_sharpe')} and Q4 {bucket_value('consensus_majority_etf', 'Q4', 'gross_sharpe')}.",
                f"- Extremes are weaker: low-activity Q1 and very high-activity Q5 tend to be poor for all strategies. For example, baseline net Sharpe is {bucket_value('baseline_lowturnover', 'Q1', 'net_sharpe')} in Q1 and {bucket_value('baseline_lowturnover', 'Q5', 'net_sharpe')} in Q5.",
                "",
                "HEADER:Interpretation",
                "Extreme-liquidity buckets appear to be harder environments for these signals. Very quiet names are more fragile after costs, while very active names may reflect event-driven or crowded flow that weakens a slow-moving HAR-style signal.",
                "The practical implication is that normalized volume is a useful diagnostic, but not a strong standalone filter. The final alpha still seems to require broad cross-sectional diversification plus low-turnover execution.",
            ],
        )
    )
    pages.append(
        (
            "Discussion",
            [
                "HEADER:What do the results suggest about ETF terms?",
                "The evidence does not support the claim that ETF information is a strong standalone forecasting driver. Across almost all experiments, the stock's own HAR history remains the dominant source of predictive power.",
                "",
                "At the same time, the results do not support the stronger claim that ETF terms are pure noise. If they were pure noise, they would not systematically improve the strategy when used as a confirmation or conditioning device. Yet that is exactly where they help the most: in the consensus strategy, the regime strategy, and the conservative blend strategy.",
                "",
                "A more defensible interpretation is that ETF-linked information has low incremental signal-to-noise ratio. Used directly as a large raw feature block, it behaves mostly like noise. Used selectively, however, it can still add value by refining the baseline rather than replacing it.",
                "",
                "HEADER:Practical takeaway",
                "Our final conclusion is therefore not that ETF terms beat the stock-only baseline on their own. It is that ETF information is weak but not useless: its economic value appears when it is used as a filter, a confirmation layer, or a regime-dependent adjustment inside a low-turnover execution framework.",
            ],
        )
    )

    with PdfPages(OUT_PDF) as pdf:
        for idx, (title, lines) in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1])
            add_page(ax, title, lines, footer + f" Page {idx} of {len(pages) + 1}.")
            pdf.savefig(fig)
            fig.savefig(PREVIEW_DIR / f"page_{idx}.png", dpi=170)
            plt.close(fig)

        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0, 0, 1, 1])
        add_table_page(
            ax,
            "Side-by-Side Comparison",
            summary_df,
            footer + f" Page {len(pages) + 1} of {len(pages) + 1}.",
            benchmark_note,
        )
        pdf.savefig(fig)
        fig.savefig(PREVIEW_DIR / f"page_{len(pages) + 1}.png", dpi=170)
        plt.close(fig)

    print(f"[done] wrote {OUT_PDF}")
    print(f"[done] preview pages in {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
