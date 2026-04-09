from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/a12345/etf-har-project")
FIG_DIR = ROOT / "report/overleaf_project/figures"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def build_initial_model_comparison() -> None:
    p = ROOT / "outputs/alphamark_eval/AM_EVAL_2006_2023_models4_baseline_etf_network/SUMMARY_STATS/summary_stats_20060103_20231229.pkl"
    df = pd.read_pickle(p)
    sub = df[
        (df["stat_type"] == "sharpe")
        & (df["qrank"] == "qr_100")
        & (df["bet_size_col"].isin(["betsize_cap250k", "betsize_mktcap_lag"]))
    ][["signal", "bet_size_col", "value"]].copy()

    signal_order = [
        "pret_baseline",
        "pret_baseline+etf",
        "pret_baseline+network",
        "pret_baseline+etf+network",
    ]
    label_map = {
        "pret_baseline": "Baseline",
        "pret_baseline+etf": "Baseline + Raw ETF",
        "pret_baseline+network": "Baseline + Network",
        "pret_baseline+etf+network": "Baseline + ETF + Network",
    }
    bet_order = ["betsize_cap250k", "betsize_mktcap_lag"]
    bet_label = {"betsize_cap250k": "cap250k", "betsize_mktcap_lag": "mktcap"}
    color_map = {"betsize_cap250k": "#1f77b4", "betsize_mktcap_lag": "#ff7f0e"}

    x = np.arange(len(signal_order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for i, bet in enumerate(bet_order):
        vals = []
        for sig in signal_order:
            m = sub[(sub["signal"] == sig) & (sub["bet_size_col"] == bet)]
            vals.append(float(m["value"].iloc[0]))
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=bet_label[bet], color=color_map[bet])

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[s] for s in signal_order], rotation=10, ha="right")
    ax.set_ylabel("Gross AlphaMark Sharpe (qr_100)")
    ax.set_title("Initial Four-Model Comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "initial_model_comparison.png", dpi=220)
    plt.close(fig)


def build_cost_robustness() -> None:
    final_p = ROOT / "outputs/model_exploration/final_strategy_ranking_v2.csv"
    base_p = ROOT / "outputs/model_exploration/low_turnover_baseline_q100_v1/best_by_scenario.csv"
    final_df = pd.read_csv(final_p)
    base_df = pd.read_csv(base_p)

    scenarios = ["net_10bps", "net_20bps", "net_50bps"]
    x = np.array([10, 20, 50], dtype=float)
    families = {
        "low_turnover_consensus_majority_etf": "Consensus",
        "low_turnover_blend_top3": "Blend Top3",
        "low_turnover_regime": "Regime",
    }
    colors = {
        "low_turnover_consensus_majority_etf": "#d62728",
        "low_turnover_blend_top3": "#2ca02c",
        "low_turnover_regime": "#9467bd",
        "baseline": "#4c4c4c",
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for fam, label in families.items():
        vals = []
        for sc in scenarios:
            v = final_df[(final_df["family"] == fam) & (final_df["scenario"] == sc)]["sharpe"].iloc[0]
            vals.append(float(v))
        ax.plot(x, vals, marker="o", linewidth=2.2, label=label, color=colors[fam])

    base_vals = [float(base_df[base_df["scenario"] == sc]["sharpe"].iloc[0]) for sc in scenarios]
    ax.plot(x, base_vals, marker="o", linewidth=2.2, linestyle="--", label="Optimized Baseline", color=colors["baseline"])

    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Cost Robustness of Final Strategies")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_robustness.png", dpi=220)
    plt.close(fig)


def build_net_cumulative_paths() -> None:
    series = {
        "Optimized Baseline": ROOT / "outputs/model_exploration/low_turnover_baseline_q100_v1/daily_signal_baseline__bet_size_mktcap_lag__q100__hold10__thr20.csv",
        "Consensus": ROOT / "outputs/model_exploration/low_turnover_consensus_majority_etf_v1/daily_signal_consensus_majority_etf__bet_size_mktcap_lag__q100__hold20__thr30.csv",
        "Regime": ROOT / "outputs/model_exploration/low_turnover_regime_v1/daily_signal_dir_regime_consensus_bear__bet_size_equal__q100__hold20__thr10.csv",
        "Blend Top3": ROOT / "outputs/model_exploration/low_turnover_blend_top3_v1/daily_signal_blend_top3_70_30__bet_size_equal__q100__hold20__thr10.csv",
    }
    colors = {
        "Optimized Baseline": "#4c4c4c",
        "Consensus": "#d62728",
        "Regime": "#9467bd",
        "Blend Top3": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for label, path in series.items():
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        wealth = (1.0 + df["net_ret_20bps"].astype(float)).cumprod()
        ax.plot(df["date"], wealth, linewidth=2.0, label=label, color=colors[label])

    ax.set_title("Cumulative Net Return Paths (20 bps)")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "net_cumulative_paths_20bps.png", dpi=220)
    plt.close(fig)


def build_quantile_comparison() -> None:
    p = ROOT / "outputs/alphamark_eval/alphamark_low_turnover_three_model_summary_qmulti.csv"
    df = pd.read_csv(p)
    q_order = ["qr_100", "qr_75", "qr_50", "qr_25"]
    q_label = {"qr_100": "100%", "qr_75": "75%", "qr_50": "50%", "qr_25": "25%"}
    family_order = ["consensus_majority_etf", "blend_top3", "regime"]
    family_label = {
        "consensus_majority_etf": "Consensus",
        "blend_top3": "Blend Top3",
        "regime": "Regime",
    }
    color_map = {
        "consensus_majority_etf": "#d62728",
        "blend_top3": "#2ca02c",
        "regime": "#9467bd",
    }

    plot_df = (
        df.groupby(["family", "qrank"], as_index=False)["value"]
        .max()
        .assign(family=lambda x: pd.Categorical(x["family"], family_order, ordered=True))
        .sort_values(["qrank", "family"])
    )

    x = np.arange(len(q_order))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, fam in enumerate(family_order):
        vals = []
        for q in q_order:
            v = plot_df[(plot_df["family"] == fam) & (plot_df["qrank"] == q)]["value"].iloc[0]
            vals.append(float(v))
        ax.bar(x + (i - 1) * width, vals, width=width, label=family_label[fam], color=color_map[fam])

    ax.set_xticks(x)
    ax.set_xticklabels([q_label[q] for q in q_order])
    ax.set_ylabel("Best gross AlphaMark Sharpe")
    ax.set_xlabel("Quantile slice")
    ax.set_title("AlphaMark Quantile Comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alphamark_quantile_comparison.png", dpi=220)
    plt.close(fig)


def build_liquidity_diagnostic() -> None:
    p = ROOT / "outputs/analysis/normalized_dollar_volume_buckets_v1/bucket_summary.csv"
    df = pd.read_csv(p)
    families = [
        "baseline_lowturnover",
        "consensus_majority_etf",
        "blend_top3",
        "regime",
    ]
    labels = {
        "baseline_lowturnover": "Optimized Baseline",
        "consensus_majority_etf": "Consensus",
        "blend_top3": "Blend Top3",
        "regime": "Regime",
    }
    colors = {
        "baseline_lowturnover": "#4c4c4c",
        "consensus_majority_etf": "#d62728",
        "blend_top3": "#2ca02c",
        "regime": "#9467bd",
    }
    bucket_order = ["ALL", "Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(bucket_order))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for fam in families:
        sub = (
            df[df["family"] == fam]
            .assign(bucket=lambda x: pd.Categorical(x["bucket"], bucket_order, ordered=True))
            .sort_values("bucket")
        )
        axes[0].plot(x, sub["gross_sharpe"].to_numpy(dtype=float), marker="o", linewidth=2.0, label=labels[fam], color=colors[fam])
        axes[1].plot(x, sub["net_sharpe"].to_numpy(dtype=float), marker="o", linewidth=2.0, label=labels[fam], color=colors[fam])

    axes[0].set_title("Gross Sharpe by NDVOL bucket")
    axes[1].set_title("Net Sharpe by NDVOL bucket (20 bps)")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_order)
        ax.set_xlabel("Normalized dollar volume bucket")
    axes[0].set_ylabel("Sharpe")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "liquidity_bucket_diagnostic.png", dpi=220)
    plt.close(fig)


def build_pipeline_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.03, 0.22, 0.17, 0.56, "#dbe9f6", "Data Panels\nStocks + ETFs"),
        (0.23, 0.22, 0.18, 0.56, "#e8f0db", "Rolling Universe\n+ Window Slicing\n10y train / 1y test"),
        (0.45, 0.22, 0.17, 0.56, "#f8e3d3", "Ridge Forecasts\nBaseline / ETF /\nNetwork-related"),
        (0.66, 0.22, 0.14, 0.56, "#eee1f7", "Signal Design\nConsensus /\nRegime / Blend"),
        (0.83, 0.22, 0.14, 0.56, "#fde2e2", "Execution + Eval\nLow-turnover\nAlphaMark + Costs"),
    ]

    for x, y, w, h, c, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="#5a5a5a", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11, linespacing=1.3)

    arrows = [
        (0.20, 0.50, 0.03, 0.0),
        (0.41, 0.50, 0.04, 0.0),
        (0.62, 0.50, 0.04, 0.0),
        (0.80, 0.50, 0.03, 0.0),
    ]
    for x, y, dx, dy in arrows:
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), arrowprops=dict(arrowstyle="->", lw=1.8, color="#444444"))

    ax.text(
        0.5,
        0.06,
        "Project workflow: weak ETF information is first compressed through rolling-window relatedness, then filtered through low-turnover execution and economic evaluation.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pipeline_schematic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    build_pipeline_schematic()
    build_initial_model_comparison()
    build_cost_robustness()
    build_net_cumulative_paths()
    build_quantile_comparison()
    build_liquidity_diagnostic()
    print(f"[done] wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
