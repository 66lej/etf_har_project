from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySpec:
    family: str
    signal_col: str
    bet_col: str
    quantile: float
    hold_days: int
    rebalance_threshold: float


def _select_top_abs(g: pd.DataFrame, signal_col: str, quantile: float) -> pd.DataFrame:
    if g.empty:
        return g
    if quantile >= 1.0:
        return g
    k = int(np.ceil(len(g) * quantile))
    if k <= 0:
        return g.iloc[0:0]
    tmp = g.assign(_abs_signal=np.abs(g[signal_col].to_numpy(dtype=float)))
    return tmp.nlargest(k, columns=["_abs_signal"]).drop(columns=["_abs_signal"])


def _desired_weights(g: pd.DataFrame, signal_col: str, bet_col: str, quantile: float) -> pd.Series:
    gq = _select_top_abs(g, signal_col=signal_col, quantile=quantile)
    if gq.empty:
        return pd.Series(dtype=float)
    side = np.sign(gq[signal_col].to_numpy(dtype=float))
    bet_abs = np.abs(gq[bet_col].to_numpy(dtype=float))
    raw = side * bet_abs
    gross = float(np.abs(raw).sum())
    if gross <= 0 or not np.isfinite(gross):
        return pd.Series(dtype=float)
    return pd.Series(raw / gross, index=gq["ticker"].to_numpy(), dtype=float)


def _turnover(prev_w: pd.Series | None, new_w: pd.Series) -> float:
    if prev_w is None:
        return float(np.abs(new_w).sum())
    aligned = pd.concat([prev_w, new_w], axis=1, sort=False).fillna(0.0)
    aligned.columns = ["prev", "new"]
    return float(np.abs(aligned["new"] - aligned["prev"]).sum())


def build_ndvol_panel(stock_path: Path, ma_window: int) -> pd.DataFrame:
    stock = pd.read_parquet(stock_path, columns=["ticker", "date", "volume", "close"])
    stock["date"] = pd.to_datetime(stock["date"])
    stock = stock.sort_values(["ticker", "date"]).copy()
    stock["dollar_vol"] = stock["volume"].astype(float) * stock["close"].abs().astype(float)
    ma = (
        stock.groupby("ticker", sort=False)["dollar_vol"]
        .transform(lambda s: s.rolling(ma_window, min_periods=ma_window).mean().shift(1))
    )
    stock["ndvol"] = stock["dollar_vol"] / ma
    stock = stock.replace([np.inf, -np.inf], np.nan)
    stock = stock.dropna(subset=["ndvol"]).copy()

    ranks = stock.groupby("date", sort=False)["ndvol"].rank(method="first")
    counts = stock.groupby("date", sort=False)["ndvol"].transform("size")
    score = (ranks - 1.0) / counts
    stock["ndvol_bucket"] = pd.cut(
        score,
        bins=[-1e-9, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        include_lowest=True,
    )
    stock.loc[counts < 5, "ndvol_bucket"] = np.nan
    return stock[["date", "ticker", "ndvol", "ndvol_bucket"]].dropna(subset=["ndvol_bucket"])


def load_signal_panel() -> pd.DataFrame:
    base = pd.read_parquet(
        "outputs/window_slicing/WS_2006_2023_train10_test1_dynamic_universe_models4/window_predictions_all.parquet",
        columns=["date", "ticker", "target_ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline"],
    )
    addl = pd.read_parquet(
        "outputs/model_exploration/additional_candidate_signals.parquet",
        columns=["date", "ticker", "signal_blend_top3_70_30", "signal_consensus_majority_etf"],
    )
    regime = pd.read_parquet(
        "outputs/model_exploration/postprocess_three_directions_v1/signal_panel.parquet",
        columns=["date", "ticker", "signal_dir_regime_consensus_bear"],
    )
    df = base.merge(addl, on=["date", "ticker"], how="left")
    df = df.merge(regime, on=["date", "ticker"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    return df


def run_filtered_strategy(
    df: pd.DataFrame,
    spec: StrategySpec,
    bucket_label: str | None,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    req = ["date", "ticker", "target_ret", spec.signal_col, spec.bet_col, "ndvol_bucket", "ndvol"]
    x = df[req].dropna().copy()
    x = x[np.isfinite(x[spec.signal_col]) & np.isfinite(x["target_ret"]) & np.isfinite(x[spec.bet_col])]
    x = x[x[spec.signal_col] != 0.0]
    if bucket_label is not None:
        x = x[x["ndvol_bucket"].eq(bucket_label)].copy()
    x = x.sort_values(["date", "ticker"])

    rows: list[dict[str, object]] = []
    current_w: pd.Series | None = None
    last_rebalance_idx = -10**9
    for day_idx, (dt, g) in enumerate(x.groupby("date", sort=True)):
        desired = _desired_weights(g, signal_col=spec.signal_col, bet_col=spec.bet_col, quantile=spec.quantile)
        do_rebalance = False
        to_desired = _turnover(current_w, desired) if not desired.empty else 0.0
        if current_w is None:
            do_rebalance = True
        elif (day_idx - last_rebalance_idx) >= spec.hold_days and to_desired >= spec.rebalance_threshold:
            do_rebalance = True

        if do_rebalance:
            current_w = desired
            turnover = to_desired
            last_rebalance_idx = day_idx
        else:
            turnover = 0.0

        if current_w is None or current_w.empty:
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "gross_ret": 0.0,
                    "net_ret": 0.0,
                    "turnover": turnover,
                    "n_names": 0,
                    "avg_ndvol": np.nan,
                    "rebalanced": int(do_rebalance),
                }
            )
            continue

        held = g[g["ticker"].isin(current_w.index)].copy()
        if held.empty:
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "gross_ret": 0.0,
                    "net_ret": 0.0,
                    "turnover": turnover,
                    "n_names": 0,
                    "avg_ndvol": np.nan,
                    "rebalanced": int(do_rebalance),
                }
            )
            continue
        held["weight"] = held["ticker"].map(current_w).astype(float)
        gross_ret = float((held["weight"] * held["target_ret"]).sum())
        net_ret = gross_ret - turnover * (cost_bps / 10_000.0)
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "gross_ret": gross_ret,
                "net_ret": net_ret,
                "turnover": turnover,
                "n_names": int(len(held)),
                "avg_ndvol": float(held["ndvol"].mean()),
                "rebalanced": int(do_rebalance),
            }
        )

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        summary = {
            "family": spec.family,
            "bucket": bucket_label or "ALL",
            "cost_bps": cost_bps,
            "gross_sharpe": np.nan,
            "net_sharpe": np.nan,
            "avg_turnover": np.nan,
            "rebalance_rate": np.nan,
            "avg_n_names": np.nan,
            "avg_ndvol": np.nan,
            "days": 0,
        }
        return daily, summary

    def _sharpe(s: pd.Series) -> float:
        sd = float(s.std(ddof=1))
        return float((s.mean() / sd) * np.sqrt(252.0)) if sd > 0 else np.nan

    summary = {
        "family": spec.family,
        "bucket": bucket_label or "ALL",
        "cost_bps": cost_bps,
        "gross_sharpe": _sharpe(daily["gross_ret"]),
        "net_sharpe": _sharpe(daily["net_ret"]),
        "gross_ann_return": float(daily["gross_ret"].mean() * 252.0),
        "net_ann_return": float(daily["net_ret"].mean() * 252.0),
        "avg_turnover": float(daily["turnover"].mean()),
        "rebalance_rate": float(daily["rebalanced"].mean()),
        "avg_n_names": float(daily["n_names"].mean()),
        "avg_ndvol": float(daily["avg_ndvol"].mean()),
        "days": int(len(daily)),
    }
    return daily, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalized dollar volume bucket analysis.")
    p.add_argument("--stock-path", type=Path, default=Path("data/processed/stock_har_window.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/analysis/normalized_dollar_volume_buckets_v1"))
    p.add_argument("--ma-window", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ndvol = build_ndvol_panel(args.stock_path, ma_window=args.ma_window)
    panel = load_signal_panel().merge(ndvol, on=["date", "ticker"], how="inner")

    specs = [
        StrategySpec("baseline_lowturnover", "signal_baseline", "bet_size_mktcap_lag", 1.0, 10, 0.2),
        StrategySpec("consensus_majority_etf", "signal_consensus_majority_etf", "bet_size_mktcap_lag", 1.0, 20, 0.3),
        StrategySpec("blend_top3", "signal_blend_top3_70_30", "bet_size_equal", 1.0, 20, 0.1),
        StrategySpec("regime", "signal_dir_regime_consensus_bear", "bet_size_equal", 1.0, 20, 0.1),
    ]
    buckets = [None, "Q1", "Q2", "Q3", "Q4", "Q5"]

    summary_rows: list[dict[str, object]] = []
    for spec in specs:
        for bucket in buckets:
            daily, summary = run_filtered_strategy(panel, spec, bucket_label=bucket, cost_bps=float(args.cost_bps))
            label = "ALL" if bucket is None else bucket
            daily.to_csv(args.out_dir / f"daily_{spec.family}_{label}.csv", index=False)
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(["family", "bucket"]).reset_index(drop=True)
    summary_df.to_csv(args.out_dir / "bucket_summary.csv", index=False)

    # Convenience wide tables for gross and net Sharpe.
    gross_wide = summary_df.pivot(index="family", columns="bucket", values="gross_sharpe").reset_index()
    net_wide = summary_df.pivot(index="family", columns="bucket", values="net_sharpe").reset_index()
    gross_wide.to_csv(args.out_dir / "bucket_gross_sharpe_wide.csv", index=False)
    net_wide.to_csv(args.out_dir / "bucket_net_sharpe_wide.csv", index=False)

    print(f"[done] {args.out_dir / 'bucket_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
