from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SearchSpec:
    signal_col: str
    bet_col: str
    quantile: float
    hold_days: int
    rebalance_threshold: float

    @property
    def label(self) -> str:
        q = int(round(self.quantile * 100))
        thr = int(round(self.rebalance_threshold * 100))
        return f"{self.signal_col}__{self.bet_col}__q{q}__hold{self.hold_days}__thr{thr}"


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
    w = raw / gross
    return pd.Series(w, index=gq["ticker"].to_numpy(), dtype=float)


def _turnover(prev_w: pd.Series | None, new_w: pd.Series) -> float:
    if prev_w is None:
        return float(np.abs(new_w).sum())
    aligned = pd.concat([prev_w, new_w], axis=1, sort=False).fillna(0.0)
    aligned.columns = ["prev", "new"]
    return float(np.abs(aligned["new"] - aligned["prev"]).sum())


def _daily_ret(current_w: pd.Series | None, ret_map: pd.Series) -> float:
    if current_w is None or current_w.empty:
        return 0.0
    aligned = current_w.reindex(ret_map.index).fillna(0.0)
    return float((aligned * ret_map).sum())


def run_strategy(df: pd.DataFrame, spec: SearchSpec, cost_bps_list: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    req = ["date", "ticker", "target_ret", spec.signal_col, spec.bet_col]
    x = df[req].dropna().copy()
    x = x[np.isfinite(x[spec.signal_col]) & np.isfinite(x["target_ret"]) & np.isfinite(x[spec.bet_col])]
    x = x[x[spec.signal_col] != 0.0]
    x = x.sort_values(["date", "ticker"])

    rows = []
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

        ret_map = g.set_index("ticker")["target_ret"].astype(float)
        gross_ret = _daily_ret(current_w, ret_map)
        row = {
            "date": pd.Timestamp(dt),
            "gross_ret": gross_ret,
            "turnover": turnover,
            "n_names": int(0 if current_w is None else len(current_w)),
            "rebalanced": int(do_rebalance),
        }
        for bps in cost_bps_list:
            row[f"net_ret_{int(round(bps))}bps"] = gross_ret - turnover * (bps / 10_000.0)
        rows.append(row)

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return daily, pd.DataFrame()

    summary_rows = []
    base = {
        "strategy": spec.label,
        "signal_col": spec.signal_col,
        "bet_col": spec.bet_col,
        "quantile": spec.quantile,
        "hold_days": spec.hold_days,
        "rebalance_threshold": spec.rebalance_threshold,
        "days": int(len(daily)),
        "avg_n_names": float(daily["n_names"].mean()),
        "avg_daily_turnover": float(daily["turnover"].mean()),
        "rebalance_rate": float(daily["rebalanced"].mean()),
    }
    for scenario, col in [("gross", "gross_ret")] + [
        (f"net_{int(round(bps))}bps", f"net_ret_{int(round(bps))}bps") for bps in cost_bps_list
    ]:
        s = daily[col]
        sd = float(s.std(ddof=1))
        summary_rows.append(
            {
                **base,
                "scenario": scenario,
                "ann_return": float(s.mean() * 252.0),
                "ann_vol": sd * np.sqrt(252.0),
                "sharpe": float((s.mean() / sd) * np.sqrt(252.0)) if sd > 0 else np.nan,
                "cum_return": float((1.0 + s).prod() - 1.0),
            }
        )
    return daily, pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search low-turnover variants for the regime signal.")
    p.add_argument(
        "--pred-path",
        type=Path,
        default=Path("outputs/model_exploration/postprocess_three_directions_v1/signal_panel.parquet"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/model_exploration/low_turnover_regime_v1"),
    )
    p.add_argument("--signal-col", type=str, default="signal_dir_regime_consensus_bear")
    p.add_argument("--bet-cols", nargs="+", default=["bet_size_equal", "bet_size_mktcap_lag"])
    p.add_argument("--quantiles", nargs="+", type=float, default=[1.0, 0.5, 0.4, 0.3])
    p.add_argument("--hold-days", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20])
    p.add_argument("--rebalance-thresholds", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    p.add_argument("--cost-bps", nargs="+", type=float, default=[10, 20, 50])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    need_cols = ["date", "ticker", "target_ret", args.signal_col] + args.bet_cols
    df = pd.read_parquet(args.pred_path, columns=need_cols)
    df["date"] = pd.to_datetime(df["date"])

    summaries = []
    for bet_col in args.bet_cols:
        for quantile in args.quantiles:
            for hold_days in args.hold_days:
                for thr in args.rebalance_thresholds:
                    spec = SearchSpec(
                        signal_col=args.signal_col,
                        bet_col=bet_col,
                        quantile=float(quantile),
                        hold_days=int(hold_days),
                        rebalance_threshold=float(thr),
                    )
                    daily, summary = run_strategy(df, spec, cost_bps_list=args.cost_bps)
                    if daily.empty:
                        continue
                    daily.to_csv(args.out_dir / f"daily_{spec.label}.csv", index=False)
                    summaries.append(summary)

    if not summaries:
        raise RuntimeError("No low-turnover strategy results produced.")

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all = summary_all.sort_values(
        ["scenario", "sharpe", "hold_days", "rebalance_threshold"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    summary_all.to_csv(args.out_dir / "low_turnover_summary.csv", index=False)

    # Best row under each scenario.
    best_rows = []
    for scenario in summary_all["scenario"].unique():
        sub = summary_all[summary_all["scenario"] == scenario].sort_values("sharpe", ascending=False).head(1)
        if not sub.empty:
            best_rows.append(sub)
    best = pd.concat(best_rows, ignore_index=True)
    best.to_csv(args.out_dir / "best_by_scenario.csv", index=False)
    print(f"[done] {args.out_dir / 'low_turnover_summary.csv'}")
    print(f"[done] {args.out_dir / 'best_by_scenario.csv'}")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
