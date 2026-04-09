from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export low-turnover regime signal to AlphaMark daily PKLs.")
    p.add_argument(
        "--pred-path",
        type=Path,
        default=Path("outputs/model_exploration/postprocess_three_directions_v1/signal_panel.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/alphamark_input/AM_INPUT_low_turnover_regime_q100"),
    )
    p.add_argument("--signal-col", type=str, default="signal_dir_regime_consensus_bear")
    p.add_argument("--bet-col", type=str, default="bet_size_equal")
    p.add_argument("--quantile", type=float, default=1.0)
    p.add_argument("--hold-days", type=int, default=20)
    p.add_argument("--rebalance-threshold", type=float, default=0.1)
    p.add_argument("--alphamark-signal-name", type=str, default="pret_regime_lowturnover")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "date",
        "ticker",
        "target_ret",
        "bet_size_equal",
        "bet_size_mktcap_lag",
        args.signal_col,
    ]
    df = pd.read_parquet(args.pred_path, columns=keep_cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "ticker", "target_ret", "bet_size_equal", "bet_size_mktcap_lag", args.signal_col])
    df = df.sort_values(["date", "ticker"])

    current_w: pd.Series | None = None
    last_rebalance_idx = -10**9
    n_written = 0
    for day_idx, (dt, g) in enumerate(df.groupby("date", sort=True)):
        desired = _desired_weights(g, signal_col=args.signal_col, bet_col=args.bet_col, quantile=float(args.quantile))
        to_desired = _turnover(current_w, desired) if not desired.empty else 0.0

        do_rebalance = False
        if current_w is None:
            do_rebalance = True
        elif (day_idx - last_rebalance_idx) >= int(args.hold_days) and to_desired >= float(args.rebalance_threshold):
            do_rebalance = True

        if do_rebalance:
            current_w = desired
            last_rebalance_idx = day_idx

        if current_w is None or current_w.empty:
            continue

        held = g[g["ticker"].isin(current_w.index)].copy()
        if held.empty:
            continue
        held[args.alphamark_signal_name] = held["ticker"].map(current_w).astype(float)
        out = held.rename(
            columns={
                "target_ret": "fret_1_MR",
                "bet_size_equal": "betsize_cap250k",
                "bet_size_mktcap_lag": "betsize_mktcap_lag",
            }
        )[
            [
                "ticker",
                args.alphamark_signal_name,
                "fret_1_MR",
                "betsize_cap250k",
                "betsize_mktcap_lag",
            ]
        ]
        out.to_pickle(args.output_dir / f"features_{pd.Timestamp(dt).strftime('%Y%m%d')}.pkl")
        n_written += 1

    print(f"[done] wrote {n_written} daily PKLs to {args.output_dir}")
    print(
        f"[done] strategy: signal={args.signal_col}, bet_col={args.bet_col}, "
        f"quantile={args.quantile}, hold_days={args.hold_days}, threshold={args.rebalance_threshold}"
    )


if __name__ == "__main__":
    main()
