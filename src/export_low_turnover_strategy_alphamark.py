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
    p = argparse.ArgumentParser(description="Export low-turnover strategy signals to AlphaMark daily PKLs.")
    p.add_argument(
        "--base-panel",
        type=Path,
        default=Path("outputs/window_slicing/WS_2006_2023_train10_test1_dynamic_universe_models4/window_predictions_all.parquet"),
    )
    p.add_argument(
        "--extra-panel",
        action="append",
        type=Path,
        default=[],
        help="Optional parquet panels to merge on date,ticker for additional signal columns.",
    )
    p.add_argument("--signal-col", type=str, required=True)
    p.add_argument("--bet-col", type=str, required=True)
    p.add_argument("--quantile", type=float, default=1.0)
    p.add_argument("--hold-days", type=int, required=True)
    p.add_argument("--rebalance-threshold", type=float, required=True)
    p.add_argument("--alphamark-signal-name", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--signal-mode",
        type=str,
        choices=["current_weight", "raw_score_signed_by_position"],
        default="raw_score_signed_by_position",
        help="How to write the AlphaMark signal once the held book is determined.",
    )
    return p.parse_args()


def load_panel(base_panel: Path, extra_panels: list[Path], signal_col: str) -> pd.DataFrame:
    base_cols = ["date", "ticker", "target_ret", "bet_size_equal", "bet_size_mktcap_lag"]
    try:
        base = pd.read_parquet(base_panel, columns=base_cols + [signal_col])
    except Exception:
        base = pd.read_parquet(base_panel, columns=base_cols)
    if signal_col not in base.columns:
        for extra in extra_panels:
            extra_df = pd.read_parquet(extra, columns=["date", "ticker", signal_col])
            base = base.merge(extra_df, on=["date", "ticker"], how="left")
            if signal_col in base.columns:
                break
    if signal_col not in base.columns:
        raise KeyError(f"signal column {signal_col!r} not found in base panel or extra panels")
    base["date"] = pd.to_datetime(base["date"])
    base = base.dropna(subset=["date", "ticker", "target_ret", "bet_size_equal", "bet_size_mktcap_lag", signal_col])
    return base.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.base_panel, args.extra_panel, args.signal_col)

    current_w: pd.Series | None = None
    last_rebalance_idx = -10**9
    n_written = 0

    for day_idx, (dt, g) in enumerate(panel.groupby("date", sort=True)):
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

        pos_sign = np.sign(held["ticker"].map(current_w).astype(float).to_numpy(dtype=float))
        if args.signal_mode == "current_weight":
            held[args.alphamark_signal_name] = held["ticker"].map(current_w).astype(float)
        else:
            raw_abs = np.abs(held[args.signal_col].to_numpy(dtype=float))
            held[args.alphamark_signal_name] = pos_sign * raw_abs

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
        f"quantile={args.quantile}, hold_days={args.hold_days}, threshold={args.rebalance_threshold}, "
        f"signal_mode={args.signal_mode}"
    )


if __name__ == "__main__":
    main()
