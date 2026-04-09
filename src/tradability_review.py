from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EvalSpec:
    signal_col: str
    bet_col: str
    quantile: float

    @property
    def label(self) -> str:
        q = int(round(self.quantile * 100))
        return f"{self.signal_col}__{self.bet_col}__q{q}"


def _select_top_abs(g: pd.DataFrame, signal_col: str, quantile: float) -> pd.DataFrame:
    if g.empty:
        return g
    if quantile >= 1.0:
        return g
    k = int(np.ceil(len(g) * quantile))
    if k <= 0:
        return g.iloc[0:0]
    tmp = g.assign(_abs_signal=np.abs(g[signal_col].to_numpy(dtype=float)))
    out = tmp.nlargest(k, columns=["_abs_signal"]).drop(columns=["_abs_signal"])
    return out


def _daily_portfolio(
    g: pd.DataFrame,
    signal_col: str,
    bet_col: str,
) -> pd.DataFrame:
    side = np.sign(g[signal_col].to_numpy(dtype=float))
    bet_abs = np.abs(g[bet_col].to_numpy(dtype=float))
    raw_pos = side * bet_abs
    gross = np.sum(np.abs(raw_pos))
    if gross <= 0 or not np.isfinite(gross):
        return pd.DataFrame(columns=["ticker", "w", "target_ret"])
    w = raw_pos / gross
    out = pd.DataFrame(
        {
            "ticker": g["ticker"].to_numpy(),
            "w": w,
            "target_ret": g["target_ret"].to_numpy(dtype=float),
        }
    )
    return out


def _turnover(prev_w: pd.Series | None, w_today: pd.Series) -> float:
    if prev_w is None:
        return float(np.abs(w_today).sum())
    aligned = pd.concat([prev_w, w_today], axis=1, sort=False).fillna(0.0)
    aligned.columns = ["prev", "today"]
    return float(np.abs(aligned["today"] - aligned["prev"]).sum())


def evaluate_strategy(df: pd.DataFrame, spec: EvalSpec, cost_bps_list: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    req = ["date", "ticker", "target_ret", spec.signal_col, spec.bet_col]
    x = df[req].dropna().copy()
    x = x[np.isfinite(x[spec.signal_col]) & np.isfinite(x["target_ret"]) & np.isfinite(x[spec.bet_col])]
    x = x[x[spec.signal_col] != 0.0]
    x = x.sort_values(["date", "ticker"])

    rows = []
    prev_w = None
    for dt, g in x.groupby("date", sort=True):
        gq = _select_top_abs(g, signal_col=spec.signal_col, quantile=spec.quantile)
        p = _daily_portfolio(gq, signal_col=spec.signal_col, bet_col=spec.bet_col)
        if p.empty:
            continue
        w_today = p.set_index("ticker")["w"]
        gross_ret = float((p["w"] * p["target_ret"]).sum())
        to = _turnover(prev_w, w_today)
        row = {
            "date": pd.Timestamp(dt),
            "gross_ret": gross_ret,
            "turnover": to,
            "n_names": int(len(p)),
        }
        for bps in cost_bps_list:
            row[f"net_ret_{int(round(bps))}bps"] = gross_ret - to * (bps / 10_000.0)
        rows.append(row)
        prev_w = w_today

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return daily, pd.DataFrame()

    summary_rows = []
    base = {
        "strategy": spec.label,
        "signal_col": spec.signal_col,
        "bet_col": spec.bet_col,
        "quantile": spec.quantile,
        "days": int(len(daily)),
        "avg_n_names": float(daily["n_names"].mean()),
        "avg_daily_turnover": float(daily["turnover"].mean()),
    }
    gross = daily["gross_ret"]
    gross_std = float(gross.std(ddof=1))
    gross_ann = float(gross.mean() * 252.0)
    gross_sharpe = float((gross.mean() / gross_std) * np.sqrt(252.0)) if gross_std > 0 else np.nan
    gross_cum = float((1.0 + gross).prod() - 1.0)
    summary_rows.append(
        {
            **base,
            "scenario": "gross",
            "ann_return": gross_ann,
            "ann_vol": gross_std * np.sqrt(252.0),
            "sharpe": gross_sharpe,
            "cum_return": gross_cum,
        }
    )

    for bps in cost_bps_list:
        c = daily[f"net_ret_{int(round(bps))}bps"]
        c_std = float(c.std(ddof=1))
        c_ann = float(c.mean() * 252.0)
        c_sharpe = float((c.mean() / c_std) * np.sqrt(252.0)) if c_std > 0 else np.nan
        c_cum = float((1.0 + c).prod() - 1.0)
        summary_rows.append(
            {
                **base,
                "scenario": f"net_{int(round(bps))}bps",
                "ann_return": c_ann,
                "ann_vol": c_std * np.sqrt(252.0),
                "sharpe": c_sharpe,
                "cum_return": c_cum,
            }
        )

    return daily, pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tradability review with turnover + transaction costs.")
    p.add_argument(
        "--pred-path",
        type=Path,
        default=Path("outputs/window_slicing/2006_2023_t10_dynamic_etf/window_predictions_all.parquet"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/tradability_review/2006_2023"),
    )
    p.add_argument(
        "--signals",
        nargs="+",
        default=[
            "signal_baseline",
            "signal_baseline+etf",
            "signal_baseline+network",
            "signal_baseline+etf+network",
        ],
    )
    p.add_argument("--bet-cols", nargs="+", default=["bet_size_equal", "bet_size_mktcap_lag"])
    p.add_argument("--quantiles", nargs="+", type=float, default=[1.0, 0.25])
    p.add_argument("--cost-bps", nargs="+", type=float, default=[10, 20, 50])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    need_cols = ["date", "ticker", "target_ret"] + args.signals + args.bet_cols
    df = pd.read_parquet(args.pred_path, columns=need_cols)
    df["date"] = pd.to_datetime(df["date"])

    summaries = []
    for sig in args.signals:
        for bet in args.bet_cols:
            for q in args.quantiles:
                spec = EvalSpec(signal_col=sig, bet_col=bet, quantile=float(q))
                daily, summary = evaluate_strategy(df, spec, cost_bps_list=args.cost_bps)
                if daily.empty:
                    continue
                daily.to_csv(out_dir / f"daily_{spec.label}.csv", index=False)
                summaries.append(summary)

    if not summaries:
        raise RuntimeError("No strategy results produced. Check signal/bet columns and input data.")

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all = summary_all.sort_values(["signal_col", "bet_col", "quantile", "scenario"]).reset_index(drop=True)
    summary_all.to_csv(out_dir / "tradability_summary.csv", index=False)

    # Compact table for quick review
    keep = ["signal_col", "bet_col", "quantile", "scenario", "ann_return", "ann_vol", "sharpe", "cum_return", "avg_daily_turnover", "avg_n_names", "days"]
    summary_all[keep].to_csv(out_dir / "tradability_summary_compact.csv", index=False)
    print(f"[done] summary: {out_dir / 'tradability_summary.csv'}")
    print(f"[done] compact: {out_dir / 'tradability_summary_compact.csv'}")


if __name__ == "__main__":
    main()
