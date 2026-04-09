from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_sector_map() -> pd.DataFrame:
    sp1500 = Path("data/raw/Sectors/Sectors_SP1500.csv")
    if sp1500.exists():
        df = pd.read_csv(sp1500, header=None, names=["sector_etf", "sector_id", "ticker", "sector"])
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["sector_etf"] = df["sector_etf"].astype(str).str.strip()
        df["sector"] = df["sector"].astype(str).str.strip()
        df = df[df["ticker"].ne("SPY")].drop_duplicates(subset=["ticker"], keep="first")
        return df[["ticker", "sector_etf", "sector"]]

    sp500 = Path("data/raw/Sectors/Sectors_SP500_YahooNWikipedia.csv")
    df = pd.read_csv(sp500)
    df["ticker"] = df["Ticker"].astype(str).str.strip()
    df["sector"] = df["Sector_Yahoo"].astype(str).str.strip()
    sector_to_etf = {
        "Materials": "XLB",
        "Consumer_Discretionary": "XLY",
        "Consumer_Staples": "XLP",
        "Energy": "XLE",
        "Financials": "XLF",
        "Health_Care": "XLV",
        "Industrials": "XLI",
        "Information_Technology": "XLK",
        "Telecommunication_Services": "XLC",
        "Communication_Services": "XLC",
        "Utilities": "XLU",
        "Real_Estate": "XLRE",
    }
    df["sector_etf"] = df["sector"].map(sector_to_etf)
    return df[["ticker", "sector_etf", "sector"]].drop_duplicates(subset=["ticker"], keep="first")


def _load_base_panel() -> pd.DataFrame:
    base = pd.read_parquet("outputs/window_slicing/WS_2006_2023_train10_test1_dynamic_universe_models4/window_predictions_all.parquet")
    top1 = pd.read_parquet("outputs/model_exploration/no_cost_search_new_models_only_v2/exploration_predictions_all.parquet")
    top3 = pd.read_parquet("outputs/model_exploration/new_candidates_v2_full/predictions_new_models.parquet")

    keep_base = ["date", "ticker", "target_ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline"]
    keep_top1 = ["date", "ticker", "signal_baseline+top1etf_tickerd"]
    keep_top3 = ["date", "ticker", "signal_baseline+top3etf_dmean", "signal_baseline+spy+idio"]

    df = base[keep_base].merge(top1[keep_top1], on=["date", "ticker"], how="left")
    df = df.merge(top3[keep_top3], on=["date", "ticker"], how="left")
    df["date"] = pd.to_datetime(df["date"])

    etf = pd.read_parquet("data/processed/etf15_har_by_date.parquet")
    etf["date"] = pd.to_datetime(etf["date"])
    etf_keep = ["date", "SPY_d", "SPY_w", "SPY_m"] + [c for c in etf.columns if c.endswith("_d") and c.split("_")[0] in {"XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"}]
    etf = etf[etf_keep].drop_duplicates(subset=["date"], keep="last")
    df = df.merge(etf, on="date", how="left")

    sector = _load_sector_map()
    df = df.merge(sector, on="ticker", how="left")

    sector_etf_d = pd.Series(np.nan, index=df.index, dtype=float)
    for etf_name in sorted(df["sector_etf"].dropna().unique()):
        col = f"{etf_name}_d"
        if col not in df.columns:
            continue
        mask = df["sector_etf"].eq(etf_name)
        if mask.any():
            sector_etf_d.loc[mask] = df.loc[mask, col].to_numpy(dtype=float)
    df["sector_etf_d"] = sector_etf_d
    return df


def _group_demean(df: pd.DataFrame, value_col: str, by: list[str]) -> pd.Series:
    return df[value_col] - df.groupby(by, sort=False)[value_col].transform("mean")


def _group_zscore(df: pd.DataFrame, value_col: str, by: list[str]) -> pd.Series:
    mu = df.groupby(by, sort=False)[value_col].transform("mean")
    sd = df.groupby(by, sort=False)[value_col].transform("std")
    out = (df[value_col] - mu) / sd.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _daily_residual_one_control(df: pd.DataFrame, y_col: str, x_col: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("date", sort=False).groups.items():
        s = df.loc[idx, [y_col, x_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20:
            continue
        y = s[y_col].to_numpy(dtype=float)
        x = s[x_col].to_numpy(dtype=float)
        x = x - x.mean()
        y = y - y.mean()
        denom = float(np.dot(x, x))
        if denom <= 0 or not np.isfinite(denom):
            out.loc[s.index] = y
            continue
        beta = float(np.dot(x, y) / denom)
        resid = y - beta * x
        out.loc[s.index] = resid
    return out


def _evaluate_gross_no_cost(
    df_pred: pd.DataFrame,
    signal_cols: list[str],
    quantiles: list[float],
    bet_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for sig in signal_cols:
        for q in quantiles:
            for bet in bet_cols:
                x = df_pred[["date", "ticker", "target_ret", sig, bet]].dropna().copy()
                x = x[x[sig] != 0.0]
                if x.empty:
                    continue

                daily = []
                for dt, g in x.groupby("date", sort=True):
                    if q < 1.0:
                        k = int(np.ceil(len(g) * q))
                        if k <= 0:
                            continue
                        g = g.assign(_abs=np.abs(g[sig].to_numpy())).nlargest(k, "_abs").drop(columns="_abs")
                    side = np.sign(g[sig].to_numpy(dtype=float))
                    b = np.abs(g[bet].to_numpy(dtype=float))
                    raw = side * b
                    gross = np.abs(raw).sum()
                    if gross <= 0 or not np.isfinite(gross):
                        continue
                    w = raw / gross
                    ret = float(np.sum(w * g["target_ret"].to_numpy(dtype=float)))
                    daily.append((dt, ret, len(g)))
                if not daily:
                    continue
                d = pd.DataFrame(daily, columns=["date", "ret", "n_names"]).sort_values("date")
                mu = float(d["ret"].mean())
                sd = float(d["ret"].std(ddof=1))
                rows.append(
                    {
                        "signal": sig,
                        "bet_col": bet,
                        "quantile": q,
                        "days": len(d),
                        "avg_n_names": float(d["n_names"].mean()),
                        "ann_return": mu * 252.0,
                        "ann_vol": sd * np.sqrt(252.0),
                        "sharpe": (mu / sd) * np.sqrt(252.0) if sd > 0 else np.nan,
                        "cum_return": float((1.0 + d["ret"]).prod() - 1.0),
                    }
                )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast post-processing search on existing signals.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/model_exploration/postprocess_three_directions"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_base_panel()
    df["signal_blend_top3_70_30"] = 0.7 * df["signal_baseline"] + 0.3 * df["signal_baseline+top3etf_dmean"]
    df["signal_blend_top1_70_30"] = 0.7 * df["signal_baseline"] + 0.3 * df["signal_baseline+top1etf_tickerd"]
    regime_bull = df["SPY_d"].gt(0) & df["SPY_w"].gt(0)

    # Direction 1: regime switching
    df["signal_dir_regime_top3_switch"] = np.where(regime_bull, df["signal_baseline"], df["signal_blend_top3_70_30"])
    df["signal_dir_regime_top1_switch"] = np.where(regime_bull, df["signal_baseline"], df["signal_blend_top1_70_30"])
    df["signal_dir_regime_consensus_bear"] = np.where(
        regime_bull,
        df["signal_baseline"],
        np.where(np.sign(df["signal_baseline"]) == np.sign(df["signal_baseline+top3etf_dmean"]), df["signal_blend_top3_70_30"], 0.0),
    )

    # Direction 2: sector conditional
    df["signal_dir_sector_gate_blend"] = np.where(
        np.sign(df["signal_blend_top3_70_30"]) == np.sign(df["sector_etf_d"]),
        df["signal_blend_top3_70_30"],
        0.0,
    )
    df["signal_dir_sector_demean_blend"] = _group_demean(df, "signal_blend_top3_70_30", ["date", "sector"])
    df["signal_dir_sector_z_blend"] = _group_zscore(df, "signal_blend_top3_70_30", ["date", "sector"])

    # Direction 3: market-neutral / residual style
    df["log_mktcap_lag"] = np.log(df["bet_size_mktcap_lag"].where(df["bet_size_mktcap_lag"] > 0))
    df["signal_dir_resid_size_blend"] = _daily_residual_one_control(df, "signal_blend_top3_70_30", "log_mktcap_lag")
    df["signal_dir_resid_spyidio_blend"] = _daily_residual_one_control(df, "signal_blend_top3_70_30", "signal_baseline+spy+idio")
    df["signal_dir_resid_spyidio_baseline"] = _daily_residual_one_control(df, "signal_baseline", "signal_baseline+spy+idio")

    signal_cols = [c for c in df.columns if c.startswith("signal_dir_")]
    keep_panel_cols = [
        "date",
        "ticker",
        "target_ret",
        "bet_size_equal",
        "bet_size_mktcap_lag",
        "signal_baseline",
    ] + signal_cols
    df[keep_panel_cols].to_parquet(args.output_dir / "signal_panel.parquet", index=False)

    metrics = _evaluate_gross_no_cost(
        df,
        signal_cols=signal_cols + ["signal_baseline"],
        quantiles=[0.3, 0.4, 0.5, 1.0],
        bet_cols=["bet_size_equal", "bet_size_mktcap_lag"],
    )
    metrics.to_csv(args.output_dir / "all_candidate_metrics.csv", index=False)

    direction_rows = []
    baseline_best = metrics[metrics["signal"] == "signal_baseline"].sort_values("sharpe", ascending=False).head(1)
    baseline_best_sharpe = float(baseline_best["sharpe"].iloc[0])
    for prefix in ["signal_dir_regime_", "signal_dir_sector_", "signal_dir_resid_"]:
        sub = metrics[metrics["signal"].str.startswith(prefix)].sort_values("sharpe", ascending=False).head(1)
        if sub.empty:
            continue
        row = sub.iloc[0].to_dict()
        row["baseline_best_sharpe"] = baseline_best_sharpe
        row["delta_vs_baseline"] = float(row["sharpe"]) - baseline_best_sharpe
        direction_rows.append(row)

    best_by_direction = pd.DataFrame(direction_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    best_by_direction.to_csv(args.output_dir / "best_by_direction.csv", index=False)

    print(f"[done] {args.output_dir / 'signal_panel.parquet'}")
    print(f"[done] {args.output_dir / 'all_candidate_metrics.csv'}")
    print(f"[done] {args.output_dir / 'best_by_direction.csv'}")
    if not best_by_direction.empty:
        print(best_by_direction.to_string(index=False))


if __name__ == "__main__":
    main()
