from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_sector_map() -> pd.DataFrame:
    sp1500 = Path("data/raw/Sectors/Sectors_SP1500.csv")
    df = pd.read_csv(sp1500, header=None, names=["sector_etf", "sector_id", "ticker", "sector"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["sector_etf"] = df["sector_etf"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip()
    df = df[df["ticker"].ne("SPY")].drop_duplicates(subset=["ticker"], keep="first")
    return df[["ticker", "sector_etf", "sector"]]


def _sign(x: pd.Series) -> pd.Series:
    return np.sign(x.astype(float))


def _majority_consensus(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    signs = np.column_stack([np.sign(df[c].to_numpy(dtype=float)) for c in cols])
    vals = np.column_stack([df[c].to_numpy(dtype=float) for c in cols])
    pos = (signs > 0).sum(axis=1)
    neg = (signs < 0).sum(axis=1)
    out = np.zeros(len(df), dtype=float)

    pos_mask = pos >= 2
    neg_mask = neg >= 2
    if pos_mask.any():
        keep = (signs[pos_mask] > 0)
        denom = keep.sum(axis=1)
        denom = np.where(denom == 0, 1, denom)
        out[pos_mask] = (vals[pos_mask] * keep).sum(axis=1) / denom
    if neg_mask.any():
        keep = (signs[neg_mask] < 0)
        denom = keep.sum(axis=1)
        denom = np.where(denom == 0, 1, denom)
        out[neg_mask] = (vals[neg_mask] * keep).sum(axis=1) / denom
    return pd.Series(out, index=df.index, dtype=float)


def main() -> None:
    base = pd.read_parquet(
        "outputs/window_slicing/WS_2006_2023_train10_test1_dynamic_universe_models4/window_predictions_all.parquet"
    )[
        [
            "date",
            "ticker",
            "target_ret",
            "bet_size_equal",
            "bet_size_mktcap_lag",
            "signal_baseline",
            "signal_baseline+network",
        ]
    ]
    top1 = pd.read_parquet(
        "outputs/model_exploration/no_cost_search_new_models_only_v2/exploration_predictions_all.parquet"
    )[
        [
            "date",
            "ticker",
            "signal_baseline+top1etf_tickerd",
        ]
    ]
    top3 = pd.read_parquet(
        "outputs/model_exploration/new_candidates_v2_full/predictions_new_models.parquet"
    )[
        [
            "date",
            "ticker",
            "signal_baseline+top3etf_dmean",
            "signal_baseline+spy+idio",
        ]
    ]
    etf = pd.read_parquet("data/processed/etf15_har_by_date.parquet")
    sector_map = _load_sector_map()

    df = base.merge(top1, on=["date", "ticker"], how="left")
    df = df.merge(top3, on=["date", "ticker"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    etf["date"] = pd.to_datetime(etf["date"])
    df = df.merge(
        etf[
            [
                "date",
                "SPY_d",
                "SPY_w",
                "XLB_d",
                "XLE_d",
                "XLF_d",
                "XLI_d",
                "XLK_d",
                "XLP_d",
                "XLU_d",
                "XLV_d",
                "XLY_d",
            ]
        ],
        on="date",
        how="left",
    )
    df = df.merge(sector_map, on="ticker", how="left")

    sector_etf_d = pd.Series(np.nan, index=df.index, dtype=float)
    for etf_name in sorted(df["sector_etf"].dropna().unique()):
        col = f"{etf_name}_d"
        if col not in df.columns:
            continue
        mask = df["sector_etf"].eq(etf_name)
        sector_etf_d.loc[mask] = df.loc[mask, col].to_numpy(dtype=float)
    df["sector_etf_d"] = sector_etf_d

    df["signal_blend_top3_70_30"] = 0.7 * df["signal_baseline"] + 0.3 * df["signal_baseline+top3etf_dmean"]
    df["signal_consensus_majority_etf"] = _majority_consensus(
        df,
        ["signal_baseline", "signal_baseline+top1etf_tickerd", "signal_baseline+top3etf_dmean"],
    )
    df["signal_network_confirmation"] = np.where(
        _sign(df["signal_baseline"]) == _sign(df["signal_baseline+network"]),
        0.5 * (df["signal_baseline"] + df["signal_baseline+network"]),
        0.0,
    )
    bear_mask = ~(df["SPY_d"].gt(0) & df["SPY_w"].gt(0))
    sector_agree = _sign(df["signal_blend_top3_70_30"]) == _sign(df["sector_etf_d"])
    df["signal_regime_sector_confirm"] = np.where(
        bear_mask & sector_agree,
        df["signal_blend_top3_70_30"],
        np.where(~bear_mask, df["signal_baseline"], 0.0),
    )
    df["signal_majority_with_network"] = _majority_consensus(
        df,
        ["signal_baseline", "signal_blend_top3_70_30", "signal_baseline+network"],
    )
    df["signal_spyidio_filter"] = np.where(
        _sign(df["signal_baseline"]) == _sign(df["signal_baseline+spy+idio"]),
        0.5 * (df["signal_baseline"] + df["signal_baseline+spy+idio"]),
        0.0,
    )

    keep_cols = [
        "date",
        "ticker",
        "target_ret",
        "bet_size_equal",
        "bet_size_mktcap_lag",
        "signal_blend_top3_70_30",
        "signal_consensus_majority_etf",
        "signal_network_confirmation",
        "signal_regime_sector_confirm",
        "signal_majority_with_network",
        "signal_spyidio_filter",
    ]
    out = Path("outputs/model_exploration/additional_candidate_signals.parquet")
    df[keep_cols].to_parquet(out, index=False)
    print(f"[done] wrote {out}")
    print(f"[done] rows={len(df):,}")


if __name__ == "__main__":
    main()
