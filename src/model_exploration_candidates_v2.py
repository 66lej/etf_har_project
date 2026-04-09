from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from window_slicing_pipeline import (
    _eligible_tickers_train_test,
    _fit_predict_ridge,
    rolling_windows,
    select_window_etf_feature_cols,
)


def _daily_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.endswith("_d")]


def _pick_ticker_topk_etf_d(train_df: pd.DataFrame, d_cols: list[str], k: int = 1, min_obs: int = 80) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ticker, g in train_df.groupby("ticker", sort=False):
        scores: list[tuple[str, float]] = []
        for c in d_cols:
            z = g[[c, "ret"]].dropna()
            if len(z) < min_obs:
                continue
            corr = z[c].corr(z["ret"])
            if pd.isna(corr):
                continue
            scores.append((c, abs(float(corr))))
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            out[ticker] = [c for c, _ in scores[:k]]
    return out


def _map_ticker_col_feature(df: pd.DataFrame, ticker_to_col: dict[str, str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if not ticker_to_col:
        return out
    inv: dict[str, list[str]] = {}
    for t, c in ticker_to_col.items():
        inv.setdefault(c, []).append(t)
    for c, tickers in inv.items():
        if c not in df.columns:
            continue
        mask = df["ticker"].isin(tickers)
        if mask.any():
            out.loc[mask] = df.loc[mask, c].to_numpy(dtype=float)
    return out


def _build_top1_har_features(df: pd.DataFrame, ticker_to_d: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["feat_top1_etf_d"] = _map_ticker_col_feature(df, ticker_to_d)
    out["feat_top1_etf_w"] = _map_ticker_col_feature(df, {t: c.replace("_d", "_w") for t, c in ticker_to_d.items()})
    out["feat_top1_etf_m"] = _map_ticker_col_feature(df, {t: c.replace("_d", "_m") for t, c in ticker_to_d.items()})
    return out


def _build_top3_dmean_feature(df: pd.DataFrame, ticker_to_top3d: dict[str, list[str]]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if not ticker_to_top3d:
        return out
    for ticker, cols in ticker_to_top3d.items():
        valid = [c for c in cols if c in df.columns]
        if not valid:
            continue
        mask = df["ticker"] == ticker
        if not mask.any():
            continue
        out.loc[mask] = df.loc[mask, valid].mean(axis=1).to_numpy(dtype=float)
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
                ann_ret = mu * 252.0
                ann_vol = sd * np.sqrt(252.0)
                sharpe = (mu / sd) * np.sqrt(252.0) if sd > 0 else np.nan
                cum = float((1.0 + d["ret"]).prod() - 1.0)
                rows.append(
                    {
                        "signal": sig,
                        "bet_col": bet,
                        "quantile": q,
                        "days": len(d),
                        "avg_n_names": float(d["n_names"].mean()),
                        "ann_return": ann_ret,
                        "ann_vol": ann_vol,
                        "sharpe": sharpe,
                        "cum_return": cum,
                    }
                )
    return pd.DataFrame(rows).sort_values(["quantile", "sharpe"], ascending=[True, False]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore new baseline+ETF model variants (no costs).")
    p.add_argument("--stock-panel", type=Path, default=Path("data/processed/stock_har_window.parquet"))
    p.add_argument("--etf-features", type=Path, default=Path("data/processed/etf15_har_by_date.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/model_exploration/new_candidates_v2"))
    p.add_argument("--train-years", type=int, default=10)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--min-test-year", type=int, default=2006)
    p.add_argument("--max-test-year", type=int, default=2023)
    p.add_argument("--min-train-rows", type=int, default=20000)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--limit-windows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stock = pd.read_parquet(args.stock_panel)
    etf = pd.read_parquet(args.etf_features)
    stock["date"] = pd.to_datetime(stock["date"])
    etf["date"] = pd.to_datetime(etf["date"])
    etf_cols = [c for c in etf.columns if c != "date" and c.endswith(("_d", "_w", "_m"))]

    base = ["r_d", "r_w", "r_m"]
    years = sorted(stock["year"].dropna().astype(int).unique().tolist())
    windows = rolling_windows(years, train_years=args.train_years, test_years=args.test_years, step_years=args.step_years)
    windows = [w for w in windows if w.test_start_year >= args.min_test_year and w.test_end_year <= args.max_test_year]
    if args.limit_windows is not None:
        windows = windows[: args.limit_windows]

    all_preds: list[pd.DataFrame] = []
    logs: list[dict[str, object]] = []

    for idx, w in enumerate(windows, start=1):
        print(f"[window {idx}/{len(windows)}] {w.label}")
        sw = stock[(stock["year"] >= w.train_start_year) & (stock["year"] <= w.test_end_year)].copy()
        if sw.empty:
            continue
        ew = etf[(etf["date"] >= sw["date"].min()) & (etf["date"] <= sw["date"].max())].copy()
        dfw = sw.merge(ew, on="date", how="left")
        train = dfw[dfw["year"].between(w.train_start_year, w.train_end_year)].copy()
        test = dfw[dfw["year"].between(w.test_start_year, w.test_end_year)].copy()
        if train.empty or test.empty:
            continue

        raw_etf_cols_w, usable_etfs_w = select_window_etf_feature_cols(train, test, etf_cols)
        d_cols_w = _daily_cols(raw_etf_cols_w)

        req = ["date", "ticker", "ret", "bet_size_equal"] + base + raw_etf_cols_w
        keep = _eligible_tickers_train_test(
            train[req].copy(),
            test[req].copy(),
            required_cols=["ret", "bet_size_equal"] + base + raw_etf_cols_w,
        )
        tr = train[train["ticker"].isin(keep)].dropna(subset=base + raw_etf_cols_w + ["ret"]).copy()
        te = test[test["ticker"].isin(keep)].dropna(subset=base + raw_etf_cols_w + ["ret"]).copy()
        if tr.empty or te.empty:
            continue

        # Per-window mappings from train only.
        top1_map: dict[str, str] = {}
        top3_map: dict[str, list[str]] = {}
        if d_cols_w:
            top1_map = {t: cols[0] for t, cols in _pick_ticker_topk_etf_d(tr, d_cols_w, k=1, min_obs=80).items()}
            top3_map = _pick_ticker_topk_etf_d(tr, d_cols_w, k=3, min_obs=80)

        tr2 = tr.copy().reset_index(drop=True)
        te2 = te.copy().reset_index(drop=True)
        if top1_map:
            ftr = _build_top1_har_features(tr2, top1_map)
            fte = _build_top1_har_features(te2, top1_map)
            tr2 = pd.concat([tr2, ftr.reset_index(drop=True)], axis=1)
            te2 = pd.concat([te2, fte.reset_index(drop=True)], axis=1)
            tr2["feat_top1_x_rd"] = tr2["feat_top1_etf_d"] * tr2["r_d"]
            tr2["feat_top1_x_rw"] = tr2["feat_top1_etf_w"] * tr2["r_w"]
            tr2["feat_top1_x_rm"] = tr2["feat_top1_etf_m"] * tr2["r_m"]
            te2["feat_top1_x_rd"] = te2["feat_top1_etf_d"] * te2["r_d"]
            te2["feat_top1_x_rw"] = te2["feat_top1_etf_w"] * te2["r_w"]
            te2["feat_top1_x_rm"] = te2["feat_top1_etf_m"] * te2["r_m"]

        # Broad market/idiosyncratic terms.
        if {"SPY_d", "SPY_w", "SPY_m"}.issubset(set(raw_etf_cols_w)):
            tr2["feat_idio_d"] = tr2["r_d"] - tr2["SPY_d"]
            tr2["feat_idio_w"] = tr2["r_w"] - tr2["SPY_w"]
            tr2["feat_idio_m"] = tr2["r_m"] - tr2["SPY_m"]
            te2["feat_idio_d"] = te2["r_d"] - te2["SPY_d"]
            te2["feat_idio_w"] = te2["r_w"] - te2["SPY_w"]
            te2["feat_idio_m"] = te2["r_m"] - te2["SPY_m"]

        # Top3 ETF mean daily feature.
        if top3_map:
            tr2["feat_top3_etf_dmean"] = _build_top3_dmean_feature(tr2, top3_map)
            te2["feat_top3_etf_dmean"] = _build_top3_dmean_feature(te2, top3_map)

        models: dict[str, list[str]] = {}
        # M1: baseline + top1 ETF HAR.
        m1 = ["feat_top1_etf_d", "feat_top1_etf_w", "feat_top1_etf_m"]
        if all(c in tr2.columns for c in m1):
            models["signal_baseline+top1etfhar_tick"] = base + m1
        # M2: M1 + interaction terms.
        m2 = m1 + ["feat_top1_x_rd", "feat_top1_x_rw", "feat_top1_x_rm"]
        if all(c in tr2.columns for c in m2):
            models["signal_baseline+top1etfhar_interact"] = base + m2
        # M3: baseline + SPY HAR + idio HAR.
        m3 = ["SPY_d", "SPY_w", "SPY_m", "feat_idio_d", "feat_idio_w", "feat_idio_m"]
        if all(c in tr2.columns for c in m3):
            models["signal_baseline+spy+idio"] = base + m3
        # M4: baseline + top3 ETF daily mean.
        m4 = ["feat_top3_etf_dmean"]
        if all(c in tr2.columns for c in m4):
            models["signal_baseline+top3etf_dmean"] = base + m4
        # M5: M3 + top3 ETF mean.
        if all(c in tr2.columns for c in (m3 + m4)):
            models["signal_baseline+spy+idio+top3d"] = base + m3 + m4

        pred_parts: list[pd.DataFrame] = []
        for name, feats in models.items():
            trm = tr2.dropna(subset=feats + ["ret"]).copy()
            tem = te2.dropna(subset=feats + ["ret"]).copy()
            ok = len(trm) >= args.min_train_rows and not tem.empty
            logs.append(
                {
                    "window": w.label,
                    "model": name,
                    "ok": bool(ok),
                    "train_rows": int(len(trm)),
                    "test_rows": int(len(tem)),
                    "n_etfs_used": int(len(usable_etfs_w)),
                    "n_top1_map": int(len(top1_map)),
                    "n_top3_map": int(len(top3_map)),
                }
            )
            if not ok:
                continue
            tem = tem.copy()
            tem[name] = _fit_predict_ridge(trm, tem, feats, alpha=args.ridge_alpha)
            pred_parts.append(
                tem[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", name]].rename(columns={"ret": "target_ret"})
            )

        if pred_parts:
            merged = None
            for p in pred_parts:
                if merged is None:
                    merged = p.copy()
                else:
                    merged = merged.merge(
                        p.drop(columns=["target_ret", "bet_size_equal", "bet_size_mktcap_lag"], errors="ignore"),
                        on=["date", "ticker"],
                        how="outer",
                    )
            if merged is not None:
                all_preds.append(merged)

    if not all_preds:
        raise RuntimeError("No predictions generated.")

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all = pred_all.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    signal_cols = [c for c in pred_all.columns if c.startswith("signal_")]

    pred_all.to_parquet(out_dir / "predictions_new_models.parquet", index=False)
    pd.DataFrame(logs).to_csv(out_dir / "window_log_new_models.csv", index=False)

    metrics = _evaluate_gross_no_cost(
        pred_all,
        signal_cols=signal_cols,
        quantiles=[1.0, 0.25],
        bet_cols=["bet_size_equal", "bet_size_mktcap_lag"],
    )
    metrics.to_csv(out_dir / "gross_sharpe_new_models.csv", index=False)
    print(f"[done] {out_dir / 'gross_sharpe_new_models.csv'}")
    print(metrics.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
