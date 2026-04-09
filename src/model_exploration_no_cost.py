from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from window_slicing_pipeline import (
    _eligible_tickers_train_test,
    _fit_predict_ridge,
    build_neighbor_weights_from_exposure,
    build_stock_etf_weights_corr,
    make_neighbor_lag_feature,
    rolling_windows,
    select_window_etf_feature_cols,
)


def _daily_d_cols(etf_cols: list[str]) -> list[str]:
    return [c for c in etf_cols if c.endswith("_d")]


def _pick_global_top1_etf_d(train_df: pd.DataFrame, d_cols: list[str]) -> str | None:
    best_col = None
    best_abs_corr = -np.inf
    for c in d_cols:
        z = train_df[[c, "ret"]].dropna()
        if len(z) < 500:
            continue
        corr = z[c].corr(z["ret"])
        if pd.isna(corr):
            continue
        ac = abs(float(corr))
        if ac > best_abs_corr:
            best_abs_corr = ac
            best_col = c
    return best_col


def _pick_ticker_top1_etf_d(train_df: pd.DataFrame, d_cols: list[str], min_obs: int = 80) -> dict[str, str]:
    out: dict[str, str] = {}
    for ticker, g in train_df.groupby("ticker", sort=False):
        best_col = None
        best_abs_corr = -np.inf
        for c in d_cols:
            z = g[[c, "ret"]].dropna()
            if len(z) < min_obs:
                continue
            corr = z[c].corr(z["ret"])
            if pd.isna(corr):
                continue
            ac = abs(float(corr))
            if ac > best_abs_corr:
                best_abs_corr = ac
                best_col = c
        if best_col is not None:
            out[ticker] = best_col
    return out


def _add_top1_feature(df: pd.DataFrame, ticker_to_col: dict[str, str], out_col: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if not ticker_to_col:
        return out
    inv: dict[str, list[str]] = {}
    for t, c in ticker_to_col.items():
        inv.setdefault(c, []).append(t)
    for c, tickers in inv.items():
        mask = df["ticker"].isin(tickers)
        if c in df.columns and mask.any():
            out.loc[mask] = df.loc[mask, c].to_numpy()
    return out


def _add_top1_har_features(df: pd.DataFrame, ticker_to_dcol: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["feat_top1_etf_d"] = _add_top1_feature(df, ticker_to_dcol, out_col="feat_top1_etf_d")
    # map *_d -> *_w, *_m for same ETF
    map_w = {t: c.replace("_d", "_w") for t, c in ticker_to_dcol.items()}
    map_m = {t: c.replace("_d", "_m") for t, c in ticker_to_dcol.items()}
    out["feat_top1_etf_w"] = _add_top1_feature(df, map_w, out_col="feat_top1_etf_w")
    out["feat_top1_etf_m"] = _add_top1_feature(df, map_m, out_col="feat_top1_etf_m")
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
    out = pd.DataFrame(rows).sort_values(["quantile", "sharpe"], ascending=[True, False]).reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore extra baseline-based models by gross Sharpe (no costs).")
    p.add_argument("--stock-panel", type=Path, default=Path("data/processed/stock_har_window.parquet"))
    p.add_argument("--etf-features", type=Path, default=Path("data/processed/etf15_har_by_date.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/model_exploration/no_cost_search_v1"))
    p.add_argument("--train-years", type=int, default=10)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--min-test-year", type=int, default=2006)
    p.add_argument("--max-test-year", type=int, default=2023)
    p.add_argument("--min-train-rows", type=int, default=20000)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--neighbor-k", type=int, default=5)
    p.add_argument("--limit-windows", type=int, default=None)
    p.add_argument(
        "--model-set",
        choices=["new_only", "all"],
        default="new_only",
        help="`new_only` skips the 4 already-ran core models and runs only new exploratory specs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_old_models = args.model_set == "all"

    stock = pd.read_parquet(args.stock_panel)
    etf = pd.read_parquet(args.etf_features)
    stock["date"] = pd.to_datetime(stock["date"])
    etf["date"] = pd.to_datetime(etf["date"])
    etf_cols = [c for c in etf.columns if c != "date" and c.endswith(("_d", "_w", "_m"))]

    base_features = ["r_d", "r_w", "r_m"]
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
        raw_features_w = base_features + raw_etf_cols_w
        d_cols_w = _daily_d_cols(raw_etf_cols_w)

        preds_window: list[pd.DataFrame] = []

        # 1) baseline (optional; already run in prior pipeline)
        if run_old_models:
            req_base = ["date", "ticker", "ret", "bet_size_equal"] + base_features
            keep_base = _eligible_tickers_train_test(
                train[req_base].copy(),
                test[req_base].copy(),
                required_cols=["ret", "bet_size_equal"] + base_features,
            )
            train_b = train[train["ticker"].isin(keep_base)].dropna(subset=base_features + ["ret"]).copy()
            test_b = test[test["ticker"].isin(keep_base)].dropna(subset=base_features + ["ret"]).copy()
            if len(train_b) >= args.min_train_rows and not test_b.empty:
                test_b = test_b.copy()
                test_b["signal_baseline"] = _fit_predict_ridge(train_b, test_b, base_features, alpha=args.ridge_alpha)
                preds_window.append(test_b[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline"]].rename(columns={"ret": "target_ret"}))
                logs.append({"window": w.label, "model": "baseline", "ok": True, "n_etfs_used": len(usable_etfs_w), "train_rows": len(train_b), "test_rows": len(test_b)})
            else:
                logs.append({"window": w.label, "model": "baseline", "ok": False, "n_etfs_used": len(usable_etfs_w)})

        # keep for ETF/network style models
        req_raw = ["date", "ticker", "ret", "bet_size_equal"] + raw_features_w
        keep_raw = _eligible_tickers_train_test(
            train[req_raw].copy(),
            test[req_raw].copy(),
            required_cols=["ret", "bet_size_equal"] + raw_features_w,
        )
        train_r = train[train["ticker"].isin(keep_raw)].dropna(subset=raw_features_w + ["ret"]).copy()
        test_r = test[test["ticker"].isin(keep_raw)].dropna(subset=raw_features_w + ["ret"]).copy()

        # 2) baseline+etf (optional; already run in prior pipeline)
        if run_old_models:
            if raw_etf_cols_w and len(train_r) >= args.min_train_rows and not test_r.empty:
                test_e = test_r.copy()
                test_e["signal_baseline+etf"] = _fit_predict_ridge(train_r, test_e, raw_features_w, alpha=args.ridge_alpha)
                preds_window.append(test_e[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline+etf"]].rename(columns={"ret": "target_ret"}))
                logs.append({"window": w.label, "model": "baseline+etf", "ok": True, "n_etfs_used": len(usable_etfs_w), "train_rows": len(train_r), "test_rows": len(test_r)})
            else:
                logs.append({"window": w.label, "model": "baseline+etf", "ok": False, "n_etfs_used": len(usable_etfs_w)})

        # 3) baseline + top1 global ETF_d
        if d_cols_w and len(train_r) >= args.min_train_rows and not test_r.empty:
            best_d = _pick_global_top1_etf_d(train_r, d_cols_w)
            if best_d is not None:
                tr = train_r.dropna(subset=base_features + [best_d, "ret"]).copy()
                te = test_r.dropna(subset=base_features + [best_d, "ret"]).copy()
                if len(tr) >= args.min_train_rows and not te.empty:
                    feats = base_features + [best_d]
                    te = te.copy()
                    te["signal_baseline+top1etf_globald"] = _fit_predict_ridge(tr, te, feats, alpha=args.ridge_alpha)
                    preds_window.append(te[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline+top1etf_globald"]].rename(columns={"ret": "target_ret"}))
                    logs.append({"window": w.label, "model": "baseline+top1etf_globald", "ok": True, "chosen_col": best_d})
                else:
                    logs.append({"window": w.label, "model": "baseline+top1etf_globald", "ok": False, "chosen_col": best_d})
            else:
                logs.append({"window": w.label, "model": "baseline+top1etf_globald", "ok": False})
        else:
            logs.append({"window": w.label, "model": "baseline+top1etf_globald", "ok": False})

        # 4) baseline + top1 ETF_d per ticker
        if d_cols_w and len(train_r) >= args.min_train_rows and not test_r.empty:
            ticker_to_d = _pick_ticker_top1_etf_d(train_r, d_cols_w, min_obs=80)
            tr = train_r.copy()
            te = test_r.copy()
            tr["feat_top1_etf_d"] = _add_top1_feature(tr, ticker_to_d, "feat_top1_etf_d")
            te["feat_top1_etf_d"] = _add_top1_feature(te, ticker_to_d, "feat_top1_etf_d")
            feats = base_features + ["feat_top1_etf_d"]
            tr = tr.dropna(subset=feats + ["ret"])
            te = te.dropna(subset=feats + ["ret"])
            if len(tr) >= args.min_train_rows and not te.empty:
                te = te.copy()
                te["signal_baseline+top1etf_tickerd"] = _fit_predict_ridge(tr, te, feats, alpha=args.ridge_alpha)
                preds_window.append(te[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline+top1etf_tickerd"]].rename(columns={"ret": "target_ret"}))
                logs.append({"window": w.label, "model": "baseline+top1etf_tickerd", "ok": True, "n_ticker_map": len(ticker_to_d)})
            else:
                logs.append({"window": w.label, "model": "baseline+top1etf_tickerd", "ok": False, "n_ticker_map": len(ticker_to_d)})
        else:
            logs.append({"window": w.label, "model": "baseline+top1etf_tickerd", "ok": False})

        # 5,6) network-related models
        if raw_etf_cols_w and len(train_r) >= args.min_train_rows and not test_r.empty:
            window_nr = pd.concat([train_r, test_r], ignore_index=True)
            W = build_stock_etf_weights_corr(
                train_r,
                raw_etf_cols_w,
                min_obs=min(120, max(30, train_r["date"].nunique() // 4)),
            )
            nbr_w = build_neighbor_weights_from_exposure(W, k=args.neighbor_k)
            nbr_feat = make_neighbor_lag_feature(window_nr, nbr_w)
            window_nr = window_nr.merge(nbr_feat, on=["date", "ticker"], how="left")

            # 5) baseline+network (optional; already run in prior pipeline)
            if run_old_models:
                net_feats = base_features + ["feat_neighbor_lag1"]
                trn = window_nr[window_nr["year"].between(w.train_start_year, w.train_end_year)].dropna(subset=net_feats + ["ret"]).copy()
                ten = window_nr[window_nr["year"].between(w.test_start_year, w.test_end_year)].dropna(subset=net_feats + ["ret"]).copy()
                if len(trn) >= args.min_train_rows and not ten.empty:
                    ten = ten.copy()
                    ten["signal_baseline+network"] = _fit_predict_ridge(trn, ten, net_feats, alpha=args.ridge_alpha)
                    preds_window.append(ten[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline+network"]].rename(columns={"ret": "target_ret"}))
                    logs.append({"window": w.label, "model": "baseline+network", "ok": True, "n_tickers_net": nbr_w.shape[0]})
                else:
                    logs.append({"window": w.label, "model": "baseline+network", "ok": False, "n_tickers_net": nbr_w.shape[0]})

            # 6) baseline+etf+network (optional; already run in prior pipeline)
            if run_old_models:
                full_feats = raw_features_w + ["feat_neighbor_lag1"]
                trf = window_nr[window_nr["year"].between(w.train_start_year, w.train_end_year)].dropna(subset=full_feats + ["ret"]).copy()
                tef = window_nr[window_nr["year"].between(w.test_start_year, w.test_end_year)].dropna(subset=full_feats + ["ret"]).copy()
                if len(trf) >= args.min_train_rows and not tef.empty:
                    tef = tef.copy()
                    tef["signal_baseline+etf+network"] = _fit_predict_ridge(trf, tef, full_feats, alpha=args.ridge_alpha)
                    preds_window.append(tef[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline+etf+network"]].rename(columns={"ret": "target_ret"}))
                    logs.append({"window": w.label, "model": "baseline+etf+network", "ok": True, "n_tickers_net": nbr_w.shape[0]})
                else:
                    logs.append({"window": w.label, "model": "baseline+etf+network", "ok": False, "n_tickers_net": nbr_w.shape[0]})

            # 7) baseline + top1 ETF HAR (ticker) + network
            if d_cols_w:
                ticker_to_d = _pick_ticker_top1_etf_d(train_r, d_cols_w, min_obs=80)
                feat_tr = _add_top1_har_features(
                    window_nr[window_nr["year"].between(w.train_start_year, w.train_end_year)].copy(),
                    ticker_to_d,
                )
                feat_te = _add_top1_har_features(
                    window_nr[window_nr["year"].between(w.test_start_year, w.test_end_year)].copy(),
                    ticker_to_d,
                )
                trh = window_nr[window_nr["year"].between(w.train_start_year, w.train_end_year)].copy().reset_index(drop=True)
                teh = window_nr[window_nr["year"].between(w.test_start_year, w.test_end_year)].copy().reset_index(drop=True)
                trh = pd.concat([trh, feat_tr.reset_index(drop=True)], axis=1)
                teh = pd.concat([teh, feat_te.reset_index(drop=True)], axis=1)
                harn_feats = base_features + ["feat_top1_etf_d", "feat_top1_etf_w", "feat_top1_etf_m", "feat_neighbor_lag1"]
                trh = trh.dropna(subset=harn_feats + ["ret"])
                teh = teh.dropna(subset=harn_feats + ["ret"])
                if len(trh) >= args.min_train_rows and not teh.empty:
                    teh = teh.copy()
                    teh["signal_baseline+top1etfhar+network"] = _fit_predict_ridge(trh, teh, harn_feats, alpha=args.ridge_alpha)
                    preds_window.append(teh[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", "signal_baseline+top1etfhar+network"]].rename(columns={"ret": "target_ret"}))
                    logs.append({"window": w.label, "model": "baseline+top1etfhar+network", "ok": True, "n_ticker_map": len(ticker_to_d)})
                else:
                    logs.append({"window": w.label, "model": "baseline+top1etfhar+network", "ok": False, "n_ticker_map": len(ticker_to_d)})
        else:
            if run_old_models:
                logs.append({"window": w.label, "model": "baseline+network", "ok": False})
                logs.append({"window": w.label, "model": "baseline+etf+network", "ok": False})
            logs.append({"window": w.label, "model": "baseline+top1etfhar+network", "ok": False})

        if preds_window:
            merged = None
            for p in preds_window:
                if merged is None:
                    merged = p.copy()
                else:
                    merged = merged.merge(
                        p.drop(columns=["target_ret", "bet_size_equal", "bet_size_mktcap_lag"], errors="ignore"),
                        on=["date", "ticker"],
                        how="outer",
                    )
            if merged is not None:
                merged["window"] = w.label
                all_preds.append(merged)

    if not all_preds:
        raise RuntimeError("No predictions generated in exploration.")

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all = pred_all.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    signal_cols = [c for c in pred_all.columns if c.startswith("signal_")]

    pred_all.to_parquet(out_dir / "exploration_predictions_all.parquet", index=False)
    pd.DataFrame(logs).to_csv(out_dir / "exploration_window_log.csv", index=False)

    metrics = _evaluate_gross_no_cost(
        pred_all,
        signal_cols=signal_cols,
        quantiles=[1.0, 0.25],
        bet_cols=["bet_size_equal", "bet_size_mktcap_lag"],
    )
    metrics.to_csv(out_dir / "exploration_gross_sharpe_ranking.csv", index=False)

    print(f"[done] predictions: {out_dir / 'exploration_predictions_all.parquet'}")
    print(f"[done] window log:  {out_dir / 'exploration_window_log.csv'}")
    print(f"[done] ranking:     {out_dir / 'exploration_gross_sharpe_ranking.csv'}")
    if not metrics.empty:
        print("[top-12 by Sharpe]")
        print(metrics.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
