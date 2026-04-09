from __future__ import annotations

import argparse
import gzip
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


ETF10 = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
ETF5 = ["IWM", "EEM", "TLT", "USO", "GLD"]
ETF15 = ETF10 + ETF5

HORIZONS = ("d", "w", "m")
WEEK_WIN = 5
MONTH_WIN = 22


@dataclass(frozen=True)
class WindowSpec:
    train_start_year: int
    train_end_year: int
    test_start_year: int
    test_end_year: int

    @property
    def label(self) -> str:
        return (
            f"train_{self.train_start_year}_{self.train_end_year}"
            f"__test_{self.test_start_year}_{self.test_end_year}"
        )


def project_root() -> Path:
    here = Path.cwd()
    if (here / "data").exists():
        return here
    if here.name == "src" and (here.parent / "data").exists():
        return here.parent
    return here


def list_year_zip_paths(raw_dir: Path, min_year: int | None = None, max_year: int | None = None) -> list[Path]:
    out: list[Path] = []
    for p in raw_dir.glob("*.zip"):
        if not re.fullmatch(r"(19|20)\d{2}\.zip", p.name):
            continue
        y = int(p.stem)
        if min_year is not None and y < min_year:
            continue
        if max_year is not None and y > max_year:
            continue
        out.append(p)
    return sorted(out, key=lambda x: int(x.stem))


def _choose_daily_csv_members(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """
    Returns sorted list of (yyyymmdd, member_name), preferring .csv.gz over .csv if both exist.
    """
    by_day: dict[str, str] = {}
    for member in z.namelist():
        base = Path(member).name
        m = re.fullmatch(r"(\d{8})\.csv(\.gz)?", base)
        if not m:
            continue
        day = m.group(1)
        current = by_day.get(day)
        if current is None or base.endswith(".gz"):
            by_day[day] = member
    return sorted(by_day.items())


def _read_member_csv(z: zipfile.ZipFile, member: str, usecols: list[str]) -> pd.DataFrame:
    raw = z.read(member)
    if member.endswith(".gz"):
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in usecols)


def extract_year_daily_returns(zip_path: Path) -> pd.DataFrame:
    """
    Extract minimal long-format data from a yearly zip.
    Keeps all instruments to support rolling universe.
    """
    usecols = ["ticker", "date", "pvCLCL", "volume", "close", "sharesOut"]
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for day, member in _choose_daily_csv_members(z):
            try:
                df = _read_member_csv(z, member, usecols=usecols)
            except Exception:
                # Some days/files may have slightly different schema; fall back to required cols only.
                df = _read_member_csv(z, member, usecols=["ticker", "pvCLCL"])
                df["date"] = int(day)
                for c in ("volume", "close", "sharesOut"):
                    df[c] = np.nan

            if "date" not in df.columns:
                df["date"] = int(day)

            keep_cols = ["ticker", "date", "pvCLCL", "volume", "close", "sharesOut"]
            for c in keep_cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[keep_cols]
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "ret", "volume", "close", "sharesOut"])

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"pvCLCL": "ret"})
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"].astype(str), format="%Y%m%d", errors="coerce")
    for c in ("ret", "volume", "close", "sharesOut"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["ticker", "date"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def build_year_caches(
    raw_dir: Path,
    cache_dir: Path,
    min_year: int | None = None,
    max_year: int | None = None,
    overwrite: bool = False,
) -> list[int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    years_done: list[int] = []
    for zp in list_year_zip_paths(raw_dir, min_year=min_year, max_year=max_year):
        year = int(zp.stem)
        out_path = cache_dir / f"daily_min_{year}.parquet"
        if out_path.exists() and not overwrite:
            years_done.append(year)
            continue
        print(f"[build-year] {year}: reading {zp.name}")
        df = extract_year_daily_returns(zp)
        df.to_parquet(out_path, index=False)
        years_done.append(year)
        print(f"[build-year] {year}: rows={len(df):,} tickers={df['ticker'].nunique():,}")
    return years_done


def _add_har_lags(df: pd.DataFrame, group_col: str = "ticker", ret_col: str = "ret") -> pd.DataFrame:
    df = df.sort_values([group_col, "date"]).copy()
    g = df.groupby(group_col, sort=False)[ret_col]
    df["r_d"] = g.shift(1)
    df["r_w"] = (
        g.rolling(WEEK_WIN, min_periods=WEEK_WIN)
        .mean()
        .reset_index(level=0, drop=True)
        .shift(1)
    )
    df["r_m"] = (
        g.rolling(MONTH_WIN, min_periods=MONTH_WIN)
        .mean()
        .reset_index(level=0, drop=True)
        .shift(1)
    )
    return df


def build_feature_tables(
    cache_dir: Path,
    out_dir: Path,
    min_year: int | None = None,
    max_year: int | None = None,
) -> tuple[Path, Path]:
    files = sorted(cache_dir.glob("daily_min_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No cached yearly parquet files found under {cache_dir}")

    frames = []
    for p in files:
        y = int(p.stem.split("_")[-1])
        if min_year is not None and y < min_year:
            continue
        if max_year is not None and y > max_year:
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        raise ValueError("No yearly cache files matched the requested year range.")

    daily = pd.concat(frames, ignore_index=True)
    daily = daily.drop_duplicates(subset=["ticker", "date"], keep="last")
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)
    daily["is_etf15"] = daily["ticker"].isin(ETF15)

    # ETF features by date
    etf = daily[daily["is_etf15"]].copy()
    etf = _add_har_lags(etf, group_col="ticker", ret_col="ret")
    etf = etf.rename(columns={"r_d": "d_lag", "r_w": "w_lag", "r_m": "m_lag"})
    etf_feats = etf[["date", "ticker", "d_lag", "w_lag", "m_lag"]].copy()
    etf_pivot = etf_feats.pivot(index="date", columns="ticker")
    etf_pivot.columns = [
        f"{ticker}_{feat}".replace("d_lag", "d").replace("w_lag", "w").replace("m_lag", "m")
        for feat, ticker in etf_pivot.columns
    ]
    etf_pivot = etf_pivot.reset_index().sort_values("date").reset_index(drop=True)

    # Stock HAR panel (keep rows even if ETF features are missing; model-specific filter happens later)
    stock = daily[~daily["is_etf15"]].copy()
    stock = _add_har_lags(stock, group_col="ticker", ret_col="ret")
    stock["mktcap"] = stock["close"] * stock["sharesOut"]
    stock["bet_size_equal"] = 1.0
    stock["bet_size_mktcap_lag"] = stock.groupby("ticker", sort=False)["mktcap"].shift(1)
    stock["year"] = stock["date"].dt.year

    stock_panel = stock[
        [
            "ticker",
            "date",
            "year",
            "ret",
            "r_d",
            "r_w",
            "r_m",
            "volume",
            "close",
            "sharesOut",
            "mktcap",
            "bet_size_equal",
            "bet_size_mktcap_lag",
        ]
    ].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    stock_out = out_dir / "stock_har_window.parquet"
    etf_out = out_dir / "etf15_har_by_date.parquet"
    stock_panel.to_parquet(stock_out, index=False)
    etf_pivot.to_parquet(etf_out, index=False)

    print(
        f"[build-features] stock rows={len(stock_panel):,}, tickers={stock_panel['ticker'].nunique():,}, "
        f"dates={stock_panel['date'].nunique():,}, range={stock_panel['date'].min()} -> {stock_panel['date'].max()}"
    )
    print(
        f"[build-features] etf rows={len(etf_pivot):,}, "
        f"range={etf_pivot['date'].min()} -> {etf_pivot['date'].max()}"
    )
    return stock_out, etf_out


def etf_feature_cols(etf_df: pd.DataFrame) -> list[str]:
    return [c for c in etf_df.columns if c != "date" and c.endswith(("_d", "_w", "_m"))]


def _get_etf_list_from_cols(cols: Iterable[str]) -> list[str]:
    return sorted({c.split("_")[0] for c in cols})


def _split_etf_cols(cols: Iterable[str]) -> tuple[list[str], dict[str, list[str]]]:
    etfs = _get_etf_list_from_cols(cols)
    by_h = {h: [f"{e}_{h}" for e in etfs if f"{e}_{h}" in cols] for h in HORIZONS}
    return etfs, by_h


def select_window_etf_feature_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_etf_cols: list[str],
    min_train_cov: float = 0.95,
    min_test_cov: float = 0.95,
) -> tuple[list[str], list[str]]:
    """
    Select ETF features dynamically per window at ETF granularity (keep d/w/m triplets together).
    This prevents early windows from being dropped just because a later ETF had not launched yet.
    """
    etfs, by_h = _split_etf_cols(candidate_etf_cols)
    if not etfs:
        return [], []

    tr_dates = (
        train_df[["date"] + candidate_etf_cols]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    te_dates = (
        test_df[["date"] + candidate_etf_cols]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    usable_etfs: list[str] = []
    for e in etfs:
        cols_e = [f"{e}_{h}" for h in HORIZONS if f"{e}_{h}" in candidate_etf_cols]
        if len(cols_e) != 3:
            continue
        tr_cov = float(tr_dates[cols_e].notna().all(axis=1).mean()) if len(tr_dates) else 0.0
        te_cov = float(te_dates[cols_e].notna().all(axis=1).mean()) if len(te_dates) else 0.0
        if tr_cov >= min_train_cov and te_cov >= min_test_cov:
            usable_etfs.append(e)

    usable_cols = [f"{e}_{h}" for e in usable_etfs for h in HORIZONS]
    return usable_cols, usable_etfs


def build_stock_etf_weights_corr(
    train_df: pd.DataFrame,
    etf_ret_cols: list[str],
    min_obs: int = 120,
    winsor: float = 0.01,
) -> pd.DataFrame:
    etfs, by_h = _split_etf_cols(etf_ret_cols)
    etf_d_cols = by_h["d"]
    if len(etf_d_cols) != len(etfs):
        raise ValueError("ETF daily columns are incomplete for W construction.")

    etf_by_date = (
        train_df[["date"] + etf_d_cols]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .set_index("date")
    )
    stock_ret = (
        train_df[["date", "ticker", "ret"]]
        .pivot(index="date", columns="ticker", values="ret")
        .sort_index()
    )

    common_dates = stock_ret.index.intersection(etf_by_date.index)
    stock_ret = stock_ret.loc[common_dates]
    etf_by_date = etf_by_date.loc[common_dates]
    if stock_ret.empty:
        return pd.DataFrame(columns=etfs, dtype=float)

    W = pd.DataFrame(index=stock_ret.columns, columns=etfs, dtype=float)
    etf_mat = etf_by_date[etf_d_cols].to_numpy()
    etf_mean = np.nanmean(etf_mat, axis=0)
    etf_std = np.nanstd(etf_mat, axis=0, ddof=1)

    for ticker in stock_ret.columns:
        x = stock_ret[ticker].to_numpy()
        mask = np.isfinite(x)
        if mask.sum() < min_obs:
            continue
        x_m = x[mask]
        x_std = np.std(x_m, ddof=1)
        if not np.isfinite(x_std) or x_std == 0:
            continue
        y = etf_mat[mask, :]
        cov = np.nanmean((x_m - x_m.mean())[:, None] * (y - etf_mean), axis=0)
        corr = cov / (x_std * etf_std)
        W.loc[ticker, :] = corr

    W = W.replace([np.inf, -np.inf], np.nan)
    if winsor > 0:
        vals = W.to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo = np.quantile(vals, winsor)
            hi = np.quantile(vals, 1 - winsor)
            W = W.clip(lower=lo, upper=hi)
    return W.fillna(0.0)


def build_neighbor_weights_from_exposure(W: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    if W.empty:
        return pd.DataFrame(dtype=float)
    X = W.fillna(0.0).to_numpy()
    sims = cosine_similarity(X)
    np.fill_diagonal(sims, -np.inf)
    n = sims.shape[0]
    k_eff = min(k, max(n - 1, 0))
    weights = np.zeros_like(sims, dtype=float)
    if k_eff == 0:
        return pd.DataFrame(weights, index=W.index, columns=W.index)

    for i in range(n):
        idx = np.argpartition(sims[i], -k_eff)[-k_eff:]
        idx = idx[np.isfinite(sims[i, idx])]
        if len(idx) == 0:
            continue
        weights[i, idx] = 1.0

    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights = weights / row_sums
    return pd.DataFrame(weights, index=W.index, columns=W.index)


def make_neighbor_lag_feature(window_df: pd.DataFrame, nbr_w: pd.DataFrame) -> pd.DataFrame:
    out = window_df[["date", "ticker"]].copy()
    if nbr_w.empty:
        out["feat_neighbor_lag1"] = np.nan
        return out

    ret_wide = (
        window_df[["date", "ticker", "ret"]]
        .pivot(index="date", columns="ticker", values="ret")
        .sort_index()
    )
    valid = nbr_w.index.intersection(ret_wide.columns)
    if len(valid) == 0:
        out["feat_neighbor_lag1"] = np.nan
        return out

    ret_mat = ret_wide[valid].to_numpy()
    w_mat = nbr_w.loc[valid, valid].to_numpy()
    nbr_ret = ret_mat @ w_mat.T
    nbr_lag = pd.DataFrame(nbr_ret, index=ret_wide.index, columns=valid).shift(1)
    nbr_long = nbr_lag.stack(dropna=False).rename("feat_neighbor_lag1").reset_index()
    out = out.merge(nbr_long, on=["date", "ticker"], how="left")
    return out


def rolling_windows(
    available_years: list[int],
    train_years: int,
    test_years: int,
    step_years: int,
) -> list[WindowSpec]:
    years = sorted(set(available_years))
    if not years:
        return []
    min_y, max_y = years[0], years[-1]
    windows: list[WindowSpec] = []
    test_start = min_y + train_years
    while test_start + test_years - 1 <= max_y:
        windows.append(
            WindowSpec(
                train_start_year=test_start - train_years,
                train_end_year=test_start - 1,
                test_start_year=test_start,
                test_end_year=test_start + test_years - 1,
            )
        )
        test_start += step_years
    return windows


def _slice_coverage_tickers(
    df: pd.DataFrame,
    required_cols: list[str],
    min_coverage: float = 0.95,
    min_dates: int = 60,
) -> pd.Index:
    """
    Rolling-universe friendly filter:
    keep tickers with sufficiently high coverage within the slice, rather than
    requiring perfect coverage across every date.
    """
    if df.empty:
        return pd.Index([], dtype=object)
    n_dates = int(df["date"].nunique())
    required_n = max(min_dates, int(np.ceil(n_dates * min_coverage)))
    ok = df[required_cols].notna().all(axis=1)
    stats = (
        df.assign(_ok=ok)
        .groupby("ticker", sort=False)
        .agg(n_ok=("_ok", "sum"))
    )
    return stats.index[stats["n_ok"] >= required_n]


def _eligible_tickers_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    required_cols: list[str],
    train_coverage: float = 0.95,
    test_coverage: float = 0.95,
    min_train_dates: int = 120,
    min_test_dates: int = 20,
) -> pd.Index:
    train_keep = _slice_coverage_tickers(
        train_df,
        required_cols=required_cols,
        min_coverage=train_coverage,
        min_dates=min_train_dates,
    )
    test_keep = _slice_coverage_tickers(
        test_df,
        required_cols=required_cols,
        min_coverage=test_coverage,
        min_dates=min_test_dates,
    )
    return train_keep.intersection(test_keep)


def _fit_predict_ridge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str = "ret",
    alpha: float = 10.0,
) -> np.ndarray:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[features])
    X_te = scaler.transform(test_df[features])
    y_tr = train_df[target_col].to_numpy()
    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _proxy_daily_pnl(
    df_pred: pd.DataFrame,
    signal_col: str,
    target_col: str = "target_ret",
    top_frac: float = 0.2,
) -> pd.DataFrame:
    rows = []
    for dt, g in df_pred.groupby("date", sort=True):
        g = g[[signal_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) < 10:
            continue
        q = max(1, int(len(g) * top_frac))
        g = g.sort_values(signal_col)
        short = g.head(q)[target_col].mean()
        long = g.tail(q)[target_col].mean()
        pnl = float(long - short)
        rows.append({"date": dt, "pnl_ls_eq": pnl, "n_names": len(g), "bucket_n": q})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["cum_pnl_ls_eq"] = out["pnl_ls_eq"].cumsum()
    return out


def run_window_slicing_models(
    stock_panel_path: Path,
    etf_features_path: Path,
    output_dir: Path,
    train_years: int = 8,
    test_years: int = 1,
    step_years: int = 1,
    ridge_alpha: float = 10.0,
    neighbor_k: int = 5,
    min_train_rows: int = 2000,
    limit_windows: int | None = None,
    min_test_year: int | None = None,
    max_test_year: int | None = None,
) -> None:
    stock = pd.read_parquet(stock_panel_path)
    etf = pd.read_parquet(etf_features_path)
    stock["date"] = pd.to_datetime(stock["date"])
    etf["date"] = pd.to_datetime(etf["date"])
    etf_cols = etf_feature_cols(etf)

    base_features = ["r_d", "r_w", "r_m"]
    model_baseline = "baseline"
    model_baseline_etf = "baseline+etf"
    model_baseline_network = "baseline+network"
    model_baseline_etf_network = "baseline+etf+network"

    sig_baseline = f"signal_{model_baseline}"
    sig_baseline_etf = f"signal_{model_baseline_etf}"
    sig_baseline_network = f"signal_{model_baseline_network}"
    sig_baseline_etf_network = f"signal_{model_baseline_etf_network}"

    years = sorted(stock["year"].dropna().astype(int).unique().tolist())
    windows = rolling_windows(years, train_years=train_years, test_years=test_years, step_years=step_years)
    if min_test_year is not None:
        windows = [w for w in windows if w.test_start_year >= min_test_year]
    if max_test_year is not None:
        windows = [w for w in windows if w.test_end_year <= max_test_year]
    if limit_windows is not None:
        windows = windows[:limit_windows]
    if not windows:
        raise ValueError("No windows generated. Check year range and window sizes.")

    output_dir.mkdir(parents=True, exist_ok=True)
    daily_combined_dir = output_dir / "alphamark_daily_combined"
    per_model_dir = output_dir / "alphamark_daily_by_model"
    daily_combined_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)

    all_preds: list[pd.DataFrame] = []
    window_log_rows: list[dict[str, object]] = []

    for idx, w in enumerate(windows, start=1):
        print(f"[window {idx}/{len(windows)}] {w.label}")
        mask_window = (stock["year"] >= w.train_start_year) & (stock["year"] <= w.test_end_year)
        sw = stock.loc[mask_window].copy()
        if sw.empty:
            continue

        ew = etf[(etf["date"] >= sw["date"].min()) & (etf["date"] <= sw["date"].max())].copy()
        dfw = sw.merge(ew, on="date", how="left")
        train = dfw[(dfw["year"] >= w.train_start_year) & (dfw["year"] <= w.train_end_year)].copy()
        test = dfw[(dfw["year"] >= w.test_start_year) & (dfw["year"] <= w.test_end_year)].copy()
        if train.empty or test.empty:
            continue

        raw_etf_cols_w, usable_etfs_w = select_window_etf_feature_cols(train, test, etf_cols)
        raw_features_w = base_features + raw_etf_cols_w
        print(f"  usable ETFs for ETF/network models: {len(usable_etfs_w)} -> {usable_etfs_w}")

        preds_window: list[pd.DataFrame] = []

        # Baseline model (rolling universe for this model)
        req_base = ["date", "ticker", "ret", "bet_size_equal"] + base_features
        keep_base = _eligible_tickers_train_test(
            train[req_base].copy(),
            test[req_base].copy(),
            required_cols=["ret", "bet_size_equal"] + base_features,
        )
        train_b = train[train["ticker"].isin(keep_base)].dropna(subset=base_features + ["ret"]).copy()
        test_b = test[test["ticker"].isin(keep_base)].dropna(subset=base_features + ["ret"]).copy()
        if len(train_b) >= min_train_rows and not test_b.empty:
            test_b = test_b.copy()
            test_b[sig_baseline] = _fit_predict_ridge(train_b, test_b, base_features, alpha=ridge_alpha)
            preds_window.append(
                test_b[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", sig_baseline]]
                .rename(columns={"ret": "target_ret"})
            )
            window_log_rows.append(
                {"window": w.label, "model": model_baseline, "train_rows": len(train_b), "test_rows": len(test_b), "n_tickers": len(keep_base)}
            )
        else:
            window_log_rows.append({"window": w.label, "model": model_baseline, "train_rows": len(train_b), "test_rows": len(test_b), "n_tickers": len(keep_base), "skipped": True})

        # baseline + ETF model
        req_raw = ["date", "ticker", "ret", "bet_size_equal"] + raw_features_w
        keep_raw = _eligible_tickers_train_test(
            train[req_raw].copy(),
            test[req_raw].copy(),
            required_cols=["ret", "bet_size_equal"] + raw_features_w,
        )
        train_r = train[train["ticker"].isin(keep_raw)].dropna(subset=raw_features_w + ["ret"]).copy()
        test_r = test[test["ticker"].isin(keep_raw)].dropna(subset=raw_features_w + ["ret"]).copy()
        if raw_etf_cols_w and len(train_r) >= min_train_rows and not test_r.empty:
            test_r = test_r.copy()
            test_r[sig_baseline_etf] = _fit_predict_ridge(train_r, test_r, raw_features_w, alpha=ridge_alpha)
            preds_window.append(
                test_r[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", sig_baseline_etf]]
                .rename(columns={"ret": "target_ret"})
            )
            window_log_rows.append(
                {
                    "window": w.label,
                    "model": model_baseline_etf,
                    "train_rows": len(train_r),
                    "test_rows": len(test_r),
                    "n_tickers": len(keep_raw),
                    "n_etfs_used": len(usable_etfs_w),
                }
            )
        else:
            window_log_rows.append(
                {
                    "window": w.label,
                    "model": model_baseline_etf,
                    "train_rows": len(train_r),
                    "test_rows": len(test_r),
                    "n_tickers": len(keep_raw),
                    "n_etfs_used": len(usable_etfs_w),
                    "skipped": True,
                }
            )

        # Network-based models (W needs ETF features):
        # 1) baseline + network
        # 2) baseline + ETF + network
        if raw_etf_cols_w and len(train_r) >= min_train_rows and not test_r.empty:
            window_nr = pd.concat([train_r, test_r], ignore_index=True)
            W = build_stock_etf_weights_corr(
                train_r,
                raw_etf_cols_w,
                min_obs=min(120, max(30, train_r["date"].nunique() // 4)),
            )
            nbr_w = build_neighbor_weights_from_exposure(W, k=neighbor_k)
            nbr_feat = make_neighbor_lag_feature(window_nr, nbr_w)
            window_nr = window_nr.merge(nbr_feat, on=["date", "ticker"], how="left")
            net_features = base_features + ["feat_neighbor_lag1"]
            full_features = raw_features_w + ["feat_neighbor_lag1"]

            train_n = window_nr[window_nr["year"].between(w.train_start_year, w.train_end_year)].dropna(subset=net_features + ["ret"]).copy()
            test_n = window_nr[window_nr["year"].between(w.test_start_year, w.test_end_year)].dropna(subset=net_features + ["ret"]).copy()
            if len(train_n) >= min_train_rows and not test_n.empty:
                test_n = test_n.copy()
                test_n[sig_baseline_network] = _fit_predict_ridge(train_n, test_n, net_features, alpha=ridge_alpha)
                preds_window.append(
                    test_n[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", sig_baseline_network]]
                    .rename(columns={"ret": "target_ret"})
                )
                window_log_rows.append(
                    {
                        "window": w.label,
                        "model": model_baseline_network,
                        "train_rows": len(train_n),
                        "test_rows": len(test_n),
                        "n_tickers": nbr_w.shape[0],
                        "n_etfs_used": len(usable_etfs_w),
                    }
                )
            else:
                window_log_rows.append(
                    {
                        "window": w.label,
                        "model": model_baseline_network,
                        "train_rows": len(train_n),
                        "test_rows": len(test_n),
                        "n_tickers": nbr_w.shape[0],
                        "n_etfs_used": len(usable_etfs_w),
                        "skipped": True,
                    }
                )

            train_fn = window_nr[window_nr["year"].between(w.train_start_year, w.train_end_year)].dropna(subset=full_features + ["ret"]).copy()
            test_fn = window_nr[window_nr["year"].between(w.test_start_year, w.test_end_year)].dropna(subset=full_features + ["ret"]).copy()
            if len(train_fn) >= min_train_rows and not test_fn.empty:
                test_fn = test_fn.copy()
                test_fn[sig_baseline_etf_network] = _fit_predict_ridge(train_fn, test_fn, full_features, alpha=ridge_alpha)
                preds_window.append(
                    test_fn[["date", "ticker", "ret", "bet_size_equal", "bet_size_mktcap_lag", sig_baseline_etf_network]]
                    .rename(columns={"ret": "target_ret"})
                )
                window_log_rows.append(
                    {
                        "window": w.label,
                        "model": model_baseline_etf_network,
                        "train_rows": len(train_fn),
                        "test_rows": len(test_fn),
                        "n_tickers": nbr_w.shape[0],
                        "n_etfs_used": len(usable_etfs_w),
                    }
                )
            else:
                window_log_rows.append(
                    {
                        "window": w.label,
                        "model": model_baseline_etf_network,
                        "train_rows": len(train_fn),
                        "test_rows": len(test_fn),
                        "n_tickers": nbr_w.shape[0],
                        "n_etfs_used": len(usable_etfs_w),
                        "skipped": True,
                    }
                )
        else:
            window_log_rows.append(
                {
                    "window": w.label,
                    "model": model_baseline_network,
                    "train_rows": len(train_r) if "train_r" in locals() else 0,
                    "test_rows": len(test_r) if "test_r" in locals() else 0,
                    "n_tickers": len(keep_raw) if "keep_raw" in locals() else 0,
                    "n_etfs_used": len(usable_etfs_w),
                    "skipped": True,
                }
            )
            window_log_rows.append(
                {
                    "window": w.label,
                    "model": model_baseline_etf_network,
                    "train_rows": len(train_r) if "train_r" in locals() else 0,
                    "test_rows": len(test_r) if "test_r" in locals() else 0,
                    "n_tickers": len(keep_raw) if "keep_raw" in locals() else 0,
                    "n_etfs_used": len(usable_etfs_w),
                    "skipped": True,
                }
            )

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
        raise RuntimeError("No predictions were produced. Try smaller min_train_rows or fewer required features.")

    pred_all = pd.concat(all_preds, ignore_index=True)
    pred_all = pred_all.sort_values(["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")

    # Write combined daily files (AlphaMark-ready)
    signal_cols = [c for c in pred_all.columns if c.startswith("signal_")]
    keep_cols = ["ticker"] + signal_cols + ["target_ret", "bet_size_equal", "bet_size_mktcap_lag"]
    for dt, g in pred_all.groupby("date", sort=True):
        f = daily_combined_dir / f"{pd.Timestamp(dt):%Y%m%d}.csv"
        g_out = g[keep_cols].copy()
        g_out.to_csv(f, index=False)

    # Also write per-model daily files (easier if AlphaMark wants single signal)
    for sig in signal_cols:
        d = per_model_dir / sig
        d.mkdir(parents=True, exist_ok=True)
        for dt, g in pred_all.groupby("date", sort=True):
            cols = ["ticker", sig, "target_ret", "bet_size_equal", "bet_size_mktcap_lag"]
            if g[sig].notna().sum() == 0:
                continue
            g[cols].dropna(subset=[sig]).to_csv(d / f"{pd.Timestamp(dt):%Y%m%d}.csv", index=False)

    pred_all.to_parquet(output_dir / "window_predictions_all.parquet", index=False)
    pd.DataFrame(window_log_rows).to_csv(output_dir / "window_training_log.csv", index=False)

    # Proxy PnL (fallback when AlphaMark package is not available)
    proxy_summary = []
    for sig in signal_cols:
        proxy = _proxy_daily_pnl(pred_all[["date", sig, "target_ret"]].dropna(subset=[sig]).copy(), signal_col=sig)
        if proxy.empty:
            continue
        proxy.to_csv(output_dir / f"proxy_pnl_{sig}.csv", index=False)
        proxy_summary.append(
            {
                "signal": sig,
                "days": len(proxy),
                "mean_daily_pnl_ls_eq": proxy["pnl_ls_eq"].mean(),
                "cum_pnl_ls_eq": proxy["pnl_ls_eq"].sum(),
                "std_daily_pnl_ls_eq": proxy["pnl_ls_eq"].std(ddof=1),
            }
        )
    if proxy_summary:
        pd.DataFrame(proxy_summary).to_csv(output_dir / "proxy_pnl_summary.csv", index=False)

    print(f"[done] predictions: {output_dir / 'window_predictions_all.parquet'}")
    print(f"[done] daily AlphaMark-ready CSVs (combined): {daily_combined_dir}")
    print(f"[done] daily AlphaMark-ready CSVs (per model): {per_model_dir}")


def try_alphamark_import() -> tuple[bool, str]:
    for mod in ("alphamark", "alpha_mark", "AlphaMark"):
        try:
            __import__(mod)
            return True, mod
        except Exception:
            continue
    return False, ""


def main() -> None:
    root = project_root()
    raw_dir_default = root / "data" / "raw"
    cache_dir_default = root / "data" / "processed" / "yearly_cache"
    feat_dir_default = root / "data" / "processed"
    out_dir_default = root / "outputs" / "window_slicing"

    parser = argparse.ArgumentParser(description="Window slicing training + AlphaMark-ready exports")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-data", help="Extract yearly zips and build stock/ETF feature tables")
    p_build.add_argument("--raw-dir", type=Path, default=raw_dir_default)
    p_build.add_argument("--cache-dir", type=Path, default=cache_dir_default)
    p_build.add_argument("--out-dir", type=Path, default=feat_dir_default)
    p_build.add_argument("--min-year", type=int, default=1990)
    p_build.add_argument("--max-year", type=int, default=2023)
    p_build.add_argument("--overwrite", action="store_true")

    p_train = sub.add_parser("run-models", help="Run baseline/raw ETF/network under window slicing")
    p_train.add_argument("--stock-panel", type=Path, default=feat_dir_default / "stock_har_window.parquet")
    p_train.add_argument("--etf-features", type=Path, default=feat_dir_default / "etf15_har_by_date.parquet")
    p_train.add_argument("--output-dir", type=Path, default=out_dir_default)
    p_train.add_argument("--train-years", type=int, default=8)
    p_train.add_argument("--test-years", type=int, default=1)
    p_train.add_argument("--step-years", type=int, default=1)
    p_train.add_argument("--ridge-alpha", type=float, default=10.0)
    p_train.add_argument("--neighbor-k", type=int, default=5)
    p_train.add_argument("--min-train-rows", type=int, default=2000)
    p_train.add_argument("--limit-windows", type=int, default=None)
    p_train.add_argument("--min-test-year", type=int, default=None)
    p_train.add_argument("--max-test-year", type=int, default=None)

    p_check = sub.add_parser("check-alphamark", help="Check whether AlphaMark package is importable")

    args = parser.parse_args()

    if args.cmd == "build-data":
        build_year_caches(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            min_year=args.min_year,
            max_year=args.max_year,
            overwrite=args.overwrite,
        )
        build_feature_tables(
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            min_year=args.min_year,
            max_year=args.max_year,
        )
        return

    if args.cmd == "run-models":
        run_window_slicing_models(
            stock_panel_path=args.stock_panel,
            etf_features_path=args.etf_features,
            output_dir=args.output_dir,
            train_years=args.train_years,
            test_years=args.test_years,
            step_years=args.step_years,
            ridge_alpha=args.ridge_alpha,
            neighbor_k=args.neighbor_k,
            min_train_rows=args.min_train_rows,
            limit_windows=args.limit_windows,
            min_test_year=args.min_test_year,
            max_test_year=args.max_test_year,
        )
        return

    if args.cmd == "check-alphamark":
        ok, mod = try_alphamark_import()
        if ok:
            print(f"AlphaMark import OK via module: {mod}")
        else:
            print("AlphaMark package not found in current environment.")
        return


if __name__ == "__main__":
    main()
