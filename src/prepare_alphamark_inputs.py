from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SIGNAL_MAP = {
    "signal_baseline": "pret_baseline",
    "signal_baseline+etf": "pret_baseline+etf",
    "signal_baseline+network": "pret_baseline+network",
    "signal_baseline+etf+network": "pret_baseline+etf+network",
}

LEGACY_SIGNAL_ALIAS = {
    "signal_baseline+etf": "signal_raw_etf",
    "signal_baseline+network": "signal_network",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack daily window-slicing CSVs into AlphaMark PKL files.")
    p.add_argument(
        "--combined-dir",
        type=Path,
        default=Path("outputs/window_slicing/2006_2023_t10_dynamic_etf/alphamark_daily_combined"),
        help="Directory containing daily combined CSV files (YYYYMMDD.csv).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write features_YYYYMMDD.pkl files.",
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
        help="Signal columns to include from combined CSV.",
    )
    p.add_argument("--target-col", type=str, default="target_ret")
    p.add_argument("--target-name", type=str, default="fret_1_MR")
    p.add_argument("--equal-col", type=str, default="bet_size_equal")
    p.add_argument("--equal-name", type=str, default="betsize_cap250k")
    p.add_argument("--mktcap-col", type=str, default="bet_size_mktcap_lag")
    p.add_argument("--mktcap-name", type=str, default="betsize_mktcap_lag")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir: Path = args.combined_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No daily CSV files found under: {in_dir}")

    signal_map = {}
    for s in args.signals:
        if s in DEFAULT_SIGNAL_MAP:
            signal_map[s] = DEFAULT_SIGNAL_MAP[s]
        elif s.startswith("signal_"):
            signal_map[s] = "pret_" + s.replace("signal_", "")
        else:
            signal_map[s] = "pret_" + s

    n_written = 0
    alias_hits: set[tuple[str, str]] = set()
    for fp in files:
        d = pd.read_csv(fp)
        resolved_signals: dict[str, str] = {}
        for wanted in signal_map.keys():
            if wanted in d.columns:
                resolved_signals[wanted] = wanted
            elif wanted in LEGACY_SIGNAL_ALIAS and LEGACY_SIGNAL_ALIAS[wanted] in d.columns:
                resolved_signals[wanted] = LEGACY_SIGNAL_ALIAS[wanted]
                alias_hits.add((wanted, LEGACY_SIGNAL_ALIAS[wanted]))
            else:
                resolved_signals[wanted] = wanted

        required = ["ticker", args.target_col, args.equal_col, args.mktcap_col] + list(resolved_signals.values())
        missing = [c for c in required if c not in d.columns]
        if missing:
            raise ValueError(f"Missing columns in {fp.name}: {missing}")

        out = pd.DataFrame({"ticker": d["ticker"]})
        for wanted, dst in signal_map.items():
            src = resolved_signals[wanted]
            out[dst] = d[src]
        out[args.target_name] = d[args.target_col]
        out[args.equal_name] = d[args.equal_col]
        out[args.mktcap_name] = d[args.mktcap_col]

        out_fp = out_dir / f"features_{fp.stem}.pkl"
        out.to_pickle(out_fp)
        n_written += 1

    print(f"[done] wrote {n_written} files to {out_dir}")
    print(f"[done] first/last: {files[0].name} -> {files[-1].name}")
    print(f"[done] signals: {signal_map}")
    if alias_hits:
        print(f"[done] used legacy signal aliases: {sorted(alias_hits)}")


if __name__ == "__main__":
    main()
