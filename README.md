# ETF-HAR Project

This repository contains a research-style project on daily U.S. equity return prediction using a stock-only HAR baseline, ETF-linked features, and execution-aware strategy evaluation.

## Main idea

The project studies whether ETF-linked information can improve stock-level daily return forecasting once evaluation shifts from statistical fit to economic significance.

The empirical workflow combines:
- a rolling-universe, window-sliced forecasting setup,
- Ridge-based prediction,
- ETF-conditioned signal design,
- AlphaMark evaluation,
- and turnover / transaction-cost sensitivity analysis.

The main conclusion is that large raw ETF feature blocks do not perform well directly, but ETF-linked information can still be useful when introduced through narrower stock-specific signals and evaluated under explicit execution constraints.

## Repository structure

- `src/`
  - core scripts for window slicing, AlphaMark input preparation, signal post-processing, tradability review, and figure generation
- `report/overleaf_project/`
  - the LaTeX source for the final report
- `alpha/`
  - AlphaMark configs and supporting reference material used in the project
- `notebooks/`
  - exploratory notebook material

## Notes

- Large raw data, processed datasets, intermediate outputs, temporary files, and local archives are intentionally excluded from the public repository.
- The vendored local AlphaMark package copy is also excluded to keep the repository lightweight. The public AlphaMark materials and config files used in this project are retained.

## Public-facing focus

This repository is intended to preserve:
- the project logic,
- the modeling and evaluation code,
- and the report source.

It is not intended to be a full data dump of local experimental artifacts.
