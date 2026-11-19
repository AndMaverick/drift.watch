"""drift.watch: drift detection that knows the difference between drift and sampling noise.

Fit a baseline once, check any later window against it, and get a per-feature verdict that accounts for sample size
(PSI noise floors and p-values), for watching many features at once (Benjamini-Hochberg), and for the changes PSI cannot
see (nulls, values outside the training range, unseen categories, type changes). Includes text and LLM-prompt features,
streaming change detectors, and a CLI whose exit code can gate a pipeline.
"""
from __future__ import annotations

from .baseline import Baseline, CategoricalBaseline, NumericBaseline, fit_categorical, fit_numeric, infer_kind
from .io import read_rows
from .measures import (
    EPS,
    bh_adjust,
    binomial_sf,
    categorical_psi,
    chi2_sf,
    chi_square,
    jensen_shannon,
    kolmogorov_sf,
    ks_stat,
    ks_test,
    psi,
    psi_noise_floor,
    psi_pvalue,
    quantile_edges,
    range_exceed_p,
    report,
    status,
    two_proportion_p,
    wasserstein,
)
from .monitor import DriftReport, FeatureDrift, Policy, assess, check
from .stream import DDM, EWMA, PageHinkley, WindowResult, first_alert, scan
from .text import dominant_script, features, vocabulary

__version__ = "0.2.0"

__all__ = [
    "EPS", "psi", "categorical_psi", "ks_stat", "ks_test", "status", "report", "wasserstein", "jensen_shannon", "chi_square", "chi2_sf",
    "kolmogorov_sf", "psi_noise_floor", "psi_pvalue", "two_proportion_p", "bh_adjust", "binomial_sf", "range_exceed_p", "quantile_edges",
    "Baseline", "NumericBaseline", "CategoricalBaseline", "fit_numeric", "fit_categorical", "infer_kind",
    "Policy", "FeatureDrift", "DriftReport", "check", "assess", "read_rows",
    "PageHinkley", "DDM", "EWMA", "scan", "first_alert", "WindowResult", "features", "dominant_script", "vocabulary", "__version__",
]
