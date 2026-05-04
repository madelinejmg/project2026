"""
leovetter_batch_analysis.py

Compile per-target LEO-Vetter parquets, derive per-test boolean flags from
stored numeric metrics, classify targets, and produce visualizations designed
for 400+ target / 5 pipeline batch runs.

Usage
-----
    cat = compile_batch_leovetter_catalog(RESULTS_ROOT)
    cat = add_per_test_flags(cat)           # derives LEOVetter_flag_* columns
    cat = derive_classification(cat)        # PC / FA / FP from flags

    # Population-level summary (primary figure for papers/proposals)
    fig1 = plot_population_summary(cat, savepath="summary.png")

    # Per-test failure rate heatmap (diagnostic, pipelines x tests)
    fig2 = plot_flag_rate_heatmap(cat, savepath="flag_rates.png")

    # Pipeline agreement matrix (which pipelines agree/disagree)
    fig3 = plot_pipeline_agreement(cat, savepath="agreement.png")
"""

# from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# FA tests (indices 1-13 in check_ALL_thresholds order)
FA_TESTS = [
    "weak", "invalid_transits", "bad_shape", "non_unique", "chases",
    "dmm", "single_event", "bad_fit", "sinusoidal", "unphysical_duration",
    "asymmetric", "chi", "data_gapped",
]
# FP tests (indices 14-18)
FP_TESTS = ["odd_even", "vshaped", "large", "secondary", "offset"]
ALL_TESTS = FA_TESTS + FP_TESTS

# Canonical LEO-Vetter thresholds (from leo_vetter.thresholds._default_thresholds)
from leo_vetter.thresholds import _default_thresholds as _LV_THRESHOLDS

# _LV_THRESHOLDS = {
#     "MES": 7.1, "N_transit": 3, "SHP": 0.2,
#     "MS1": 0.0, "MS2": 0.0, "MS3": 0.0,
#     "DMM": 0.3, "MS4": 0.0, "FRED": 1.8,
#     "SES": 3.0, "AIC1": 0.0, "AIC2": 0.0,
#     "SIN": 0.85, "DUR_min": 0.5, "DUR_max": 15.0,
#     "ASY": 0.2, "CHI": 2.0, "GAP": 0.5,
#     "OE": 3.0, "V_shape": 1.3, "size": 20.0,
#     "MS5": 0.0, "offset": 3.0,
# }

_CLASS_COLORS = {"PC": "#2196F3", "FA": "#FF9800", "FP": "#E53935"}
_CLASS_ORDER  = ["PC", "FA", "FP"]

PIPELINE_ORDER = ["NEMESIS", "QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Catalog compilation
# ─────────────────────────────────────────────────────────────────────────────
import re
def compile_batch_leovetter_catalog_filtered(
    results_root: str,
    filtered_df: pd.DataFrame,
    tic_id_col: str = "TIC ID",
    glob_pattern: str = "**/TIC*_leovetter_catalog.parquet",
    catalog_tic_id_col: str = "ID",
) -> pd.DataFrame:
    """
    Compile a LEO-Vetter catalog DataFrame restricted to TIC IDs present in
    ``filtered_df``, rather than loading all parquets under ``results_root``.

    Filenames are expected to follow the pipeline convention:
        TIC{tic_id}_{sector_tag}_leovetter_catalog.parquet
    e.g. TIC410153553_S0001toS0027_leovetter_catalog.parquet

    Multiple parquets per TIC (e.g. re-runs with different sector ranges) are
    all included -- no deduplication by TIC ID.

    Parameters
    ----------
    results_root : str
        Batch run root directory (RESULTS_ROOT in batch_runner.py).
    filtered_df : pd.DataFrame
        Pre-filtered target dataframe. Must contain ``tic_id_col``.
    tic_id_col : str
        Column in ``filtered_df`` holding TIC IDs. Default ``"TIC ID"``.
        Expected dtype: int64 or str. Float columns (e.g. 410153553.0)
        are coerced automatically.
    glob_pattern : str
        Glob pattern relative to ``results_root`` for leovetter parquets.
        Default ``"**/TIC*_leovetter_catalog.parquet"``.
    catalog_tic_id_col : str
        TIC identifier column name inside each parquet. Default ``"ID"``.

    Returns
    -------
    pd.DataFrame
        Long-format catalog: one row per (TIC, pipeline), restricted to
        TIC IDs in ``filtered_df``. Empty DataFrame if no matches found.

    Raises
    ------
    FileNotFoundError
        If no leovetter_catalog.parquet files exist under ``results_root``.
    KeyError
        If ``tic_id_col`` is not a column of ``filtered_df``.

    Example
    -------
    >>> filtered = nearby_TOI_MD_df[mask].copy()
    >>> cat = compile_batch_leovetter_catalog_filtered(
    ...     results_root="/path/to/batch_results/saved_results",
    ...     filtered_df=filtered,
    ...     tic_id_col="TIC ID",
    ... )
    >>> print(cat.shape, cat["pipeline"].value_counts())
    """
    if tic_id_col not in filtered_df.columns:
        raise KeyError(f"{tic_id_col!r} not found in filtered_df columns.")

    # Coerce to str robustly: handles int64, float64 (410153553.0), and str
    allowlist: set[str] = set(
        filtered_df[tic_id_col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Glob all available parquets
    all_parquets = sorted(Path(results_root).glob(glob_pattern))
    if not all_parquets:
        raise FileNotFoundError(
            f"No leovetter_catalog.parquet files found under {results_root!r} "
            f"with pattern {glob_pattern!r}"
        )

    # Extract TIC ID from stem: TIC{digits}_{sector_tag}_leovetter_catalog
    def _tic_from_path(p: Path) -> str | None:
        match = re.search(r"TIC(\d+)_", p.stem)
        return match.group(1) if match else None

    # Filter glob results against allowlist -- keep ALL files per TIC
    # (handles re-runs with different sector_tags without silent dedup)
    matched_paths = [p for p in all_parquets if _tic_from_path(p) in allowlist]
    matched_tics  = {_tic_from_path(p) for p in matched_paths}
    missing       = allowlist - matched_tics

    if missing:
        print(
            f"[compile_filtered] WARNING: {len(missing)} TIC IDs in filtered_df "
            f"have no corresponding parquet:\n  "
            + ", ".join(sorted(missing)[:10])
            + ("..." if len(missing) > 10 else "")
        )
    if not matched_paths:
        print("[compile_filtered] No matching parquets found. Returning empty DataFrame.")
        return pd.DataFrame()

    frames = []
    for p in matched_paths:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as e:
            print(f"[compile_filtered] WARNING: could not read {p}: {e}")

    if not frames:
        raise RuntimeError("All matched parquet reads failed.")

    cat = pd.concat(frames, ignore_index=True)
    cat[catalog_tic_id_col] = cat[catalog_tic_id_col].astype(str)

    print(
        f"[compile_filtered] Loaded {len(cat)} rows across {len(frames)} parquets "
        f"({len(matched_tics)} unique TICs, {len(missing)} not found on disk)."
    )
    return cat


def compile_batch_leovetter_catalog(
    results_root: str,
    glob_pattern: str = "**/TIC*_leovetter_catalog.parquet",
    tic_id_col: str = "ID",
) -> pd.DataFrame:
    """
    Glob all per-target leovetter_catalog.parquet files and concatenate.

    Parameters
    ----------
    results_root : str
        Batch run root directory (RESULTS_ROOT in batch_runner.py).
    glob_pattern : str
        Glob pattern relative to results_root.
    tic_id_col : str
        TIC identifier column name. Default "ID".

    Returns
    -------
    pd.DataFrame
        Long-format catalog: one row per (TIC, pipeline).

    Example
    -------
    >>> cat = compile_batch_leovetter_catalog("/path/to/batch_results")
    """
    parquets = sorted(Path(results_root).glob(glob_pattern))
    if not parquets:
        raise FileNotFoundError(
            f"No leovetter_catalog.parquet files found under {results_root!r}"
        )
    frames = []
    for p in parquets:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as e:
            print(f"[compile] WARNING: could not read {p}: {e}")
    if not frames:
        raise RuntimeError("All parquet reads failed.")
    cat = pd.concat(frames, ignore_index=True)
    cat[tic_id_col] = cat[tic_id_col].astype(str)
    return cat


# ─────────────────────────────────────────────────────────────────────────────
# 2. Derive per-test boolean flags from stored numeric metrics
# ─────────────────────────────────────────────────────────────────────────────

def add_per_test_flags_OLD(
    cat: pd.DataFrame,
    thresholds: dict = _LV_THRESHOLDS,
    prefix: str = "LEOVetter_",
    chi_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Derive one boolean flag column per LEO-Vetter test from stored numeric
    metric columns. Column name mappings are resolved against the actual
    output of TCELightCurve.compute_flux_metrics (audited from parquet).

    Resolved name mismatches vs. raw metric keys
    ---------------------------------------------
    dmm       -> LEOVetter_DMM
    SES       -> LEOVetter_max_SES
    SIN       -> LEOVetter_sine_sig   (sinusoidal significance, not amplitude)
    ASY       -> LEOVetter_sig_r      (relative asymmetry proxy)
    GAP       -> LEOVetter_N_gap_0.5  (n transits within 0.5-day gap; >0 = gapped)
    sig_oe    -> derived from odd_dep, even_dep, odd_dep_err, even_dep_err
    offset_qual -> not stored; test silently skipped (returns False)

    CHI note
    --------
    LEOVetter_CHI measures per-transit depth scatter relative to formal
    uncertainty. The default threshold (2.0) fires on virtually all active
    M-dwarfs because stellar variability inflates per-transit scatter. The
    chi_threshold parameter lets you raise this for M-dwarf samples. Set
    chi_threshold=None to use the canonical value from thresholds dict.

    Parameters
    ----------
    cat : pd.DataFrame
        Compiled catalog from compile_batch_leovetter_catalog.
    thresholds : dict
        LEO-Vetter threshold dict. Defaults to canonical values.
    prefix : str
        Column prefix for stored metrics. Default "LEOVetter_".
    chi_threshold : float or None
        Override CHI threshold. None uses thresholds["CHI"] (default 2.0).
        Recommended for M-dwarf samples: 5.0-10.0 depending on activity level.

    Returns
    -------
    pd.DataFrame
        Input catalog with LEOVetter_flag_* columns appended.

    Example
    -------
    >>> cat = add_per_test_flags(cat, chi_threshold=7.0)
    >>> cat.filter(like="LEOVetter_flag_").sum()
    """
    df = cat.copy()
    t  = thresholds
    p  = prefix

    def col(name: str) -> pd.Series:
        """Return metric column or NaN series if absent."""
        full = p + name
        return df[full] if full in df.columns else pd.Series(np.nan, index=df.index)

    # ── FA tests ─────────────────────────────────────────────────────────────

    df["LEOVetter_flag_weak"] = col("MES") < t["MES"]

    df["LEOVetter_flag_invalid_transits"] = (
        (col("new_MES") < t["MES"]) | (col("new_N_transit") < t["N_transit"])
    )

    df["LEOVetter_flag_bad_shape"] = col("SHP") > t["SHP"]

    # non_unique: MS1 = sig_pri/Fred - FA1; MS2/MS3 use sig_ter, sig_pos
    ms1 = (col("sig_pri") / col("Fred").replace(0, np.nan)) - col("FA1")
    ms2 = col("sig_pri") - col("sig_ter") - col("FA2")
    ms3 = col("sig_pri") - col("sig_pos") - col("FA2")
    df["LEOVetter_flag_non_unique"] = (ms1 < t["MS1"]) | (ms2 < t["MS2"]) | (ms3 < t["MS3"])

    # chases: flag = (N_transit <= 5) AND (mean_chases < thresholds["chases"])
    # This test is ONLY active for TCEs with <= 5 observed transits where
    # phased modshift statistics have low power. With N_transit >> 5 (typical
    # for multi-sector M-dwarf runs), this fires on zero rows -- correct behavior.
    # thresholds["chases"] is not in _LV_THRESHOLDS; default from leo_vetter source ~0.8
    df["LEOVetter_flag_chases"] = (
        (col("N_transit") <= 5) & (col("mean_chases") < t["chases"]) #t.get("chases", 0.8))
    )

    # dmm: DMM > threshold (flag = mean depth >> median depth, i.e. one deep outlier)
    # Negative DMM (median > mean) does NOT flag -- no abs()
    df["LEOVetter_flag_dmm"] = col("DMM") > t['DMM']# t.get("DMM", 0.3)

    # single_event: flag = (max_SES / MES > max_SES_to_MES_threshold) AND N_transit <= 10
    # Mirrors thresholds.py: one transit dominates the combined MES for short series
    _ratio = col("max_SES") / col("MES").replace(0, np.nan)
#     df["LEOVetter_flag_single_event"] = (
#         (_ratio > t.get("max_SES_to_MES", 0.8)) & (col("N_transit") <= 10)
#     )
    df["LEOVetter_flag_single_event"] = (
        (_ratio > t["max_SES_to_MES"]) & (col("N_transit") <= 10)
    )

    # bad_fit: AIC/chisqr comparison (trap model vs line model)
    # Use trap_aic / trap_chisqr as the transit model (trapezoidal fit)
    daic = col("trap_aic") - col("line_aic")
    df["LEOVetter_flag_bad_fit"] = (
        col("trap_aic").isna() |
        (col("trap_chisqr") > col("line_chisqr")) |
        (daic > t["AIC1"])
    )

    # sinusoidal: stored as LEOVetter_sine_sig (significance of sinusoidal fit)
    # Threshold: sine_sig > SIN means sinusoid is a better fit than transit
    df["LEOVetter_flag_sinusoidal"] = col("sine_sig") > t["SWEET"] #t.get("SWEET", 0.85)

    # unphysical_duration: LEOVetter_dur is stored in phase units (qtran);
    # LEOVetter_transit_dur is in days -- use that, convert to hours
    dur_hrs = col("transit_dur") * 24.0
#     df["LEOVetter_flag_unphysical_duration"] = (
#         (dur_hrs < t.get("DUR_min", 0.5)) | (dur_hrs > t.get("DUR_max", 15.0))
#     )
    
    df["LEOVetter_flag_unphysical_duration"] = (
        (dur_hrs < t["qtran_lo"]) | (dur_hrs > t["qtran_hi"] )
    )    

    # asymmetric: stored as LEOVetter_sig_r (relative asymmetry statistic)
    df["LEOVetter_flag_asymmetric"] = col("sig_r") > t['ASYM'] #t.get("ASY", 0.2)

    # chi: flag = CHI < threshold  (LOW chi = per-transit SNRs are INCONSISTENT)
    # Mean CHI ~31 on M-dwarfs reflects inflated scatter from stellar activity.
    # The canonical threshold is 2.0; raise for active M-dwarf samples.
    chi_thr = chi_threshold if chi_threshold is not None else t['CHI'] #t.get("CHI", 2.0)
    df["LEOVetter_flag_chi"] = col("CHI") < chi_thr   # NOTE: < not >
    df["LEOVetter_chi_threshold_used"] = chi_thr

    # data_gapped: flag = N_gap_2.0 / N_transit >= frac_gap threshold
    # Uses the 2.0-day gap window (widest), normalized by total transit count.
    # frac_gap default from leo_vetter source ~0.5
    _frac_gap = col("N_gap_2.0") / col("N_transit").replace(0, np.nan)
    df["LEOVetter_flag_data_gapped"] = _frac_gap >= t["frac_gap"] #t.get("frac_gap", 0.5)

    # ── FP tests ─────────────────────────────────────────────────────────────

    # odd_even: derive sig_oe from odd/even depth difference
    # sig_oe = |odd_dep - even_dep| / sqrt(odd_dep_err^2 + even_dep_err^2)
    odd_d  = col("odd_dep")
    eve_d  = col("even_dep")
    odd_e  = col("odd_dep_err")
    eve_e  = col("even_dep_err")
    denom  = np.sqrt(odd_e**2 + eve_e**2)
    sig_oe = (odd_d - eve_d).abs() / denom.replace(0, np.nan)
    df["LEOVetter_flag_odd_even"] = sig_oe > t["OE"] #t.get("OE", 3.0)

    df["LEOVetter_flag_vshaped"] = (col("transit_b") + col("transit_RpRs")) > t["V_shape"]

    df["LEOVetter_flag_large"] = col("Rp") > t["size"]

    # secondary: composite gate
    ms4 = (col("sig_sec") / col("Fred").replace(0, np.nan)) - col("FA2")
    ms5 = col("sig_sec") - col("sig_ter") - col("FA2")
    ms6 = col("sig_sec") - col("sig_pos") - col("FA2")
    albedo_ok = (col("albedo") < 1.0) & (col("dep_sec") < 0.1 * col("dep"))
#     df["LEOVetter_flag_secondary"] = (
#         (ms4 > t["MS4"]) & ((ms5 > t.get("MS5", 0)) | (ms6 > t.get("MS3", 0))) & ~albedo_ok
#     )

    df["LEOVetter_flag_secondary"] = (
        (ms4 > t["MS4"]) & ((ms5 > t["MS5"]) | (ms6 > t["MS6"] )) & ~albedo_ok
    )

    # offset: not stored in parquets -- skip silently
    df["LEOVetter_flag_offset"] = False

    # ── Aggregates ────────────────────────────────────────────────────────────
    fa_flag_cols = [f"LEOVetter_flag_{t_}" for t_ in FA_TESTS if f"LEOVetter_flag_{t_}" in df.columns]
    fp_flag_cols = [f"LEOVetter_flag_{t_}" for t_ in FP_TESTS if f"LEOVetter_flag_{t_}" in df.columns]

    df["LEOVetter_flag_any_FA"] = df[fa_flag_cols].any(axis=1)
    df["LEOVetter_flag_any_FP"] = df[fp_flag_cols].any(axis=1)

    return df


def add_per_test_flags(
    cat: pd.DataFrame,
    thresholds: dict = _LV_THRESHOLDS,
    prefix: str = "LEOVetter_",
    chi_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Derive one boolean flag column per LEO-Vetter test from stored numeric
    metric columns in a compiled batch catalog.

    Design principle
    ----------------
    Every predicate here is a vectorized mirror of the corresponding scalar
    test function in NEMESIS_pipeline_new_temp_4.py (which itself mirrors
    leo_vetter/thresholds.py). Threshold keys and comparison directions are
    taken verbatim from those implementations -- no local overrides.

    The ``thresholds`` argument must be the live object returned by::

        from leo_vetter.thresholds import _default_thresholds

    Do not pass a hand-crafted dict.

    Column naming
    -------------
    All metric columns are expected under the ``prefix`` namespace, i.e.
    the stored parquet column ``LEOVetter_MES`` is accessed as col("MES").
    Output flag columns are written as ``LEOVetter_flag_<test>``.

    Tests that require columns not stored by compute_flux_metrics (e.g.
    offset_qual) produce a constant-False flag column and are documented
    below.

    Metric -> parquet column mapping (audited against compute_flux_metrics)
    -----------------------------------------------------------------------
    MES, new_MES, N_transit, new_N_transit   -- stored directly
    SHP                                       -- stored directly
    sig_pri, sig_sec, sig_ter, sig_pos        -- stored directly
    Fred, FA1, FA2                            -- stored directly
    DMM                                       -- stored as LEOVetter_DMM
    max_SES                                   -- stored as LEOVetter_max_SES
    transit_aic, line_aic                     -- stored directly (bad_fit)
    transit_chisqr, line_chisqr               -- stored directly (bad_fit)
    sine_sig, per                             -- stored directly (sinusoidal)
    transit_aRs, aRs, q, trap_qtran, sig_sec  -- stored directly (unphysical_duration)
    trap_qtran_left/right + errs              -- stored directly (asymmetric)
    CHI                                       -- stored directly
    N_gap_2.0                                 -- stored directly
    odd_dep, even_dep, odd_dep_err, even_dep_err -- stored directly (odd_even)
    transit_b, transit_RpRs                   -- stored directly (vshaped)
    Rp                                        -- stored directly (large)
    albedo, dep_sec, dep                      -- stored directly (secondary)
    mean_chases                               -- stored directly (chases)
    offset_qual                               -- NOT stored; flag hardcoded False

    Parameters
    ----------
    cat : pd.DataFrame
        Compiled catalog from compile_batch_leovetter_catalog. One row per
        (TIC, pipeline).
    thresholds : dict
        Live _default_thresholds from leo_vetter.thresholds. Default is the
        module-level import of _LV_THRESHOLDS.
    prefix : str
        Column prefix for stored metrics. Default "LEOVetter_".
    chi_threshold : float or None
        Override for the CHI threshold only. None uses thresholds["CHI"]
        (canonical value 7.8). For active M-dwarf samples, consider raising
        this; validate against confirmed planets in your sample first.

    Returns
    -------
    pd.DataFrame
        Input catalog with LEOVetter_flag_* columns appended.

    Example
    -------
    >>> from leo_vetter.thresholds import _default_thresholds
    >>> cat = add_per_test_flags(cat, thresholds=_default_thresholds)
    >>> cat.filter(like="LEOVetter_flag_").sum()
    """
    df = cat.copy()
    t  = thresholds
    p  = prefix

    def col(name: str) -> pd.Series:
        """Return metric column or all-NaN Series if absent."""
        full = p + name
        return df[full].copy() if full in df.columns else pd.Series(
            np.nan, index=df.index, dtype=float
        )

    # ── FA test 1: weak ───────────────────────────────────────────────────────
    # flag = MES < thresholds["MES"]
    df["LEOVetter_flag_weak"] = col("MES") < t["MES"]

    # ── FA test 2: invalid_transits ───────────────────────────────────────────
    # flag = (new_MES < MES_thr) OR (new_N_transit < N_transit_thr)
    df["LEOVetter_flag_invalid_transits"] = (
        (col("new_MES") < t["MES"]) | (col("new_N_transit") < t["N_transit"])
    )

    # ── FA test 3: bad_shape ──────────────────────────────────────────────────
    # flag = SHP > thresholds["SHP"]
    df["LEOVetter_flag_bad_shape"] = col("SHP") > t["SHP"]

    # ── FA test 4: non_unique ─────────────────────────────────────────────────
    # MS1 = sig_pri / Fred - FA1;  MS2 = sig_pri - sig_ter - FA2
    # MS3 = sig_pri - sig_pos - FA2
    # flag = (MS1 < MS1_thr) OR (MS2 < MS2_thr) OR (MS3 < MS3_thr)
    _fred_safe = col("Fred").replace(0, np.nan)
    ms1 = (col("sig_pri") / _fred_safe) - col("FA1")
    ms2 = col("sig_pri") - col("sig_ter") - col("FA2")
    ms3 = col("sig_pri") - col("sig_pos") - col("FA2")
    df["LEOVetter_flag_non_unique"] = (
        (ms1 < t["MS1"]) | (ms2 < t["MS2"]) | (ms3 < t["MS3"])
    )

    # ── FA test 5: chases ─────────────────────────────────────────────────────
    # flag = (N_transit <= 5) AND (mean_chases < thresholds["chases"])
    # Fires near-zero for multi-sector targets (N_transit >> 5); correct behavior.
    df["LEOVetter_flag_chases"] = (
        (col("N_transit") <= 5) & (col("mean_chases") < t["chases"])
    )

    # ── FA test 6: dmm ────────────────────────────────────────────────────────
    # flag = DMM > thresholds["DMM"]
    # Negative DMM (median > mean) does NOT flag; no abs().
    df["LEOVetter_flag_dmm"] = col("DMM") > t["DMM"]

    # ── FA test 7: single_event ───────────────────────────────────────────────
    # flag = (max_SES / MES > thresholds["max_SES_to_MES"]) AND (N_transit <= 10)
    _mes_safe  = col("MES").replace(0, np.nan)
    _ses_ratio = col("max_SES") / _mes_safe
    df["LEOVetter_flag_single_event"] = (
        (_ses_ratio > t["max_SES_to_MES"]) & (col("N_transit") <= 10)
    )

    # ── FA test 8: bad_fit ────────────────────────────────────────────────────
    # flag = transit_aic is NaN
    #        OR transit_chisqr > line_chisqr
    #        OR (delta_aic > AIC1  when N_transit <= 10)
    #        OR (delta_aic > AIC2  when N_transit > 10)
    # Mirrors bad_fit() in NEMESIS_pipeline_new_temp_4 exactly.
    _transit_aic = col("transit_aic")
    _line_aic    = col("line_aic")
    _daic        = _transit_aic - _line_aic
    _n           = col("N_transit")
    df["LEOVetter_flag_bad_fit"] = (
        _transit_aic.isna()
        | (col("transit_chisqr") > col("line_chisqr"))
        | ((_daic > t["AIC1"]) & (_n <= 10))
        | ((_daic > t["AIC2"]) & (_n > 10))
    )

    # ── FA test 9: sinusoidal ─────────────────────────────────────────────────
    # flag = (sine_sig > thresholds["SWEET"]) AND (per < 10)
    # Both conditions required; mirrors sinusoidal() in NEMESIS_pipeline_new_temp_4.
    df["LEOVetter_flag_sinusoidal"] = (
        (col("sine_sig") > t["SWEET"]) & (col("per") < 10)
    )

    # ── FA test 10: unphysical_duration ───────────────────────────────────────
    # Mirrors unphysical_duration() in NEMESIS_pipeline_new_temp_4 exactly.
    # No qtran_lo/qtran_hi keys exist in _default_thresholds; the test uses
    # geometric sub-conditions on transit_aRs, aRs, q, trap_qtran, sig_sec.
    #
    # low_aRs    = (transit_aRs < 1.5) OR (aRs < 2)
    # q_over_trap = q / trap_qtran   (NaN if trap_qtran <= 0 or non-finite)
    # long_dur   = (q_over_trap < 0.6) OR (sig_sec is NaN) OR (trap_qtran > 0.5)
    # flag       = low_aRs OR long_dur
    _trap_qtran  = col("trap_qtran").replace(0, np.nan)
    _q_over_trap = col("q") / _trap_qtran

    _low_aRs  = (col("transit_aRs") < 1.5) | (col("aRs") < 2)
    _long_dur = (
        (_q_over_trap < 0.6)
        | col("sig_sec").isna()
        | (_trap_qtran > 0.5)
    )
    df["LEOVetter_flag_unphysical_duration"] = _low_aRs | _long_dur

    # ── FA test 11: asymmetric ────────────────────────────────────────────────
    # diff = |trap_qtran_left - trap_qtran_right|
    # err  = sqrt(trap_qtran_err_left^2 + trap_qtran_err_right^2)
    # flag = diff / err > thresholds["ASYM"]
    _qL   = col("trap_qtran_left")
    _qR   = col("trap_qtran_right")
    _eL   = col("trap_qtran_err_left")
    _eR   = col("trap_qtran_err_right")
    _diff = (_qL - _qR).abs()
    _err  = np.sqrt(_eL**2 + _eR**2).replace(0, np.nan)
    df["LEOVetter_flag_asymmetric"] = (_diff / _err) > t["ASYM"]

    # ── FA test 12: chi ───────────────────────────────────────────────────────
    # flag = CHI < thresholds["CHI"]
    # Low CHI = per-transit SNRs are mutually inconsistent.
    # chi_threshold override is provided for M-dwarf samples where stellar
    # activity inflates scatter; validate against confirmed planets before use.
    chi_thr = chi_threshold if chi_threshold is not None else t["CHI"]
    df["LEOVetter_flag_chi"] = col("CHI") < chi_thr
    df["LEOVetter_chi_threshold_used"] = chi_thr

    # ── FA test 13: data_gapped ───────────────────────────────────────────────
    # flag = N_gap_2.0 / N_transit >= thresholds["frac_gap"]
    _frac_gap = col("N_gap_2.0") / col("N_transit").replace(0, np.nan)
    df["LEOVetter_flag_data_gapped"] = _frac_gap >= t["frac_gap"]

    # ── FP test 14: odd_even ─────────────────────────────────────────────────
    # sig_oe = |odd_dep - even_dep| / sqrt(odd_dep_err^2 + even_dep_err^2)
    # flag   = sig_oe > 3.0
    # The sigma=3 threshold is hardcoded in leo_vetter/thresholds.py (not a
    # named key in _default_thresholds); keep as literal here.
    _odd_d  = col("odd_dep")
    _eve_d  = col("even_dep")
    _odd_e  = col("odd_dep_err")
    _eve_e  = col("even_dep_err")
    _oe_denom = np.sqrt(_odd_e**2 + _eve_e**2).replace(0, np.nan)
    _sig_oe   = (_odd_d - _eve_d).abs() / _oe_denom
    df["LEOVetter_flag_odd_even"] = _sig_oe > 3.0

    # ── FP test 15: vshaped ───────────────────────────────────────────────────
    # flag = (transit_b + transit_RpRs) > thresholds["V_shape"]
    df["LEOVetter_flag_vshaped"] = (
        (col("transit_b") + col("transit_RpRs")) > t["V_shape"]
    )

    # ── FP test 16: large ─────────────────────────────────────────────────────
    # flag = Rp > thresholds["size"]
    df["LEOVetter_flag_large"] = col("Rp") > t["size"]

    # ── FP test 17: secondary ─────────────────────────────────────────────────
    # Mirrors secondary() in NEMESIS_pipeline_new_temp_4 exactly.
    # MS4 = (sig_sec / Fred) - FA1   [note: FA1, not FA2]
    # MS5 = (sig_sec - sig_ter) - FA2
    # MS6 = (sig_sec - sig_pos) - FA2
    # albedo_ok = (albedo < 1.0) AND (dep_sec < 0.1 * dep)
    # flag = (MS4 > MS4_thr) AND (MS5 > MS5_thr OR MS6 > MS6_thr) AND NOT albedo_ok
    _ms4 = (col("sig_sec") / col("Fred").replace(0, np.nan)) - col("FA1")
    _ms5 = col("sig_sec") - col("sig_ter") - col("FA2")
    _ms6 = col("sig_sec") - col("sig_pos") - col("FA2")
    _albedo_ok = (col("albedo") < 1.0) & (col("dep_sec") < 0.1 * col("dep"))
    df["LEOVetter_flag_secondary"] = (
        (_ms4 > t["MS4"])
        & ((_ms5 > t["MS5"]) | (_ms6 > t["MS6"]))
        & ~_albedo_ok
    )

    # ── FP test 18: offset ────────────────────────────────────────────────────
    # offset_qual is not stored by compute_flux_metrics in the parquet schema.
    # Hardcoded False; consistent with check_ALL_thresholds skipping the test
    # when 'offset_qual' is absent from the metrics dict.
    df["LEOVetter_flag_offset"] = False

    # ── Aggregates ────────────────────────────────────────────────────────────
    fa_cols = [f"LEOVetter_flag_{t_}" for t_ in FA_TESTS if f"LEOVetter_flag_{t_}" in df.columns]
    fp_cols = [f"LEOVetter_flag_{t_}" for t_ in FP_TESTS if f"LEOVetter_flag_{t_}" in df.columns]
    df["LEOVetter_flag_any_FA"] = df[fa_cols].any(axis=1)
    df["LEOVetter_flag_any_FP"] = df[fp_cols].any(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Classification
# ─────────────────────────────────────────────────────────────────────────────

def derive_classification(
    cat: pd.DataFrame,
    fa_col: str = "LEOVetter_flag_any_FA",
    fp_col: str = "LEOVetter_flag_any_FP",
) -> pd.DataFrame:
    """
    Add "classification" column: PC / FA / FP. FA takes precedence over FP.

    Falls back to scanning LEOVetter_is_false_alarm / LEOVetter_is_false_positive
    aggregate columns if the flag_ columns are absent.

    Parameters
    ----------
    cat : pd.DataFrame
    fa_col, fp_col : str
        Column names for aggregate FA and FP booleans.

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> cat = derive_classification(cat)
    >>> cat["classification"].value_counts()
    """
    df = cat.copy()

    if fa_col in df.columns and fp_col in df.columns:
        is_fa = df[fa_col].fillna(False).astype(bool)
        is_fp = df[fp_col].fillna(False).astype(bool)
    elif "LEOVetter_is_false_alarm" in df.columns:
        is_fa = df["LEOVetter_is_false_alarm"].fillna(False).astype(bool)
        is_fp = df["LEOVetter_is_false_positive"].fillna(False).astype(bool)
    else:
        is_fa = pd.Series(False, index=df.index)
        is_fp = pd.Series(False, index=df.index)

    df["classification"] = "PC"
    df.loc[is_fp, "classification"] = "FP"
    df.loc[is_fa, "classification"] = "FA"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4a. Population summary figure  (primary output for papers/proposals)
# ─────────────────────────────────────────────────────────────────────────────

def plot_population_summary(
    cat: pd.DataFrame,
    tic_id_col: str = "ID",
    pipeline_col: str = "pipeline",
    class_col: str = "classification",
    figsize: tuple[float, float] = (14, 10),
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel population-level summary. Designed for 400+ targets.

    Panels
    ------
    (A) Stacked horizontal bar -- one bar per pipeline, stacked by
        classification count. Immediately shows which pipeline is most
        conservative / liberal.
    (B) Grouped bar -- per-pipeline classification counts split by
        FA vs FP (only targets that failed; PC omitted). Shows the FA/FP
        composition of failures.
    (C) Pipeline agreement bar -- for each pair of pipelines that both
        ran on a target, fraction of (target) pairs where they agree on
        classification. Quantifies cross-pipeline consistency.

    Parameters
    ----------
    cat : pd.DataFrame
        Compiled catalog with "classification" column.
    tic_id_col : str
        TIC identifier column. Default "ID".
    pipeline_col : str
        Pipeline label column. Default "pipeline".
    class_col : str
        Classification column. Default "classification".
    figsize : tuple
        Figure dimensions. Default (14, 10).
    savepath : str or None
        Output PNG path.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_population_summary(cat, savepath="pop_summary.png")
    """
    df = cat.copy()
    pipes = [p for p in PIPELINE_ORDER if p in df[pipeline_col].unique()]
    pipes += [p for p in df[pipeline_col].unique() if p not in pipes]

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"LEO-Vetter Batch Summary  "
        f"({df[tic_id_col].nunique()} targets, {df[pipeline_col].nunique()} pipelines)",
        fontsize=13, y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4,
                           left=0.08, right=0.97, top=0.93, bottom=0.08)
    ax_stacked = fig.add_subplot(gs[0, :])   # full width top
    ax_fa_fp   = fig.add_subplot(gs[1, 0])
    ax_agree   = fig.add_subplot(gs[1, 1])

    # ── A: stacked horizontal bar per pipeline ────────────────────────────────
    counts = (
        df.groupby([pipeline_col, class_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=_CLASS_ORDER, fill_value=0)
    )
    counts = counts.reindex(pipes, fill_value=0)
    n_total = counts.sum(axis=1)

    lefts = np.zeros(len(pipes))
    for cls in _CLASS_ORDER:
        vals = counts[cls].values if cls in counts.columns else np.zeros(len(pipes))
        bars = ax_stacked.barh(
            pipes, vals, left=lefts,
            color=_CLASS_COLORS[cls], edgecolor="white", linewidth=0.5,
            label=cls,
        )
        # Annotate counts inside bar if wide enough
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 0:
                frac = v / max(n_total.iloc[i], 1)
                if frac > 0.05:
                    ax_stacked.text(
                        l + v / 2, i, str(v),
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold",
                    )
        lefts += vals

    # Percentage labels at right end
    for i, (pipe, tot) in enumerate(zip(pipes, n_total)):
        pc_n = counts.loc[pipe, "PC"] if "PC" in counts.columns else 0
        ax_stacked.text(
            tot + tot * 0.005, i,
            f"{pc_n/max(tot,1)*100:.0f}% PC",
            va="center", fontsize=8, color="gray",
        )

    ax_stacked.set_xlabel("Number of targets")
    ax_stacked.set_title("(A) Classification count per pipeline", fontsize=10)
    ax_stacked.legend(
        handles=[mpatches.Patch(color=_CLASS_COLORS[c], label=c) for c in _CLASS_ORDER],
        loc="lower right", fontsize=9, frameon=False,
    )
    ax_stacked.set_xlim(0, n_total.max() * 1.12)
    ax_stacked.invert_yaxis()

    # ── B: FA vs FP breakdown among failures only ────────────────────────────
    fail = df[df[class_col] != "PC"]
    fa_fp_counts = (
        fail.groupby([pipeline_col, class_col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["FA", "FP"], fill_value=0)
        .reindex(pipes, fill_value=0)
    )
    x = np.arange(len(pipes))
    w = 0.35
    ax_fa_fp.bar(x - w/2, fa_fp_counts["FA"], width=w,
                 color=_CLASS_COLORS["FA"], label="FA", edgecolor="white")
    ax_fa_fp.bar(x + w/2, fa_fp_counts["FP"], width=w,
                 color=_CLASS_COLORS["FP"], label="FP", edgecolor="white")
    ax_fa_fp.set_xticks(x)
    ax_fa_fp.set_xticklabels(pipes, rotation=30, ha="right", fontsize=8)
    ax_fa_fp.set_ylabel("Count")
    ax_fa_fp.set_title("(B) FA vs FP failure composition", fontsize=10)
    ax_fa_fp.legend(fontsize=8, frameon=False)
    if fa_fp_counts.values.sum() == 0:
        ax_fa_fp.text(0.5, 0.5, "No failures in this batch",
                      ha="center", va="center", transform=ax_fa_fp.transAxes,
                      fontsize=10, color="gray")

    # ── C: pairwise pipeline agreement ───────────────────────────────────────
    from itertools import combinations
    pair_labels, pair_agree = [], []
    pivot = df.pivot_table(index=tic_id_col, columns=pipeline_col,
                           values=class_col, aggfunc="first")
    for p1, p2 in combinations(pipes, 2):
        if p1 not in pivot.columns or p2 not in pivot.columns:
            continue
        both = pivot[[p1, p2]].dropna()
        if len(both) == 0:
            continue
        agree_frac = (both[p1] == both[p2]).mean()
        
        #pair_labels.append(f"{p1[:4]}\nvs\n{p2[:4]}")
        if p1=='GSFC-ELEANOR-LITE':
            p1='ELEANOR'
        if p2=='GSFC-ELEANOR-LITE':
            p2='ELEANOR'
            
        if p1=='TESS-SPOC':
            p1='SPOC'
        if p2=='TESS-SPOC':
            p2='SPOC'            
        pair_labels.append(f"{p1}\nvs\n{p2}")
        
        pair_agree.append(agree_frac)

    if pair_agree:
        colors_agree = [
            "#43A047" if v >= 0.9 else "#FB8C00" if v >= 0.7 else "#E53935"
            for v in pair_agree
        ]
        ax_agree.bar(range(len(pair_labels)), pair_agree,
                     color=colors_agree, edgecolor="white")
        ax_agree.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax_agree.set_xticks(range(len(pair_labels)))
        ax_agree.set_xticklabels(pair_labels, fontsize=7)
        ax_agree.set_ylim(0, 1.05)
        ax_agree.set_ylabel("Fraction of targets in agreement")
        ax_agree.set_title("(C) Pairwise pipeline agreement", fontsize=10)
        for i, v in enumerate(pair_agree):
            ax_agree.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    else:
        ax_agree.text(0.5, 0.5, "Insufficient overlap for pairwise comparison",
                      ha="center", va="center", transform=ax_agree.transAxes,
                      fontsize=9, color="gray")

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Per-test failure rate heatmap (diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def plot_flag_rate_heatmap(
    cat: pd.DataFrame,
    pipeline_col: str = "pipeline",
    figsize: tuple[float, float] = (14, 5),
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of per-test failure rates: rows = pipelines, cols = LEO-Vetter tests.

    Cell value = fraction of targets where that test fired for that pipeline.
    Separates FA tests (blue colormap) from FP tests (red colormap) with a
    vertical divider.

    This is the primary diagnostic for understanding WHICH tests drive
    classification differences across pipelines.

    Parameters
    ----------
    cat : pd.DataFrame
        Catalog with LEOVetter_flag_* columns (output of add_per_test_flags).
    pipeline_col : str
        Pipeline label column. Default "pipeline".
    figsize : tuple
        Figure dimensions. Default (14, 5).
    savepath : str or None
        Output PNG path.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_flag_rate_heatmap(cat, savepath="flag_rates.png")
    """
    df = cat.copy()
    pipes = [p for p in PIPELINE_ORDER if p in df[pipeline_col].unique()]
    pipes += [p for p in df[pipeline_col].unique() if p not in pipes]

    flag_cols = [f"LEOVetter_flag_{t_}" for t_ in ALL_TESTS]
    flag_cols = [c for c in flag_cols if c in df.columns]
    test_names = [c.replace("LEOVetter_flag_", "") for c in flag_cols]

    if not flag_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5,
                "No LEOVetter_flag_* columns found.\nRun add_per_test_flags() first.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    # Build rate matrix: (n_pipelines, n_tests)
    rate_matrix = np.zeros((len(pipes), len(flag_cols)))
    for i, pipe in enumerate(pipes):
        sub = df[df[pipeline_col] == pipe]
        for j, fcol in enumerate(flag_cols):
            if fcol in sub.columns and len(sub) > 0:
                rate_matrix[i, j] = sub[fcol].fillna(False).astype(float).mean()

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle("LEO-Vetter Per-Test Failure Rate  (fraction of targets flagged)",
                 fontsize=12)

    im = ax.imshow(rate_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(pipes)):
        for j in range(len(test_names)):
            v = rate_matrix[i, j]
            color = "white" if v > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Divider between FA and FP tests
    n_fa = sum(1 for t_ in FA_TESTS if f"LEOVetter_flag_{t_}" in flag_cols)
    if n_fa < len(flag_cols):
        ax.axvline(x=n_fa - 0.5, color="white", linewidth=2.5)
        ax.text(n_fa / 2 - 0.5, -0.8, "← False Alarm tests",
                ha="center", va="top", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform())
        ax.text(n_fa + (len(flag_cols) - n_fa) / 2 - 0.5, -0.8,
                "False Positive tests →",
                ha="center", va="top", fontsize=8, color="gray",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pipes)))
    ax.set_yticklabels(pipes, fontsize=9)

    plt.colorbar(im, ax=ax, label="Failure rate", shrink=0.7)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Pipeline disagreement figure (which targets are contested)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pipeline_agreement(
    cat: pd.DataFrame,
    tic_id_col: str = "ID",
    pipeline_col: str = "pipeline",
    class_col: str = "classification",
    top_n_contested: int = 30,
    figsize: tuple[float, float] = (14, 8),
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel figure showing pipeline classification disagreement.

    Panel (A): Distribution of "n_pipelines_agree" -- how many pipelines
    agree on the plurality classification for each target. Spike at max
    = high consistency; flat distribution = systematic disagreements.

    Panel (B): Heatmap of the top_n_contested targets (those with lowest
    n_pipelines_agree). Rows = targets, cols = pipelines. Each cell is
    colored by classification. Useful for identifying individual targets
    that warrant manual follow-up.

    Parameters
    ----------
    cat : pd.DataFrame
    tic_id_col, pipeline_col, class_col : str
    top_n_contested : int
        Number of most contested targets to show in panel B. Default 30.
    figsize : tuple
        Default (14, 8).
    savepath : str or None

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_pipeline_agreement(cat, top_n_contested=30)
    """
    df = cat.copy()
    pipes = [p for p in PIPELINE_ORDER if p in df[pipeline_col].unique()]
    pipes += [p for p in df[pipeline_col].unique() if p not in pipes]

    pivot = df.pivot_table(index=tic_id_col, columns=pipeline_col,
                           values=class_col, aggfunc="first")
    pivot = pivot.reindex(columns=pipes)

    def _plurality_agree(row: pd.Series) -> int:
        vals = row.dropna()
        if vals.empty:
            return 0
        return int(vals.value_counts().iloc[0])

    n_agree = pivot.apply(_plurality_agree, axis=1)
    n_ran   = pivot.notna().sum(axis=1)

    fig, (ax_hist, ax_heat) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 2]},
    )
    fig.suptitle("Pipeline Classification Agreement", fontsize=13)

    # ── A: histogram of agreement counts ─────────────────────────────────────
    max_pipes = n_ran.max()
    bins = np.arange(0.5, max_pipes + 1.5, 1)
    ax_hist.hist(n_agree.values, bins=bins, color="#455A64", edgecolor="white")
    ax_hist.set_xlabel("# pipelines agreeing on plurality class", fontsize=9)
    ax_hist.set_ylabel("# targets", fontsize=9)
    ax_hist.set_title("(A) Agreement distribution", fontsize=10)
    ax_hist.set_xticks(range(1, int(max_pipes) + 1))
    # Annotate unanimous fraction
    unanimous = (n_agree == n_ran).sum()
    ax_hist.text(0.97, 0.97,
                 f"Unanimous: {unanimous}/{len(n_agree)} "
                 f"({unanimous/len(n_agree)*100:.0f}%)",
                 ha="right", va="top", transform=ax_hist.transAxes, fontsize=9)

    # ── B: heatmap of most contested targets ─────────────────────────────────
    class_to_int = {"PC": 0, "FA": 1, "FP": 2}
    cmap = ListedColormap([_CLASS_COLORS[c] for c in _CLASS_ORDER])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    contested_ids = n_agree.nsmallest(top_n_contested).index.tolist()
    if not contested_ids:
        ax_heat.text(0.5, 0.5, "All targets unanimous",
                     ha="center", va="center", transform=ax_heat.transAxes)
    else:
        sub_pivot = pivot.loc[contested_ids]
        grid = sub_pivot.replace(class_to_int).astype(float).values

        ax_heat.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        ax_heat.set_xticks(range(len(pipes)))
        ax_heat.set_xticklabels(pipes, rotation=30, ha="right", fontsize=8)
        ax_heat.set_yticks(range(len(contested_ids)))
        ax_heat.set_yticklabels(
            [f"TIC {t}" for t in contested_ids], fontsize=7
        )
        # Annotate
        for i in range(len(contested_ids)):
            for j in range(len(pipes)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax_heat.text(j, i, _CLASS_ORDER[int(v)],
                                 ha="center", va="center", fontsize=6,
                                 color="white", fontweight="bold")
                else:
                    ax_heat.text(j, i, "N/A", ha="center", va="center",
                                 fontsize=5, color="white")

        ax_heat.set_title(
            f"(B) Most contested {len(contested_ids)} targets", fontsize=10
        )
        legend_patches = [
            mpatches.Patch(color=_CLASS_COLORS[c], label=c)
            for c in _CLASS_ORDER
        ]
        ax_heat.legend(handles=legend_patches, loc="lower right",
                       fontsize=8, frameon=True,
                       bbox_to_anchor=(1.0, -0.05))

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Agreement table utility
# ─────────────────────────────────────────────────────────────────────────────

def classification_agreement_table(
    cat: pd.DataFrame,
    tic_id_col: str = "ID",
    pipeline_col: str = "pipeline",
    class_col: str = "classification",
) -> pd.DataFrame:
    """
    Pivot to (targets x pipelines) classification table with consensus column.

    Parameters
    ----------
    cat, tic_id_col, pipeline_col, class_col : standard

    Returns
    -------
    pd.DataFrame
        Columns: pipeline names + ["consensus", "n_agree", "n_ran"].

    Example
    -------
    >>> tbl = classification_agreement_table(cat)
    >>> tbl[tbl["consensus"].str.endswith("*")]  # contested targets
    """
    pivot = cat.pivot_table(index=tic_id_col, columns=pipeline_col,
                            values=class_col, aggfunc="first")
    pipes = [p for p in PIPELINE_ORDER if p in pivot.columns]
    pipes += [c for c in pivot.columns if c not in pipes]
    pivot = pivot.reindex(columns=pipes)

    def _consensus(row: pd.Series) -> str:
        vals = row.dropna()
        if vals.empty:
            return "N/A"
        vc = vals.value_counts()
        suffix = "" if vc.iloc[0] == len(vals) else "*"
        return vc.index[0] + suffix

    pivot["consensus"] = pivot.apply(_consensus, axis=1)
    pivot["n_agree"]   = pivot[pipes].apply(
        lambda row: (row.dropna() == pivot.loc[row.name, "consensus"].rstrip("*")).sum(),
        axis=1,
    )
    pivot["n_ran"] = pivot[pipes].notna().sum(axis=1)
    return pivot






"""
plot_leovetter_metric_distributions.py

Stacked histogram grid of all LEO-Vetter numeric metrics, with one
histogram per metric panel, stacked by pipeline. A vertical dashed line
marks the threshold where one exists. Panels are grouped FA tests first,
then FP tests, then derived/contextual quantities.

This is the primary tool for:
  - Threshold calibration (especially CHI for M-dwarfs)
  - Identifying which metrics drive cross-pipeline differences
  - Spotting pathological distributions before the full 400-target run
"""

# from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Import real thresholds directly -- do NOT redefine or override this.
from leo_vetter.thresholds import _default_thresholds as _LV_THRESHOLDS


# ─────────────────────────────────────────────────────────────────────────────
# Metric catalogue
# Each entry: (col_suffix, x_label, threshold_key_or_None, flag_direction, log_x)
#
# threshold_key: a key into _LV_THRESHOLDS, or a bare float for thresholds that
# are not named keys (e.g. the odd_even sigma=3 implicit threshold), or None.
#
# flag_direction: ">" flag when metric > threshold (shade right)
#                 "<" flag when metric < threshold (shade left)
#                 None no threshold line
#
# Column suffixes match actual stored parquet names (without "LEOVetter_" prefix),
# audited against TCELightCurve.compute_flux_metrics output.
#
# Corrections vs. prior version
# ------------------------------
# DMM threshold:    0.3  -> "DMM"  (real value 1.5 from _default_thresholds)
# SHP threshold:    0.2  -> "SHP"  (real value 0.6)
# MES threshold:    7.1  -> "MES"  (real value 6.2)
# CHI threshold:    7.0  -> "CHI"  (real value 7.8; overrideable per M-dwarf sample)
# mean_chases thr:  0.5  -> "chases" (real value 0.78; direction corrected to "<")
# med_chases:       removed (not stored by compute_flux_metrics)
# sine_sig thr:     added "SWEET" key (real value 15); panel split into
#                   sine_sig (> SWEET) and per (< 10 gate, no named threshold)
# N_gap panels:     corrected to N_gap_2.0 only (used by data_gapped flag);
#                   N_gap_0.5 and N_gap_1.0 kept as informational (no threshold)
# asymmetry metric: sig_r removed (not the correct stored column for asymmetric
#                   test); correct columns are trap_qtran_left/right + errors,
#                   but ratio is derived -- kept as informational panel
# unphysical_dur:   transit_dur panel kept as informational; threshold is a
#                   compound geometric predicate, not a simple scalar cutoff,
#                   so no threshold line is drawn
# Rp threshold:     20.0 -> "size" (real value 22)
# V_shape threshold:1.3  -> "V_shape" (real value 1.5)
# AIC panels:       added AIC1/AIC2 reference lines for bad_fit diagnostics
# ─────────────────────────────────────────────────────────────────────────────

_FA_METRICS = [
    # (col_suffix,         xlabel,               threshold_key,  direction, log_x)
    ("MES",                "MES",                "MES",          "<",       False),
    ("new_MES",            "new MES",            "MES",          "<",       False),
    ("new_N_transit",      "new N transit",      "N_transit",    "<",       False),
    ("SHP",                "SHP",                "SHP",          ">",       False),
    ("sig_pri",            "sig_pri",            None,           None,      False),
    ("sig_ter",            "sig_ter",            None,           None,      False),
    ("sig_pos",            "sig_pos",            None,           None,      False),
    ("Fred",               "Fred",               None,           None,      False),
    ("DMM",                "DMM",                "DMM",          ">",       False),
    ("max_SES",            "max SES",            None,           None,      False),
    # chases: flag fires when mean_chases < threshold AND N_transit <= 5.
    # Direction is "<": low mean_chases means signal is NOT unique locally.
    ("mean_chases",        "mean chases",        "chases",       "<",       False),
    ("transit_aic",        "transit AIC",        "AIC1",         ">",       False),
    ("line_aic",           "line AIC",           None,           None,      False),
    ("transit_chisqr",     "transit chi2",       None,           None,      True),
    ("line_chisqr",        "line chi2",          None,           None,      True),
    # sinusoidal: flag = sine_sig > SWEET AND per < 10.
    # Both columns shown; threshold drawn on sine_sig panel only.
    ("sine_sig",           "sine_sig",           "SWEET",        ">",       False),
    ("per",                "period [d]",         None,           None,      False),
    # transit_dur: unphysical_duration uses a compound geometric predicate
    # (transit_aRs, aRs, q, trap_qtran, sig_sec). No single scalar threshold.
    ("transit_dur",        "transit dur [d]",    None,           None,      False),
    ("trap_qtran",         "trap_qtran",         None,           None,      False),
    ("transit_aRs",        "transit a/Rs",       None,           None,      False),
    ("aRs",                "a/Rs (catalog)",     None,           None,      False),
    # asymmetric: flag derived from |trap_qtran_L - trap_qtran_R| / sqrt(eL^2+eR^2) > ASYM.
    # Individual qtran columns shown for inspection; no threshold on raw columns.
    ("trap_qtran_left",    "trap_qtran_L",       None,           None,      False),
    ("trap_qtran_right",   "trap_qtran_R",       None,           None,      False),
    # CHI: flag when CHI < threshold (low CHI = inconsistent per-transit SNRs).
    # M-dwarf note: canonical threshold 7.8 will fire broadly on active stars;
    # use chi_threshold parameter to override for your sample.
    ("CHI",                "CHI",                "CHI",          "<",       True),
    # data_gapped: flag when N_gap_2.0 / N_transit >= frac_gap.
    # Show N_gap_2.0 directly; the ratio panel is more diagnostic but requires
    # N_transit to compute -- done in the derived section.
    ("N_gap_2.0",          "N gap (2.0d)",       None,           None,      False),
    ("N_gap_1.0",          "N gap (1.0d)",       None,           None,      False),
    ("N_gap_0.5",          "N gap (0.5d)",       None,           None,      False),
]

_FP_METRICS = [
    ("odd_dep",            "odd depth",          None,           None,      False),
    ("even_dep",           "even depth",         None,           None,      False),
    ("odd_dep_err",        "odd depth err",      None,           None,      False),
    ("even_dep_err",       "even depth err",     None,           None,      False),
    ("transit_b",          "transit b",          None,           None,      False),
    ("transit_RpRs",       "Rp/Rs",              None,           None,      False),
    # large: flag when Rp > thresholds["size"] (= 22 Re)
    ("Rp",                 "Rp [Re]",            "size",         ">",       False),
    ("albedo",             "albedo",             None,           None,      True),
    ("dep",                "primary depth",      None,           None,      False),
    ("dep_sec",            "secondary depth",    None,           None,      False),
    ("sig_sec",            "sig_sec",            None,           None,      False),
    # vshaped: flag when transit_b + transit_RpRs > V_shape (= 1.5)
    # Shown as a combined derived panel in _DERIVED_METRICS below.
]

_DERIVED_METRICS = [
    # N_transit: informational; flag gate is N_transit <= 5 (chases) or <= 10 (single_event)
    ("N_transit",          "N transit",          "N_transit",    "<",       False),
    ("qtran",              "qtran",              None,           None,      False),
    ("FA1",                "FA1",                None,           None,      False),
    ("FA2",                "FA2",                None,           None,      False),
    ("q",                  "q (dur/period)",     None,           None,      False),
    ("Teq",                "Teq [K]",            None,           None,      False),
    ("Seff",               "Seff",               None,           None,      False),
]

PIPELINE_ORDER = ["NEMESIS", "QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]
PIPE_COLORS    = ["#000000", "#E07B39", "#1F77B4", "#9467BD", "#2CA02C"]


def plot_leovetter_metric_distributions(
    cat: pd.DataFrame,
    pipeline_col: str = "pipeline",
    prefix: str = "LEOVetter_",
    n_bins: int = 30,
    chi_threshold: Optional[float] = None,
    figsize_per_panel: tuple = (3.0, 2.2),
    n_cols: int = 6,
    show_fa_metrics: bool = True,
    show_fp_metrics: bool = True,
    show_derived_metrics: bool = False,
    savepath: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked histogram grid of all LEO-Vetter numeric metrics, one panel per
    metric, stacked by pipeline color. Threshold lines are overlaid where
    applicable.

    Thresholds are read live from leo_vetter.thresholds._default_thresholds.
    No threshold values are hardcoded in this function.

    Layout: FA-test metrics first (top block), FP-test metrics second (middle
    block), optional derived/contextual metrics last (bottom block).

    Parameters
    ----------
    cat : pd.DataFrame
        Compiled catalog from compile_batch_leovetter_catalog. Must contain
        LEOVetter_* prefixed metric columns.
    pipeline_col : str
        Pipeline label column. Default "pipeline".
    prefix : str
        LEOVetter column prefix. Default "LEOVetter_".
    n_bins : int
        Number of histogram bins per panel. Default 30.
    chi_threshold : float or None
        Override for the CHI threshold line only. None uses
        _default_thresholds["CHI"] (= 7.8). For active M-dwarf samples,
        set this after validating against confirmed planets in your sample.
    figsize_per_panel : tuple
        Width x height of each panel in inches. Default (3.0, 2.2).
    n_cols : int
        Number of columns in the grid. Default 6.
    show_fa_metrics : bool
        Include FA-test metric panels. Default True.
    show_fp_metrics : bool
        Include FP-test metric panels. Default True.
    show_derived_metrics : bool
        Include derived/contextual metric panels. Default False.
    savepath : str or None
        Save path for PNG output. Default None.

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> from leo_vetter.thresholds import _default_thresholds
    >>> cat = compile_batch_leovetter_catalog(RESULTS_ROOT)
    >>> fig = plot_leovetter_metric_distributions(
    ...     cat, chi_threshold=7.8,
    ...     savepath="metric_distributions.png")
    """
    df = cat.copy()
    t  = _LV_THRESHOLDS  # live reference; never override

    # Build pipeline list in canonical order
    pipes_present  = [p for p in PIPELINE_ORDER if p in df[pipeline_col].unique()]
    pipes_present += [p for p in df[pipeline_col].unique() if p not in pipes_present]
    pipe_color_map = {p: PIPE_COLORS[i % len(PIPE_COLORS)]
                      for i, p in enumerate(pipes_present)}

    # ── Resolve threshold values from _LV_THRESHOLDS ─────────────────────────
    # Each metric entry carries a threshold_key (string key into _LV_THRESHOLDS,
    # bare float, or None). Resolve here so panels use the live dict values.
    def _resolve_thr(key, col_suf):
        """Return (threshold_float_or_None) from a key or bare float."""
        if key is None:
            return None
        if isinstance(key, (int, float)):
            return float(key)
        # CHI may be overridden for M-dwarf samples
        if key == "CHI" and chi_threshold is not None:
            return float(chi_threshold)
        return t.get(key, None)

    # Assemble panel list
    panels = []
    if show_fa_metrics:
        panels += [("FA", *m) for m in _FA_METRICS]
    if show_fp_metrics:
        panels += [("FP", *m) for m in _FP_METRICS]
    if show_derived_metrics:
        panels += [("derived", *m) for m in _DERIVED_METRICS]

    # Resolve threshold values and drop panels where the column is absent
    resolved_panels = []
    for grp, col_suf, xlabel, thr_key, direction, log_x in panels:
        full_col = prefix + col_suf
        if full_col not in df.columns:
            continue  # column not stored for this batch; skip silently
        thr_val = _resolve_thr(thr_key, col_suf)
        resolved_panels.append((grp, col_suf, xlabel, thr_val, direction, log_x))

    n_panels = len(resolved_panels)
    if n_panels == 0:
        raise ValueError(
            "No LEOVetter_* metric columns found in catalog. "
            "Check that prefix matches stored column names."
        )

    n_rows = int(np.ceil(n_panels / n_cols))
    fw     = figsize_per_panel[0] * n_cols
    fh     = figsize_per_panel[1] * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fw, fh),
        constrained_layout=True,
    )
    axes_flat = np.array(axes).flatten()

    _grp_colors = {"FA": "#FFF3E0", "FP": "#FFEBEE", "derived": "#E8F5E9"}

    for idx, (grp, col_suf, xlabel, thr_val, direction, log_x) in enumerate(resolved_panels):
        ax       = axes_flat[idx]
        full_col = prefix + col_suf

        # Per-pipeline data arrays
        pipe_data, pipe_labels, pipe_colors_list = [], [], []
        for pipe in pipes_present:
            vals = df.loc[df[pipeline_col] == pipe, full_col].dropna().to_numpy(dtype=float)
            if vals.size > 0:
                pipe_data.append(vals)
                pipe_labels.append(pipe)
                pipe_colors_list.append(pipe_color_map[pipe])

        if not pipe_data:
            ax.set_visible(False)
            continue

        all_vals = np.concatenate(pipe_data)
        finite   = all_vals[np.isfinite(all_vals)]
        if finite.size == 0:
            ax.set_visible(False)
            continue

        # Common bin edges
        if log_x and np.all(finite > 0):
            lo = finite.min() * 0.9
            hi = finite.max() * 1.1
            if lo <= 0:
                lo = 1e-6
            edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
            ax.set_xscale("log")
        else:
            lo  = np.percentile(finite, 1)
            hi  = np.percentile(finite, 99)
            pad = (hi - lo) * 0.05 or 0.1
            edges = np.linspace(lo - pad, hi + pad, n_bins + 1)

        # Stacked histogram
        bottom = np.zeros(n_bins)
        for pvals, pcolor, plabel in zip(pipe_data, pipe_colors_list, pipe_labels):
            pfinite        = pvals[np.isfinite(pvals)]
            counts, _      = np.histogram(pfinite, bins=edges)
            ax.bar(
                edges[:-1], counts, width=np.diff(edges),
                bottom=bottom, color=pcolor, alpha=0.85,
                align="edge", label=plabel, linewidth=0,
            )
            bottom += counts

        # Threshold line and shading
        if thr_val is not None and np.isfinite(thr_val):
            ax.axvline(thr_val, color="black", linestyle="--",
                       linewidth=1.2, zorder=5, alpha=0.9)
            if direction == ">":
                ax.axvspan(thr_val, edges[-1], alpha=0.08, color="red", zorder=0)
            elif direction == "<":
                ax.axvspan(edges[0], thr_val, alpha=0.08, color="red", zorder=0)
            # Annotate threshold value so it is legible without hovering
            ax.text(
                thr_val, bottom.max() * 0.97,
                f" {thr_val:.2g}", fontsize=6, color="black",
                va="top", ha="left", zorder=6,
            )

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("count", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
        ax.set_facecolor(_grp_colors.get(grp, "white"))
        ax.text(
            0.98, 0.97, grp, transform=ax.transAxes,
            fontsize=6, ha="right", va="top", color="gray", style="italic",
        )

    # Hide unused axes
    for idx in range(len(resolved_panels), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=pipe_color_map[p], alpha=0.85)
        for p in pipes_present
    ]
    fig.legend(
        legend_handles, pipes_present,
        loc="lower center", ncol=len(pipes_present),
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
    )

    n_targets = df["ID"].nunique() if "ID" in df.columns else len(df)
    n_pipes   = len(pipes_present)
    chi_note  = (f"  [CHI threshold overridden: {chi_threshold:.1f}]"
                 if chi_threshold is not None else "")
    fig.suptitle(
        f"LEO-Vetter Metric Distributions  "
        f"({n_pipes} pipelines, {n_targets} targets){chi_note}",
        fontsize=12, y=1.01,
    )

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")

    return fig


def derive_detrending_agnostic_classification(
    cat: pd.DataFrame,
    *,
    excluded_tests: list[str] = ("chi", "sinusoidal", "non_unique"),
    fa_col: str = "LEOVetter_flag_any_FA",
    fp_col: str = "LEOVetter_flag_any_FP",
    class_col: str = "classification",
    agnostic_fa_col: str = "LEOVetter_flag_any_FA_agnostic",
    agnostic_fp_col: str = "LEOVetter_flag_any_FP_agnostic",
    agnostic_class_col: str = "classification_agnostic",
    contested_col: str = "classification_contested",
) -> pd.DataFrame:
    """
    Derive a secondary "detrending-agnostic" classification that excludes
    LEO-Vetter tests whose flag rates are known to differ systematically
    across pipelines due to detrending method and error-array differences
    rather than genuine signal quality.

    Background
    ----------
    Three FA tests are pipeline-dependent in this M-dwarf TOI sample:

    chi
        CHI measures per-transit depth scatter normalized by Detrended Error.
        TGLC and GSFC-ELEANOR-LITE lack native error columns; their rolling-MAD
        fallback errors are less precise than the propagated photon-noise errors
        used by NEMESIS, QLP, and TESS-SPOC. This makes CHI systematically lower
        for TGLC and GSFC-ELEANOR-LITE on active M-dwarfs, inflating their FA
        rate from this test regardless of whether the transit signal is genuine.

    sinusoidal
        sine_sig reflects residual stellar variability in the phased LC. NEMESIS
        uses GP-proxy PLD that specifically suppresses rotation-period residuals.
        CBV cotrending (QLP) and PCA cotrending (GSFC-ELEANOR-LITE) leave more
        variability residuals in the detrended flux, producing higher sine_sig
        values on active short-period rotators independent of transit quality.

    non_unique
        The MS1 sub-condition depends on Fred (red-noise factor), which is
        computed from error-weighted autocorrelation and inherits both the
        error-array and detrending-residual differences. Pipelines with worse
        correlated noise produce higher Fred and lower MS1, increasing flag rates
        for this test.

    Usage
    -----
    Run this function AFTER add_per_test_flags() and derive_classification().
    It adds three new columns:

    classification_agnostic
        PC / FA / FP using all tests except the excluded ones.

    classification_contested
        True if the full classification and agnostic classification disagree.
        These targets changed status solely because of pipeline-detrending
        differences and are highest priority for visual inspection.

    The full classification (using all tests) is preserved untouched as
    `classification`. Both columns are stored in the output catalog.

    Parameters
    ----------
    cat : pd.DataFrame
        Catalog from compile_batch_leovetter_catalog with LEOVetter_flag_*
        columns already added by add_per_test_flags() and a `classification`
        column from derive_classification().
    excluded_tests : list of str
        Test names to exclude from the agnostic classification. Defaults to
        ("chi", "sinusoidal", "non_unique"). Override to experiment with
        different exclusion sets.
    fa_col : str
        Full FA aggregate flag column name. Default "LEOVetter_flag_any_FA".
    fp_col : str
        Full FP aggregate flag column name. Default "LEOVetter_flag_any_FP".
    class_col : str
        Full classification column name. Default "classification".
    agnostic_fa_col : str
        Output column name for agnostic FA aggregate. Default
        "LEOVetter_flag_any_FA_agnostic".
    agnostic_fp_col : str
        Output column name for agnostic FP aggregate. Default
        "LEOVetter_flag_any_FP_agnostic".
    agnostic_class_col : str
        Output column name for agnostic classification. Default
        "classification_agnostic".
    contested_col : str
        Output column name for contested flag. Default
        "classification_contested".

    Returns
    -------
    pd.DataFrame
        Input catalog with agnostic_fa_col, agnostic_fp_col,
        agnostic_class_col, and contested_col appended.

    Example
    -------
    >>> from leo_vetter.thresholds import _default_thresholds
    >>> cat = compile_batch_leovetter_catalog(RESULTS_ROOT)
    >>> cat = add_per_test_flags(cat, thresholds=_default_thresholds)
    >>> cat = derive_classification(cat)
    >>> cat = derive_detrending_agnostic_classification(cat)
    >>>
    >>> # Fraction contested per pipeline
    >>> cat.groupby("pipeline")["classification_contested"].mean()
    >>>
    >>> # Targets that flip FA -> PC when detrending-sensitive tests are dropped
    >>> flipped = cat[
    ...     (cat["classification"] == "FA") &
    ...     (cat["classification_agnostic"] == "PC")
    ... ]
    >>> flipped[["ID", "pipeline", "classification", "classification_agnostic"]]
    """
    from leovetter_batch_analysis import FA_TESTS, FP_TESTS

    df = cat.copy()

    # ── Identify flag columns to include in agnostic aggregates ──────────────
    # Build the set of tests to exclude, normalizing to lowercase for safety.
    excluded = {t.lower() for t in excluded_tests}

    # FA tests retained in agnostic mode
    fa_agnostic = [
        t for t in FA_TESTS
        if t.lower() not in excluded
    ]
    # FP tests: none of the default excluded tests are FP tests, but allow
    # for user-specified exclusions that might be FP tests.
    fp_agnostic = [
        t for t in FP_TESTS
        if t.lower() not in excluded
    ]

    # Resolve to actual column names present in the catalog
    fa_flag_cols = [
        f"LEOVetter_flag_{t}" for t in fa_agnostic
        if f"LEOVetter_flag_{t}" in df.columns
    ]
    fp_flag_cols = [
        f"LEOVetter_flag_{t}" for t in fp_agnostic
        if f"LEOVetter_flag_{t}" in df.columns
    ]

    # Warn if any excluded test columns are absent (not an error -- they may
    # simply not have fired at all and were dropped by add_per_test_flags).
    for t in excluded_tests:
        col = f"LEOVetter_flag_{t}"
        if col not in df.columns:
            import warnings
            warnings.warn(
                f"derive_detrending_agnostic_classification: "
                f"excluded test '{t}' has no flag column '{col}' in catalog. "
                f"Exclusion has no effect for this test.",
                stacklevel=2,
            )

    # ── Compute agnostic aggregates ───────────────────────────────────────────
    if fa_flag_cols:
        df[agnostic_fa_col] = df[fa_flag_cols].any(axis=1)
    else:
        # All FA tests excluded -- nothing can flag FA
        df[agnostic_fa_col] = False

    if fp_flag_cols:
        df[agnostic_fp_col] = df[fp_flag_cols].any(axis=1)
    else:
        df[agnostic_fp_col] = False

    # ── Agnostic classification: FA takes precedence over FP ─────────────────
    # Mirrors the logic in derive_classification().
    conditions = [
        df[agnostic_fa_col],
        df[agnostic_fp_col],
    ]
    choices = ["FA", "FP"]
    df[agnostic_class_col] = np.select(conditions, choices, default="PC")

    # ── Contested flag: classification changed between full and agnostic ──────
    # A target is contested if dropping the detrending-sensitive tests changes
    # its outcome. These are the targets where pipeline-detrending differences
    # (not signal quality) are driving the classification.
    if class_col in df.columns:
        df[contested_col] = df[class_col] != df[agnostic_class_col]
    else:
        import warnings
        warnings.warn(
            f"derive_detrending_agnostic_classification: "
            f"full classification column '{class_col}' not found. "
            f"Run derive_classification() first. "
            f"'{contested_col}' will be all-False.",
            stacklevel=2,
        )
        df[contested_col] = False

    # ── Summary print ─────────────────────────────────────────────────────────
    n_total     = len(df)
    n_contested = int(df[contested_col].sum())
    print(
        f"[agnostic classification] excluded tests: {list(excluded_tests)}\n"
        f"  FA tests retained: {fa_agnostic}\n"
        f"  FP tests retained: {fp_agnostic}\n"
        f"  Total rows: {n_total}  |  Contested (changed class): "
        f"{n_contested} ({100*n_contested/n_total:.1f}%)"
    )

    if n_contested > 0 and class_col in df.columns:
        # Per-pipeline breakdown of contested rows
        contested_by_pipe = (
            df[df[contested_col]]
            .groupby("pipeline")[contested_col]
            .count()
            .rename("n_contested")
        )
        total_by_pipe = df.groupby("pipeline")[contested_col].count().rename("n_total")
        breakdown = pd.concat([contested_by_pipe, total_by_pipe], axis=1).fillna(0)
        breakdown["frac_contested"] = breakdown["n_contested"] / breakdown["n_total"]
        print("\n  Contested fraction by pipeline:")
        print(breakdown[["n_contested", "n_total", "frac_contested"]].to_string())

        # Transition matrix: what classifications changed to what
        transitions = (
            df[df[contested_col]]
            .groupby([class_col, agnostic_class_col])
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={
                class_col: "full_classification",
                agnostic_class_col: "agnostic_classification",
            })
        )
        print("\n  Classification transitions (full -> agnostic):")
        print(transitions.to_string(index=False))

    return df