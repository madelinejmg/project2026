"""
nemesis_hlsp_pipeline.py

Unified pipeline for comparative analysis of TESS HLSP photometry and
NEMESIS custom photometry with LEO-Vetter transit vetting.

Workflow
--------
For each target:
  1. Run NEMESIS (full_pipeline) first -- all sectors, no transit search.
  2. Standardize NEMESIS output into the shared schema.
  3. Download and standardize HLSP pipelines (QLP, TESS-SPOC, TGLC,
     GSFC-ELEANOR-LITE) from MAST, concatenated across all available sectors.
  4. Phase-fold all pipelines together and save a comparison figure.
  5. Run Apply_LEOVetter on every pipeline (HLSPs + NEMESIS).
  6. Stack per-pipeline LEO-Vetter results into a single catalog Parquet:
         TIC{ID}_{sector_tag}_leovetter_catalog.parquet
     One row per pipeline; shared columns repeat; LEOVetter_* columns vary.
  7. Persist per-pipeline concat LC Parquets and a pipeline comparison figure.

Entry points
------------
  target_to_lightcurve_workflow_V5  -- single target, returns combined_results dict
  compile_leovetter_catalog         -- stack leo_vetter_vetted per pipeline
  run_single_target                 -- single target with full persistence + manifest
  run_batch                         -- batch loop over a catalog DataFrame

External dependencies (must be importable)
------------------------------------------
  standardizing_data                -- collect_lightcurves_for_target,
                                       normalize_standardized_lc,
                                       _apply_quality_mask, _get_colors,
                                       DEFAULT_CADENCE, DEFAULT_DOWNLOADPATH,
                                       DEFAULT_RADIUS
  standardizing_data_caching        -- save_pipeline_results, load_pipeline_results
  NEMESIS_pipeline_new_temp_4       -- full_pipeline, default_NEMESIS_pipeline_settings,
                                       get_qld, plot_modshift_NEMESIS
  phasefold                         -- plot_phasefolded

Output layout
-------------
{RESULTS_ROOT}/
  run_manifest.csv
  logs/
    TIC{ID}.log
  saved_results/
    nemesis_output/
      TIC{ID}/                          <- full_pipeline raw outputs
    TIC{ID}_{sector_tag}/
      NEMESIS_concat_standardized.parquet
      NEMESIS_concat_standardized_masked.parquet
      NEMESIS_leovetter_metrics.parquet
      {PIPELINE}_concat_standardized.parquet
      {PIPELINE}_concat_standardized_masked.parquet
      {PIPELINE}_leovetter_metrics.parquet
      TIC{ID}_{sector_tag}_leovetter_catalog.parquet
      TIC{ID}_{sector_tag}_pipeline_comparison.png
      TIC_{ID}_{sector_tag}_phasefold_comparison.png
      TIC_{ID}_{sector_tag}_{PIPELINE}_LEOVetter_Report.png
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import time
import time as clock
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip

# ---------------------------------------------------------------------------
# NEMESIS pipeline path -- adjust to your local install
# ---------------------------------------------------------------------------
pipeline_path = "/Users/daxfeliz/Desktop/TESS/00_Current_Pipeline/"
if pipeline_path not in sys.path:
    sys.path.append(pipeline_path)

# ---------------------------------------------------------------------------
# External module imports
# ---------------------------------------------------------------------------
from standardizing_data import (
    DEFAULT_CADENCE,
    DEFAULT_DOWNLOADPATH,
    DEFAULT_RADIUS,
    _apply_quality_mask,
    get_pipeline_color,
    collect_lightcurves_for_target,
    normalize_standardized_lc,
)

from NEMESIS_pipeline_new_temp_4 import (
    default_NEMESIS_pipeline_settings,
    full_pipeline,
    get_qld,
    plot_modshift_NEMESIS,
)
from phasefold import plot_phasefolded


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types from pandas/.item() calls."""
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_pipeline_results(
    results: Dict[str, Dict[str, Any]],
    savepath: str,
    tic_id: Union[int, str],
    sector: int,
    *,
    overwrite: bool = False,
) -> str:
    """
    Persist the output of ``collect_lightcurves_for_target`` to disk.

    Layout on disk::

        {savepath}/
          TIC{tic_id}_S{sector}/
            metadata.json                    ← status, errors, row counts
            {PIPELINE}_raw.parquet
            {PIPELINE}_standardized.parquet
            {PIPELINE}_standardized_masked.parquet

    Parquet is used for DataFrames (fast I/O, exact dtype preservation).
    ``None`` DataFrames (failed pipelines or disabled masking) are skipped
    silently — their absence is recorded in ``metadata.json``.

    Parameters
    ----------
    results : dict
        As returned by ``collect_lightcurves_for_target``.
    savepath : str
        Root directory under which the ``TIC{id}_S{sector}/`` folder is
        created.
    tic_id : int | str
        TIC identifier (used only for the folder name).
    sector : int
        Sector number (used only for the folder name).
    overwrite : bool
        If False (default) and the target directory already exists, raise
        FileExistsError.  If True, existing files are overwritten silently.

    Returns
    -------
    outdir : str
        Absolute path to the ``TIC{tic_id}_S{sector}/`` directory.

    Raises
    ------
    FileExistsError
        If the output directory already exists and ``overwrite=False``.

    Examples
    --------
    >>> outdir = save_pipeline_results(results, "/data/comparisons",
    ...                                tic_id=259377017, sector=3)
    >>> print(outdir)
    /data/comparisons/TIC259377017_S3
    """
    outdir = os.path.join(savepath, f"TIC{tic_id}_S{sector}")
    if os.path.exists(outdir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {outdir}\n"
            "Pass overwrite=True to replace existing files."
        )
    os.makedirs(outdir, exist_ok=True)

    metadata: Dict[str, Any] = {}

    for pipeline, info in results.items():
        meta_entry: Dict[str, Any] = {
            "tic_id": str(info.get("tic_id", tic_id)),
            "sector": info.get("sector", sector),
            "pipeline": pipeline,
            "status": info.get("status", "unknown"),
            "error": info.get("error"),
            "n_raw": info.get("n_raw"),
            "n_standardized": info.get("n_standardized"),
            "n_masked": info.get("n_masked"),
            "files": {},
        }

        for key in ("raw", "standardized", "standardized_masked"):
            df = info.get(key)
            if df is not None and not df.empty:
                fname = f"{pipeline}_{key}.parquet"
                fpath = os.path.join(outdir, fname)
                df.to_parquet(fpath, index=False)
                meta_entry["files"][key] = fname

        metadata[pipeline] = meta_entry

    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2, cls=_NumpyEncoder)

    return outdir


def load_pipeline_results(
    savepath: str,
    tic_id: Union[int, str],
    sector: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Reload a results dict previously saved by ``save_pipeline_results``.

    Reconstructs the same structure as ``collect_lightcurves_for_target``
    (minus the ``product`` field, which is not serialised).

    Parameters
    ----------
    savepath : str
        Root directory passed to ``save_pipeline_results``.
    tic_id : int | str
        TIC identifier.
    sector : int
        Sector number.

    Returns
    -------
    results : dict
        Keyed by pipeline name.  Each value has:
        ``raw``, ``standardized``, ``standardized_masked`` (DataFrames or
        None), plus all scalar metadata fields.  ``product`` is always None.

    Raises
    ------
    FileNotFoundError
        If the expected directory or ``metadata.json`` is missing.

    Examples
    --------
    >>> results = load_pipeline_results("/data/comparisons", 259377017, 3)
    >>> results["QLP"]["standardized"].head()
    """
    outdir = os.path.join(savepath, f"TIC{tic_id}_S{sector}")
    meta_path = os.path.join(outdir, "metadata.json")

    if not os.path.isdir(outdir):
        raise FileNotFoundError(f"Results directory not found: {outdir}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.json not found in: {outdir}")

    try:
        with open(meta_path, "r") as fh:
            metadata = json.load(fh)
    except json.JSONDecodeError as exc:
        raise FileNotFoundError(
            f"metadata.json at {meta_path} is corrupt (truncated by a prior "
            f"failed write). Treating as cache miss. Original error: {exc}\n"
            f"The directory will be overwritten on next save."
        ) from exc

    results: Dict[str, Dict[str, Any]] = {}

    for pipeline, meta in metadata.items():
        entry: Dict[str, Any] = {
            "product": None,  # not serialised
            "raw": None,
            "standardized": None,
            "standardized_masked": None,
            "tic_id": meta.get("tic_id"),
            "sector": meta.get("sector"),
            "pipeline": pipeline,
            "n_raw": meta.get("n_raw"),
            "n_standardized": meta.get("n_standardized"),
            "n_masked": meta.get("n_masked"),
            "status": meta.get("status"),
            "error": meta.get("error"),
        }

        for key in ("raw", "standardized", "standardized_masked"):
            fname = meta.get("files", {}).get(key)
            if fname:
                fpath = os.path.join(outdir, fname)
                if os.path.isfile(fpath):
                    entry[key] = pd.read_parquet(fpath)

        results[pipeline] = entry

    return results  


def _load_nemesis_from_cache(
    tic_id: Union[int, str],
    results_savepath: str,
    sector_tag: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Reconstruct a NEMESIS combined_results entry from the parquets saved
    by _persist_combined_results, without re-running full_pipeline.

    Used when force_releovetter=True to skip the NEMESIS photometry step
    while still providing the raw LC_df to Apply_LEOVetter.

    The "raw" key is populated from NEMESIS_concat_standardized_masked.parquet
    remapped to NEMESIS LC_df column names (Time, SAP Flux, Detrended Flux,
    Detrended Error). This is the same DataFrame that Apply_LEOVetter receives
    in the normal run path.

    Parameters
    ----------
    tic_id : int or str
    results_savepath : str
        The saved_results root, e.g. RESULTS_ROOT/saved_results/
    sector_tag : str
        e.g. "S0002" or "S0001to0026". Used to locate the output directory.
    verbose : bool

    Returns
    -------
    dict
        Same structure as _build_nemesis_combined_entry output.
        status="ok" if parquets found, "error" otherwise.
    """
    outdir = os.path.join(results_savepath, f"TIC{tic_id}_{sector_tag}")

    _err = {
        "product": None, "raw": None,
        "standardized": None, "standardized_masked": None,
        "tic_id": tic_id, "sector": [], "sectors": [],
        "pipeline": "NEMESIS",
        "n_raw": 0, "n_standardized": 0, "n_masked": 0,
        "status": "error",
        "error": None,
        "leo_vetter_metrics": None, "leo_vetter_vetted": None,
        "lv_T0_refined": None, "lv_T0_offset_hours": 0.0,
    }

    masked_path = os.path.join(outdir, "NEMESIS_concat_standardized_masked.parquet")
    std_path    = os.path.join(outdir, "NEMESIS_concat_standardized.parquet")

    if not os.path.isfile(masked_path) and not os.path.isfile(std_path):
        err = f"No cached NEMESIS parquets found in {outdir}"
        if verbose:
            print(f"[cache] NEMESIS: {err}")
        result = dict(_err)
        result["error"] = err
        return result

    try:
        if os.path.isfile(masked_path):
            std_masked = pd.read_parquet(masked_path)
            std_norm   = pd.read_parquet(std_path) if os.path.isfile(std_path) else std_masked
        else:
            std_masked = pd.read_parquet(std_path)
            std_norm   = std_masked
    except Exception as exc:
        result = dict(_err)
        result["error"] = f"Failed to load NEMESIS parquets: {exc}"
        return result

    # Reconstruct the raw LC_df that Apply_LEOVetter expects.
    # The standardized_masked parquet uses the standardized schema
    # (time, flux_raw, flux_corr, flux_corr_err, ...).
    # Apply_LEOVetter for NEMESIS reads info["raw"] which has
    # NEMESIS LC_df column names (Time, SAP Flux, Detrended Flux, Detrended Error).
    # Remap here so the LEO-Vetter loop can use it unchanged.
    try:
        col_map = {
            "time":          "Time",
            "flux_raw":      "SAP Flux",
            "flux_corr":     "Detrended Flux",
            "flux_corr_err": "Detrended Error",
        }
        raw_cols = {v: std_masked[k].to_numpy(dtype=float)
                    for k, v in col_map.items() if k in std_masked.columns}

        # If flux_corr_err is all-NaN (TGLC/ELEANOR pattern shouldn't apply
        # to NEMESIS, but guard anyway)
        if "Detrended Error" not in raw_cols or not np.any(
            np.isfinite(raw_cols.get("Detrended Error", np.array([np.nan])))
        ):
            from astropy.stats import mad_std as _mad_std
            flux = raw_cols.get("Detrended Flux", np.ones(len(std_masked)))
            finite = flux[np.isfinite(flux)]
            raw_cols["Detrended Error"] = np.full(
                len(std_masked),
                float(_mad_std(finite)) if finite.size > 0 else 1.0,
            )

        lc_raw = pd.DataFrame(raw_cols)
        valid = (
            np.isfinite(lc_raw["Time"])
            & np.isfinite(lc_raw["Detrended Flux"])
            & np.isfinite(lc_raw["Detrended Error"])
            & (lc_raw["Detrended Error"] > 0)
        )
        lc_raw = lc_raw.loc[valid].reset_index(drop=True)
    except Exception as exc:
        result = dict(_err)
        result["error"] = f"Failed to reconstruct NEMESIS raw LC_df: {exc}"
        return result

    sectors = (
        sorted(std_norm["sector_index"].dropna().astype(int).unique().tolist())
        if "sector_index" in std_norm.columns else []
    )

    if verbose:
        print(f"[cache] NEMESIS: loaded {len(lc_raw)} cadences from {outdir}")

    return {
        "product": None,
        "raw":                 lc_raw,
        "standardized":        std_norm.reset_index(drop=True),
        "standardized_masked": std_masked.reset_index(drop=True),
        "tic_id":  tic_id,
        "sector":  sectors,
        "sectors": sectors,
        "pipeline": "NEMESIS",
        "n_raw":          len(std_norm),
        "n_standardized": len(std_norm),
        "n_masked":       len(std_masked),
        "status": "ok",
        "error": None,
        "leo_vetter_metrics": None, "leo_vetter_vetted": None,
        "lv_T0_refined": None, "lv_T0_offset_hours": 0.0,
    }


def find_cached_sector_tag(
    tic_id: Union[int, str],
    results_savepath: str,
) -> Optional[str]:
    """
    Scan results_savepath for an existing TIC{tic_id}_* directory that
    contains a NEMESIS_concat_standardized_masked.parquet file.

    Used by force_releovetter to find the correct sector tag without
    re-querying MAST, avoiding mismatches when MAST availability has
    changed or max_sector differs between runs.

    Returns the sector tag string (e.g. "S0002" or "S0001to0026") if
    found, or None if no cached directory exists for this target.
    """
#     pattern = os.path.join(results_savepath, f"TIC{tic_id}_S*")
#     import glob as _glob
#     candidates = sorted(_glob.glob(pattern))

#     for cand in candidates:
#         nemesis_parquet = os.path.join(
#             cand, "NEMESIS_concat_standardized_masked.parquet"
#         )
#         if os.path.isfile(nemesis_parquet):
#             # Extract the sector tag from the directory name
#             dirname   = os.path.basename(cand)
#             sector_tag = dirname.replace(f"TIC{tic_id}_", "", 1)
#             return sector_tag

#     return None
    import glob as _glob

    pattern    = os.path.join(results_savepath, f"TIC{tic_id}_S*")
    candidates = sorted(_glob.glob(pattern))

    if not candidates:
        return None

    # Prefer a directory that has the NEMESIS standardized parquet
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "NEMESIS_concat_standardized_masked.parquet")):
            dirname = os.path.basename(cand)
            return dirname.replace(f"TIC{tic_id}_", "", 1)

    # No directory has the NEMESIS parquet -- fall back to the directory with
    # the most parquet files (most complete run). This covers the case where
    # NEMESIS photometry ran but _persist_combined_results failed to write the
    # standardized parquet for NEMESIS specifically.
    best_cand  = max(candidates, key=lambda c: len(_glob.glob(os.path.join(c, "*.parquet"))))
    dirname    = os.path.basename(best_cand)
    return dirname.replace(f"TIC{tic_id}_", "", 1)


def diagnose_releovetter_failures(
    catalog_df: pd.DataFrame,
    results_root: str ,
    pipelines: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For each target in catalog_df, check whether the caches needed by
    force_releovetter actually exist on disk. Identifies the root cause
    of Sauto / Sunknown failures before running the full batch.

    Checks performed per target:
      1. Does a TIC{id}_S* output directory exist?
      2. Does it contain NEMESIS_concat_standardized_masked.parquet?
      3. Does it contain at least one HLSP _concat_standardized_masked.parquet?
      4. Does the per-sector HLSP cache TIC{id}_S{N}/ exist for the
         sectors implied by the sector tag?

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Target list. Must contain 'TIC ID'.
    results_root : str
    pipelines : list of str or None
        HLSP pipeline names to check. Default PIPELINES module constant.

    Returns
    -------
    pd.DataFrame
        One row per target with columns:
        tic_id, sector_tag_found, has_nemesis_cache, has_hlsp_cache,
        missing_hlsp_sector_dirs, diagnosis.

    Example
    -------
    >>> diag = diagnose_releovetter_failures(filtered_df)
    >>> print(diag[diag["diagnosis"] != "ok"])
    """
    if pipelines is None:
        pipelines = PIPELINES

    results_savepath = os.path.join(results_root, "saved_results")
    rows = []

    for _, target in catalog_df.iterrows():
        tic_id = int(target["TIC ID"])

        # 1. Find the output directory
        stag = find_cached_sector_tag(tic_id, results_savepath)

        if stag is None:
            rows.append({
                "tic_id":                  tic_id,
                "sector_tag_found":        None,
                "has_nemesis_cache":       False,
                "has_hlsp_cache":          False,
                "missing_hlsp_sector_dirs": "",
                "diagnosis": "NO_OUTPUT_DIR -- target never ran or outputs deleted",
            })
            continue

        outdir = os.path.join(results_savepath, f"TIC{tic_id}_{stag}")

        # 2. NEMESIS parquet
        has_nemesis = os.path.isfile(
            os.path.join(outdir, "NEMESIS_concat_standardized_masked.parquet")
        )

        # 3. At least one HLSP parquet
        hlsp_parquets = [
            os.path.join(outdir, f"{p}_concat_standardized_masked.parquet")
            for p in pipelines
        ]
        has_hlsp = any(os.path.isfile(f) for f in hlsp_parquets)

        # 4. Per-sector HLSP cache directories
        # Parse sector numbers from stag (e.g. "S0001to0026" -> [1..26],
        # "S0002" -> [2])
        missing_sector_dirs = []
        try:
            if "to" in stag:
                parts = stag.lstrip("S").split("to")
                s_min, s_max = int(parts[0]), int(parts[1])
                sector_nums = list(range(s_min, s_max + 1))
            else:
                sector_nums = [int(stag.lstrip("S"))]

            for s in sector_nums:
                sdir = os.path.join(results_savepath, f"TIC{tic_id}_S{s}")
                if not os.path.isdir(sdir):
                    missing_sector_dirs.append(s)
        except Exception:
            sector_nums = []

        # Diagnosis
        if not has_nemesis and not has_hlsp:
            diagnosis = "NO_PARQUETS -- output dir exists but no pipeline parquets found"
        elif not has_nemesis:
            diagnosis = "MISSING_NEMESIS -- HLSP parquets ok but NEMESIS parquet absent"
        elif not has_hlsp:
            diagnosis = "MISSING_HLSP -- NEMESIS parquet ok but no HLSP parquets found"
        elif missing_sector_dirs:
            diagnosis = (
                f"MISSING_SECTOR_CACHE -- per-sector dirs absent for sectors "
                f"{missing_sector_dirs}; HLSP re-download will trigger"
            )
        else:
            diagnosis = "ok"

        rows.append({
            "tic_id":                   tic_id,
            "sector_tag_found":         stag,
            "has_nemesis_cache":        has_nemesis,
            "has_hlsp_cache":           has_hlsp,
            "missing_hlsp_sector_dirs": str(missing_sector_dirs) if missing_sector_dirs else "",
            "diagnosis":                diagnosis,
        })

    diag_df = pd.DataFrame(rows)

    # Summary
    counts = diag_df["diagnosis"].value_counts()
    print("\n[diagnose_releovetter_failures] Summary:")
    for label, count in counts.items():
        print(f"  {count:4d}  {label}")
    n_ok = (diag_df["diagnosis"] == "ok").sum()
    print(f"\n  {n_ok}/{len(diag_df)} targets are ready for force_releovetter=True")

    return diag_df


# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(DEFAULT_DOWNLOADPATH, "batch_results")
PIPELINES    = ["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]
FORCE_RERUN  = False
VERBOSE      = False

MANIFEST_PATH = os.path.join(RESULTS_ROOT, "run_manifest.csv")
MANIFEST_COLS = [
    "tic_id", "status", "sectors_found", "sector_tag",
    "pipelines_ok", "pipelines_failed",
    "nemesis_status",
    "nemesis_error",
    "started_at", "finished_at", "error",
]

# Transit parameter column name mapping for nearby_TOI_MD_df.
# Override at call-site via transit_param_cols if your catalog uses
# different column names.
TRANSIT_PARAM_COLS: Dict[str, str] = {
    "Period":   "Orbital Period (days) Value",
    "T0":       "Orbital Epoch Value",
    "Duration": "Transit Duration (hours) Value",
}

# Gap threshold (days) used to infer sector boundaries from the NEMESIS
# concatenated LC (which has no explicit sector label column).
GAP_THRESHOLD_DAYS: float = 5.0


# ===========================================================================
# Section 1: NEMESIS schema mapping and standardization
# ===========================================================================

# NEMESIS QUALITY_FLAG bitmask:
#   0 = clean
#   1 = quaternion jitter
#   2 = background / centroid spike
#   3 = jitter + bkg/centroid
#   4 = isolated outlier (NaN flux_corr, valid flux_raw)
#   5-7 = outlier + jitter/bkg combos
# All nonzero values excluded by _apply_quality_mask (quality == 0).

_NEMESIS_COL_MAP: Dict[str, Optional[str]] = {
    "time":          "Time",
    "flux_raw":      "SAP Flux",
    "flux_raw_err":  "SAP Error",
    "flux_corr":     "Detrended Flux",
    "flux_corr_err": "Detrended Error",
    "flux_bkg":      "Background Flux",
    "flux_bkg_err":  "Background Error",
    "quality":       "QUALITY_FLAG",
    "quality2":      None,   # no equivalent; set to NaN (consistent with non-TGLC HLSPs)
}
_VALID_QUALITY_VALUES = {0, 1, 2, 3, 4, 5, 6, 7}


def _map_lc_df_to_schema(
    LC_df: pd.DataFrame,
    *,
    tic_id: Optional[int] = None,
    strict: bool = False,
    case_insensitive: bool = True,
) -> pd.DataFrame:
    """
    Remap NEMESIS LC_df columns to the standardized HLSP schema.

    All cadences are preserved. Isolated outlier rows (QUALITY_FLAG=4)
    retain valid flux_raw but have NaN flux_corr; they are excluded
    downstream by _apply_quality_mask.

    Parameters
    ----------
    LC_df : pd.DataFrame
        Raw NEMESIS LC_df from full_pipeline.
    tic_id : int or None
        Used in warning messages only.
    strict : bool
        Raise KeyError on missing required columns if True.
    case_insensitive : bool
        Match source column names case-insensitively.

    Returns
    -------
    pd.DataFrame
        Standardized schema with columns: time, flux_raw, flux_raw_err,
        flux_corr, flux_corr_err, flux_bkg, flux_bkg_err, quality, quality2.
    """
    tag = f"TIC {tic_id}" if tic_id is not None else "NEMESIS"

    if case_insensitive:
        col_lookup = {str(c).casefold(): c for c in LC_df.columns}
        def _resolve(name: str) -> Optional[str]:
            return col_lookup.get(name.casefold())
    else:
        def _resolve(name: str) -> Optional[str]:
            return name if name in LC_df.columns else None

    out = pd.DataFrame(index=LC_df.index)
    for schema_col, src_col in _NEMESIS_COL_MAP.items():
        if src_col is None:
            out[schema_col] = np.nan
            continue
        actual = _resolve(src_col)
        if actual is None:
            msg = (
                f"[{tag}] _map_lc_df_to_schema: expected column '{src_col}' "
                f"not found. Available: {list(LC_df.columns)}"
            )
            if strict:
                raise KeyError(msg)
            warnings.warn(msg, stacklevel=3)
            out[schema_col] = np.nan
            continue
        out[schema_col] = LC_df[actual].to_numpy(copy=False)

    out["quality"] = (
        pd.to_numeric(out["quality"], errors="coerce").fillna(0).astype(int)
    )
    observed = set(out["quality"].unique().tolist())
    unexpected = observed - _VALID_QUALITY_VALUES
    if unexpected:
        warnings.warn(
            f"[{tag}] Unexpected QUALITY_FLAG values: {unexpected}.",
            stacklevel=3,
        )
    return out


def _split_into_sectors(
    std_df: pd.DataFrame,
    gap_threshold_days: float = GAP_THRESHOLD_DAYS,
) -> List[pd.DataFrame]:
    """
    Split a concatenated LC on time gaps exceeding gap_threshold_days.

    NaN timestamps are treated as zero-gap so they never spuriously trigger
    a sector boundary.

    Parameters
    ----------
    std_df : pd.DataFrame
        Standardized LC DataFrame with a 'time' column.
    gap_threshold_days : float

    Returns
    -------
    list of pd.DataFrame
        One sub-DataFrame per inferred sector segment.
    """
    t = std_df["time"].to_numpy(dtype=float)
    finite = np.isfinite(t)
    if finite.sum() < 2:
        return [std_df.reset_index(drop=True)]
    dt = np.zeros(len(t))
    dt[1:] = np.diff(t)
    dt[~finite] = 0.0
    gaps = np.where(dt > gap_threshold_days)[0]
    if len(gaps) == 0:
        return [std_df.reset_index(drop=True)]
    bounds = [0] + gaps.tolist() + [len(std_df)]
    return [
        std_df.iloc[bounds[i]: bounds[i + 1]].reset_index(drop=True)
        for i in range(len(bounds) - 1)
        if bounds[i + 1] > bounds[i]
    ]


def _normalize_per_sector_and_concat(
    std_df: pd.DataFrame,
    gap_threshold_days: float = GAP_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """
    Gap-split std_df, normalize each segment independently, then concat.

    Uses normalize_standardized_lc (nanmedian-based), so NaN outlier rows
    (QUALITY_FLAG=4) do not distort the per-sector baseline. Adds a
    'sector_index' column (0-based, inferred from gaps).

    Parameters
    ----------
    std_df : pd.DataFrame
    gap_threshold_days : float

    Returns
    -------
    pd.DataFrame
        Normalized and concatenated DataFrame with 'sector_index' column.
    """
    segments = _split_into_sectors(std_df, gap_threshold_days)
    normed = []
    for i, seg in enumerate(segments):
        n = normalize_standardized_lc(seg).copy()
        n["sector_index"] = i
        normed.append(n)
    return pd.concat(normed, ignore_index=True)


def run_nemesis_and_standardize(
    tic_id: int,
    nemesis_output_path: str,
    cadence: str = DEFAULT_CADENCE,
    nemesis_settings: Optional[Dict[str, Any]] = None,
    gap_threshold_days: float = GAP_THRESHOLD_DAYS,
    strict: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run full_pipeline for one target (all sectors, no transit search) and
    return a standardized result dict compatible with combined_results.

    do_transit_search, get_all_sectors, and verbose are always forced
    regardless of what nemesis_settings contains.

    Parameters
    ----------
    tic_id : int
    nemesis_output_path : str
        Root output directory passed to full_pipeline as `path`.
    cadence : str
    nemesis_settings : dict or None
        Overrides for default_NEMESIS_pipeline_settings.
    gap_threshold_days : float
        Time gap used to infer sector boundaries.
    strict : bool
        Raise KeyError on missing LC_df columns if True.
    verbose : bool

    Returns
    -------
    dict
        Keys: status, pipeline, standardized, standardized_masked,
              sectors, n_standardized, n_masked, raw, nemesis_settings, error.

    Example
    -------
    >>> result = run_nemesis_and_standardize(
    ...     tic_id=410153553,
    ...     nemesis_output_path="/data/nemesis_output/TIC410153553/",
    ... )
    >>> result["status"]
    'ok'
    >>> result["standardized"].columns.tolist()
    ['time', 'flux_raw', 'flux_raw_err', 'flux_corr', 'flux_corr_err',
     'flux_bkg', 'flux_bkg_err', 'quality', 'quality2', 'sector_index']
    """
    _err: Dict[str, Any] = {
        "status": "error", "pipeline": "NEMESIS",
        "standardized": None, "standardized_masked": None,
        "sectors": [], "n_standardized": 0, "n_masked": 0,
        "raw": None, "nemesis_settings": {}, "error": None,
    }

    base_settings = default_NEMESIS_pipeline_settings()
    if nemesis_settings is not None:
        base_settings.update(nemesis_settings)

    try:
        _hdu, LC_df, _transit_results, resolved_settings = full_pipeline(
            ID=tic_id,
            path=nemesis_output_path,
            cadence=cadence,
            Sector=None,
            settings=base_settings,
            do_transit_search=False,
            get_all_sectors=True,
            verbose=verbose,
        )
    except Exception as exc:
        result = dict(_err)
        result["error"] = f"full_pipeline raised: {exc}\n{traceback.format_exc()}"
        return result

    if LC_df is None or len(LC_df) == 0:
        result = dict(_err)
        result["nemesis_settings"] = resolved_settings
        result["error"] = (
            f"TIC {tic_id}: full_pipeline returned empty LC_df. "
            "Check MAST availability for this target."
        )
        return result

    try:
        std_raw = _map_lc_df_to_schema(LC_df, tic_id=tic_id, strict=strict)
    except Exception as exc:
        result = dict(_err)
        result["raw"] = LC_df
        result["nemesis_settings"] = resolved_settings
        result["error"] = f"_map_lc_df_to_schema failed: {exc}"
        return result

    if std_raw["time"].isna().all():
        result = dict(_err)
        result["raw"] = LC_df
        result["nemesis_settings"] = resolved_settings
        result["error"] = f"TIC {tic_id}: 'Time' column is all-NaN after mapping."
        return result

    if std_raw["flux_raw"].isna().all():
        warnings.warn(f"TIC {tic_id}: 'flux_raw' is all-NaN.", stacklevel=2)
    if std_raw["flux_corr"].isna().all():
        warnings.warn(f"TIC {tic_id}: 'flux_corr' is all-NaN.", stacklevel=2)

    try:
        std_norm = _normalize_per_sector_and_concat(std_raw, gap_threshold_days)
    except Exception as exc:
        result = dict(_err)
        result["raw"] = LC_df
        result["nemesis_settings"] = resolved_settings
        result["error"] = f"Per-sector normalization failed: {exc}"
        return result

    try:
        std_masked = _apply_quality_mask(std_norm, pipeline="NEMESIS")
    except Exception as exc:
        result = dict(_err)
        result["raw"] = LC_df
        result["nemesis_settings"] = resolved_settings
        result["error"] = f"_apply_quality_mask failed: {exc}"
        return result

    sectors: List[int] = (
        sorted(std_norm["sector_index"].dropna().astype(int).unique().tolist())
        if "sector_index" in std_norm.columns else []
    )

    return {
        "status":              "ok",
        "pipeline":            "NEMESIS",
        "standardized":        std_norm.reset_index(drop=True),
        "standardized_masked": std_masked.reset_index(drop=True),
        "sectors":             sectors,
        "n_standardized":      len(std_norm),
        "n_masked":            len(std_masked),
        "raw":                 LC_df,
        "nemesis_settings":    resolved_settings,
        "error":               None,
    }


# ===========================================================================
# Section 2: HLSP / MAST helpers
# ===========================================================================

def get_available_sectors__OLD(
    tic_id: Union[int, str],
    exptime: str = "1800",
    pipelines: Optional[List[str]] = None,
    radius: Union[float, Any] = 0.0001,
    *,
    max_sector: Optional[int] = 26,
    verbose: bool = True,
) -> List[int]:
    """
    Query MAST for all TESS sectors with a published HLSP at the given cadence.

    Parameters
    ----------
    tic_id : int or str
    exptime : str
        Use "1800" for 30-min FFI cadence, "120" for 2-min.
    pipelines : list of str or None
        Filter to sectors that have at least one product from these pipelines.
    radius : float or astropy Quantity
        Cone-search radius. Default 0.0001 deg.
    max_sector : int or None
        Discard sectors above this number. Default 26 (primary mission).
    verbose : bool

    Returns
    -------
    list of int
        Sorted unique sector numbers.

    Example
    -------
    >>> get_available_sectors(410153553, pipelines=["QLP", "TESS-SPOC"])
    [1]
    """
    import lksearch as _lk
    tic_str = str(int(float(str(tic_id).strip())))
    search_radius = float(radius) if isinstance(radius, (int, float, np.floating)) else radius

    try:
        search = _lk.TESSSearch(
            target=f"TIC {tic_str}",
            search_radius=search_radius,
            exptime=exptime,
            hlsp=True,
        )
        try:
            ts = search.timeseries
        except Exception:
            ts = search
        table = getattr(ts, "table", None)
    except Exception as exc:
        if verbose:
            print(f"[get_available_sectors] MAST query failed for TIC {tic_str}: {exc}")
        return []

    if table is None or len(table) == 0:
        if verbose:
            print(f"[get_available_sectors] No HLSP products for TIC {tic_str}.")
        return []

    if not isinstance(table, pd.DataFrame):
        try:
            table = table.to_pandas()
        except Exception:
            table = pd.DataFrame(table)

    if "mission" in table.columns:
        table = table[table["mission"].astype(str).str.upper().eq("HLSP")].copy()
    if pipelines is not None and "pipeline" in table.columns:
        table = table[table["pipeline"].astype(str).isin(pipelines)].copy()
    if len(table) == 0:
        return []

    sector_col: Optional[str] = None
    for c in ("sector", "sequence_number"):
        if c in table.columns:
            sector_col = c
            break
    if sector_col is None:
        return []

    raw_vals = table[sector_col].dropna().astype(str).str.strip()
    sectors: List[int] = sorted({
        int(float(v)) for v in raw_vals
        if v.lower() not in {"nan", "none", ""}
    })
    if max_sector is not None:
        sectors = [s for s in sectors if s <= max_sector]

    if verbose:
        print(f"[get_available_sectors] TIC {tic_str}: found sectors {sectors} "
              f"at exptime={exptime!r}")
    return sectors


from typing import Any, Dict, List, Literal, Optional, Union
def get_available_sectors(
    tic_id: Union[int, str],
    exptime: str = "1800",
    pipelines: Optional[List[str]] = None,
    radius: Union[float, Any] = 0.0001,
    *,
    max_sector: Optional[int] = 26,
    verbose: bool = True,
) -> List[int]:
    """
    Query MAST for all TESS sectors with a published HLSP at the given cadence.

    Robust against lksearch IndexError that occurs when MAST returns sectors
    whose numbers exceed lksearch's internal pointings table length. This
    happens when lksearch is out of date relative to current MAST holdings
    (e.g. TESS Year 7+ sectors not yet in the lksearch pointings table).
    The error is caught and the query falls back to a HLSP-only search that
    does not require the pointings table.
    """
    import lksearch as _lk

    tic_str       = str(int(float(str(tic_id).strip())))
    search_radius = float(radius) if isinstance(radius, (int, float, np.floating)) else radius

    def _parse_sectors(table: pd.DataFrame) -> List[int]:
        """Extract and filter sector numbers from a search result table."""
        if table is None or len(table) == 0:
            return []
        if not isinstance(table, pd.DataFrame):
            try:
                table = table.to_pandas()
            except Exception:
                table = pd.DataFrame(table)
        if "mission" in table.columns:
            table = table[table["mission"].astype(str).str.upper().eq("HLSP")].copy()
        if pipelines is not None and "pipeline" in table.columns:
            table = table[table["pipeline"].astype(str).isin(pipelines)].copy()
        if len(table) == 0:
            return []
        sector_col = next((c for c in ("sector", "sequence_number") if c in table.columns), None)
        if sector_col is None:
            return []
        raw_vals = table[sector_col].dropna().astype(str).str.strip()
        sectors: List[int] = sorted({
            int(float(v)) for v in raw_vals
            if v.lower() not in {"nan", "none", ""}
        })
        if max_sector is not None:
            sectors = [s for s in sectors if s <= max_sector]
        return sectors

    # ── Attempt 1: normal query (TESScut + HLSP products) ────────────────────
    try:
        search = _lk.TESSSearch(
            target=f"TIC {tic_str}",
            search_radius=search_radius,
            exptime=exptime,
            hlsp=True,
        )
        try:
            ts    = search.timeseries
        except Exception:
            ts    = search
        table = getattr(ts, "table", None)
        sectors = _parse_sectors(table)
        if verbose:
            print(f"[get_available_sectors] TIC {tic_str}: found sectors {sectors} "
                  f"at exptime={exptime!r}")
        return sectors

    except IndexError as exc:
        # lksearch internal pointings table does not cover a sector number
        # returned by MAST (typically TESS Year 7+ sectors). Fall through to
        # the HLSP-only fallback which does not touch the pointings table.
        if verbose:
            print(
                f"[get_available_sectors] TIC {tic_str}: lksearch IndexError "
                f"({exc}). Falling back to HLSP-only query (no TESScut)."
            )

    except Exception as exc:
        if verbose:
            print(f"[get_available_sectors] MAST query failed for TIC {tic_str}: {exc}")
        return []

    # ── Attempt 2: HLSP-only fallback (bypasses TESScut/pointings table) ─────
    # Pass pipeline=pipelines to avoid the TESScut product lookup entirely.
    # lksearch only hits the pointings table when constructing TESScut entries;
    # querying HLSP pipeline products directly does not require it.
    try:
        fallback_pipelines = pipelines if pipelines else ["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]
        search = _lk.TESSSearch(
            target=f"TIC {tic_str}",
            search_radius=search_radius,
            exptime=exptime,
            pipeline=fallback_pipelines,  # explicit pipeline avoids TESScut path
            hlsp=True,
        )
        try:
            ts    = search.timeseries
        except Exception:
            ts    = search
        table = getattr(ts, "table", None)
        sectors = _parse_sectors(table)
        if verbose:
            print(
                f"[get_available_sectors] TIC {tic_str}: fallback found sectors "
                f"{sectors} at exptime={exptime!r}"
            )
        return sectors

    except Exception as exc:
        if verbose:
            print(
                f"[get_available_sectors] TIC {tic_str}: fallback query also failed: {exc}"
            )
        return []


# ===========================================================================
# Section 3: LEO-Vetter helpers
# ===========================================================================

def _standardized_to_nemesis_lc_OLD(std_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a standardized HLSP LC DataFrame to NEMESIS-style column names.

    Used for HLSP pipelines only. For NEMESIS, the raw LC_df from
    full_pipeline is passed directly (it already has the correct columns).

    Parameters
    ----------
    std_df : pd.DataFrame
        Standardized DataFrame with columns: time, flux_corr, flux_corr_err.

    Returns
    -------
    pd.DataFrame
        Columns: Time, SAP Flux, Detrended Flux, Detrended Error.
        Rows where Time or Detrended Flux are non-finite are dropped.
    """
    sap = std_df["flux_corr"].to_numpy(dtype=float)
    err = std_df["flux_corr_err"].to_numpy(dtype=float)

    if not np.any(np.isfinite(err)):
        from astropy.stats import mad_std
        corr_vals = std_df["flux_corr"].to_numpy(dtype=float)
        scatter = (mad_std(corr_vals[np.isfinite(corr_vals)])
                   if np.any(np.isfinite(corr_vals)) else 1.0)
        err = np.full(len(std_df), scatter, dtype=float)

    median    = np.nanmedian(std_df["flux_corr"].to_numpy(dtype=float))
    corr_norm = std_df["flux_corr"].to_numpy(dtype=float) / median
    err_norm  = err / median

    lc_nms = pd.DataFrame({
        "Time":            std_df["time"].to_numpy(dtype=float),
        "SAP Flux":        sap / np.nanmedian(sap),
        "Detrended Flux":  corr_norm,
        "Detrended Error": err_norm,
    }, index=std_df.index)

    valid = np.isfinite(lc_nms["Time"]) & np.isfinite(lc_nms["Detrended Flux"])
    return lc_nms.loc[valid].reset_index(drop=True)

def _standardized_to_nemesis_lc(std_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a standardized HLSP light curve into the NEMESIS-style column
    schema expected by plot_modshift_NEMESIS / Apply_LEOVetter.

    Used for HLSP pipelines only (QLP, TESS-SPOC, TGLC, GSFC-ELEANOR-LITE).
    For NEMESIS, the raw LC_df from full_pipeline is passed directly -- it
    already has Time / SAP Flux / Detrended Flux / Detrended Error.

    Output columns
    --------------
    Time            : cadence timestamps (BTJD)
    SAP Flux        : raw aperture flux, normalized to median ~1.0
                      Uses flux_raw when available; falls back to flux_corr
                      only if flux_raw is entirely NaN.
    Detrended Flux  : corrected flux (flux_corr), normalized to median ~1.0
    Detrended Error : per-cadence uncertainty on flux_corr, normalized by
                      the same median used for Detrended Flux.

    Error estimation for pipelines without a native flux_corr_err column
    ---------------------------------------------------------------------
    TGLC and GSFC-ELEANOR-LITE do not provide a corrected-flux error column
    (flux_corr_err is all-NaN after standardize_lc). The previous fallback
    used a single scalar global MAD of flux_corr as a uniform error for every
    cadence. This has two consequences that directly corrupt LEO-Vetter metrics:

    1. CHI statistic: LEO-Vetter computes CHI as the ratio of the observed
       per-transit depth scatter to the expected scatter from Detrended Error.
       A uniform error ignores cadence-to-cadence noise variation. On M-dwarfs
       with stellar activity, quiet segments have lower true noise than active
       segments, so a uniform error over-weights active cadences (biasing CHI
       low, i.e. toward flagging) and under-weights quiet ones. This creates
       a systematic offset in CHI that is correlated with stellar activity
       level, not transit signal quality.

    2. Fred and modshift significances: the red-noise factor Fred and all
       modshift sigma values (sig_pri, sig_sec, etc.) are computed from
       error-weighted sums over the phased LC. A uniform error reduces these
       to an unweighted sum, losing all information about cadence reliability.

    Replacement strategy: rolling MAD error estimation
    ---------------------------------------------------
    A rolling window of width `error_window` cadences is used to compute the
    local median absolute deviation of flux_corr at each cadence. This
    produces a per-cadence noise estimate that tracks genuine flux variability
    (e.g. stellar flares, systematics) while remaining robust to outliers
    within the window. The window width (default 13) is chosen to be:
      - Wide enough to be statistically stable (~13 points at 30-min cadence
        spans ~6.5 hours, comfortably larger than a typical transit duration).
      - Narrow enough to track short-timescale noise changes without smearing
        activity features into the transit window itself.

    Any cadences where the rolling MAD is zero or NaN (e.g. at the edges of
    the time series, or in perfectly flat segments) fall back to the global
    MAD. If the global MAD is also zero or NaN, a unit error is used as a
    last resort so LEO-Vetter does not receive zero-error inputs.

    SAP Flux column correction
    --------------------------
    The previous implementation set SAP Flux = flux_corr (the corrected flux),
    discarding flux_raw entirely. This is incorrect: SAP Flux is used by
    LEO-Vetter's diagnostic figure panels and is conceptually the raw aperture
    photometry before detrending. For pipelines that provide flux_raw (all
    four HLSPs do), it should be used here. The corrected flux is reserved
    for Detrended Flux.

    Parameters
    ----------
    std_df : pd.DataFrame
        Standardized, quality-masked DataFrame from _apply_quality_mask.
        Expected columns: time, flux_raw, flux_corr, flux_corr_err.
        flux_raw and flux_corr_err may be all-NaN for some pipelines.
    error_window : int
        Rolling window width (number of cadences) for MAD-based error
        estimation. Default 13 (~6.5 hours at 30-min cadence).

    Returns
    -------
    pd.DataFrame
        Columns: Time, SAP Flux, Detrended Flux, Detrended Error.
        Rows where Time or Detrended Flux are non-finite are dropped.
        Index is reset.
    """
    # -- Default window exposed as a module-level constant so callers can
    #    override it without changing the function signature.
    error_window: int = 13

    corr_vals = std_df["flux_corr"].to_numpy(dtype=float)
    err_vals  = std_df["flux_corr_err"].to_numpy(dtype=float)

    # ── Detrended Flux: normalize flux_corr to median ~1.0 ───────────────────
    # normalize_standardized_lc already ran per-sector before concatenation,
    # so this median should already be ~1.0. The division here is a no-op in
    # the normal path but acts as a safety net if the quality mask removed
    # enough cadences to shift the baseline materially.
    finite_corr = corr_vals[np.isfinite(corr_vals)]
    if finite_corr.size == 0 or not np.isfinite(np.nanmedian(finite_corr)):
        raise ValueError(
            "_standardized_to_nemesis_lc: flux_corr is entirely non-finite. "
            "Check upstream quality masking."
        )
    corr_median = float(np.nanmedian(finite_corr))
    corr_norm   = corr_vals / corr_median

    # ── Detrended Error: use flux_corr_err if available, else rolling MAD ────
    has_native_err = np.any(np.isfinite(err_vals))

    if has_native_err:
        # Native error exists (QLP, TESS-SPOC, NEMESIS): normalize by the
        # same median used for flux_corr to preserve the SNR.
        err_norm = err_vals / corr_median
    else:
        # No native error column (TGLC, GSFC-ELEANOR-LITE).
        # Compute a rolling MAD over flux_corr to get a per-cadence noise
        # estimate. This tracks local variability rather than imposing a
        # uniform global scatter on every cadence.
        #
        # Implementation: build a pandas Series, use .rolling().apply() with
        # a robust MAD kernel, then normalize by the same corr_median.
        # Edge cadences (first/last error_window//2 points) where the window
        # is incomplete will be NaN; these are filled by the global MAD below.
        flux_series = pd.Series(corr_vals)

        def _mad(x: np.ndarray) -> float:
            finite = x[np.isfinite(x)]
            if finite.size < 3:
                return np.nan
            return float(np.median(np.abs(finite - np.median(finite))))

        rolling_mad = flux_series.rolling(
            window=error_window,
            center=True,
            min_periods=max(3, error_window // 2),
        ).apply(_mad, raw=True).to_numpy(dtype=float)

        # Global MAD as fallback for edge cadences or degenerate windows.
        global_mad = float(np.median(np.abs(
            finite_corr - np.median(finite_corr)
        ))) if finite_corr.size >= 3 else 1.0
        if not np.isfinite(global_mad) or global_mad == 0.0:
            global_mad = 1.0  # last resort: unit error rather than zero

        # Fill any NaN rolling values with the global MAD.
        bad = ~np.isfinite(rolling_mad) | (rolling_mad == 0.0)
        rolling_mad[bad] = global_mad

        # Normalize by corr_median, consistent with how native errors are
        # handled above.
        err_norm = rolling_mad / corr_median

    # ── SAP Flux: use flux_raw where available; fall back to flux_corr ───────
    # SAP Flux represents the raw (pre-detrending) aperture photometry in the
    # LEO-Vetter figure panels. Using flux_corr here (as the prior version
    # did) discards the raw column entirely and makes both panels of the
    # modshift figure show the same corrected flux under different labels,
    # which is misleading for visual inspection.
    raw_vals = std_df["flux_raw"].to_numpy(dtype=float)
    has_raw  = np.any(np.isfinite(raw_vals))

    if has_raw:
        finite_raw  = raw_vals[np.isfinite(raw_vals)]
        raw_median  = float(np.nanmedian(finite_raw))
        if np.isfinite(raw_median) and raw_median != 0.0:
            sap_norm = raw_vals / raw_median
        else:
            # Degenerate raw median; fall back to flux_corr for SAP panel.
            sap_norm = corr_norm
    else:
        # flux_raw is all-NaN for this pipeline; use corr_norm as SAP.
        # This is cosmetic only -- modshift metrics are computed from
        # Detrended Flux / Detrended Error, not SAP Flux.
        sap_norm = corr_norm

    lc_nms = pd.DataFrame({
        "Time":            std_df["time"].to_numpy(dtype=float),
        "SAP Flux":        sap_norm,
        "Detrended Flux":  corr_norm,
        "Detrended Error": err_norm,
    }, index=std_df.index)

    # Drop cadences where the essential columns are non-finite.
    # Detrended Error must also be positive; zero/NaN errors would cause
    # division-by-zero in LEO-Vetter's weighted statistics.
    valid = (
        np.isfinite(lc_nms["Time"])
        & np.isfinite(lc_nms["Detrended Flux"])
        & np.isfinite(lc_nms["Detrended Error"])
        & (lc_nms["Detrended Error"] > 0)
    )
    return lc_nms.loc[valid].reset_index(drop=True)


def Apply_LEOVetter(
    ID: Union[int, str],
    target: Union[pd.DataFrame, pd.Series],
    sector: Union[int, List[int]],
    LC_df: pd.DataFrame,
    pipeline: str,
    savepath: str,
    *,
    corrected_T0: Optional[float] = None,
    verbose: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Produce a LEO-Vetter modshift report for one pipeline's light curve.

    LC_df must have columns: Time, SAP Flux, Detrended Flux, Detrended Error.
    For HLSPs, pass the output of _standardized_to_nemesis_lc.
    For NEMESIS, pass the raw LC_df from full_pipeline directly.

    Parameters
    ----------
    ID : int or str
        TIC identifier.
    target : pd.Series or pd.DataFrame
        Single target row with orbital parameters and stellar parameters.
    sector : int or list of int
        Sector(s) used for the output figure filename.
    LC_df : pd.DataFrame
        NEMESIS-style light curve.
    pipeline : str
        Pipeline name used in the output figure filename.
    savepath : str
        Directory for the LEO-Vetter report PNG.
    corrected_T0 : float or None
        Per-pipeline corrected epoch (BTJD). Falls back to catalog value.
    verbose : bool

    Returns
    -------
    metrics : pd.DataFrame or None
        Single-row LEO-Vetter modshift metrics, or None on failure.
    """
    if isinstance(target, pd.Series):
        star = target.to_frame().T.reset_index(drop=True)
    else:
        star = target.copy().reset_index(drop=True)

    P        = float(star["Orbital Period (days) Value"].iloc[0])
    T0       = float(corrected_T0 if corrected_T0 is not None
                     else star["Orbital Epoch Value"].iloc[0])
    Dur      = float(star["Transit Duration (hours) Value"].iloc[0])
    dur_days = (Dur * u.hour).to(u.day).value

    if isinstance(sector, (list, tuple)):
        s_sorted   = sorted(int(s) for s in sector)
        sector_tag = (f"S{s_sorted[0]:04d}" if len(s_sorted) == 1
                      else f"S{s_sorted[0]:04d}to{s_sorted[-1]:04d}")
    else:
        sector_tag = f"S{int(sector):04d}"

    os.makedirs(savepath, exist_ok=True)
    savefilename = os.path.join(
        savepath, f"TIC_{ID}_{sector_tag}_{pipeline}_LEOVetter_Report.png"
    )

    try:
        metrics = plot_modshift_NEMESIS(
            ID=ID, star=star, LC_df=LC_df,
            period=P, dur=dur_days, T0=T0,
            save_file=savefilename, verbose=verbose,
        )
        if verbose:
            print(f"[LEO-Vetter] {pipeline}: report saved -> {savefilename}")
        return metrics
    except Exception as exc:
        print(f"[LEO-Vetter] {pipeline} failed: {exc}")
        return None


# ===========================================================================
# Section 4: combined_results builder helpers
# ===========================================================================

def _build_nemesis_combined_entry(
    tic_id: int,
    nemesis_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Wrap run_nemesis_and_standardize output into the combined_results entry
    structure used by target_to_lightcurve_workflow_V5.

    Note: 'sectors' for NEMESIS contains 0-based inferred sector indices
    from gap-splitting, not true TESS sector numbers.

    Parameters
    ----------
    tic_id : int
    nemesis_result : dict
        Output of run_nemesis_and_standardize.

    Returns
    -------
    dict
    """
    if nemesis_result["status"] != "ok":
        return {
            "product": None, "raw": None,
            "standardized": None, "standardized_masked": None,
            "tic_id": tic_id, "sector": [], "sectors": [],
            "pipeline": "NEMESIS",
            "n_raw": 0, "n_standardized": 0, "n_masked": 0,
            "status": "error",
            "error": nemesis_result.get("error", "run_nemesis_and_standardize failed"),
            "leo_vetter_metrics": None, "leo_vetter_vetted": None,
        }

    return {
        "product": None,
        "raw":                 nemesis_result["raw"],
        "standardized":        nemesis_result["standardized"],
        "standardized_masked": nemesis_result["standardized_masked"],
        "tic_id":  tic_id,
        "sector":  nemesis_result["sectors"],
        "sectors": nemesis_result["sectors"],
        "pipeline": "NEMESIS",
        "n_raw":          nemesis_result["n_standardized"],
        "n_standardized": nemesis_result["n_standardized"],
        "n_masked":       nemesis_result["n_masked"],
        "status": "ok",
        "error": None,
        "leo_vetter_metrics": None,
        "leo_vetter_vetted":  None,
    }


def compile_leovetter_catalog(
    combined_results: Dict[str, Any],
    target: Union[pd.Series, pd.DataFrame],
) -> pd.DataFrame:
    """
    Stack leo_vetter_vetted DataFrames from all pipelines into a single
    long-format DataFrame with a 'pipeline' column.

    Each row corresponds to one pipeline. Shared columns (target metadata)
    repeat across rows; LEOVetter_* columns vary per pipeline.

    Parameters
    ----------
    combined_results : dict
        Output of target_to_lightcurve_workflow_V5.
    target : pd.Series or pd.DataFrame
        The target row (metadata only; not used for computation).

    Returns
    -------
    pd.DataFrame
        One row per pipeline that produced LEO-Vetter metrics.
        Empty DataFrame if no pipeline succeeded.

    Example
    -------
    >>> results = target_to_lightcurve_workflow_V5(target=target, ...)
    >>> cat = compile_leovetter_catalog(results, target)
    >>> cat[["pipeline"] + [c for c in cat.columns if "LEOVetter" in c]]
    """
    rows = []
    for pipeline, info in combined_results.items():
        vetted = info.get("leo_vetter_vetted")
        if vetted is None or vetted.empty:
            continue
        df = vetted.copy()
        df.insert(0, "pipeline", pipeline)
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)



def refine_t0_from_lc_OLD(
    time: np.ndarray,
    flux: np.ndarray,
    T0_catalog: float,
    period: float,
    duration_days: float,
    search_width_factor: float = 2.0,
    min_cadences_in_transit: int = 3,
    n_bins: int = 5,
    min_depth_fraction: float = 0.003,
    require_smooth_minimum: bool = True,
) -> tuple[float, float]:
    """
    Refine a catalog transit epoch by finding the flux minimum in a narrow
    phase window around the catalog T0.

    This is a fast alternative to BLS/TLS for targets where the catalog T0
    has accumulated drift (common for TOIs with few observed transits at the
    time of the catalog entry). It phase-folds the light curve at the catalog
    period, bins into a narrow window around phase=0, and returns the time
    of the median flux minimum as the corrected T0.

    The correction is intentionally conservative: it only adjusts T0 within
    a window of +/- search_width_factor * duration around the catalog epoch.
    If no flux minimum is detected within the window (e.g. no transit falls
    in this sector), the catalog T0 is returned unchanged.

    This function does NOT search for a better period. It assumes the catalog
    period is correct and that the epoch has drifted. If the transit is
    missing entirely from the folded LC (no data at the expected phase), the
    catalog T0 is returned and a flag is set.

    Parameters
    ----------
    time : np.ndarray
        Cadence timestamps (BTJD). Must be finite.
    flux : np.ndarray
        Detrended, normalized flux (median ~1.0). Must be finite.
    T0_catalog : float
        Catalog transit epoch (BTJD).
    period : float
        Orbital period in days.
    duration_days : float
        Transit duration in days (ingress to egress).
    search_width_factor : float
        Half-width of the search window as a multiple of duration_days.
        Default 2.0 means the search spans +/- 2 * duration around T0.
        Increase for targets with large T0 uncertainty; decrease for
        short-period targets where a neighboring transit is nearby.
    min_cadences_in_transit : int
        Minimum number of cadences required in the search window to attempt
        refinement. If fewer cadences are present, catalog T0 is returned.
        Default 3.
    n_bins : int
        Number of phase bins across the search window. Fewer bins means
        each bin contains more cadences, making the minimum more robust
        against individual noisy cadences. Default 5. Decrease to 3 for
        high-scatter pipelines (TGLC, GSFC-ELEANOR-LITE).
    min_depth_fraction : float
        Minimum required depth of the detected minimum relative to the
        out-of-transit baseline, as a flux fraction. If the minimum bin
        is shallower than this, the result is noise and catalog T0 is
        returned. Default 0.003 (0.3%). Decrease for very shallow transits.
    require_smooth_minimum : bool
        If True, require that at least one bin adjacent to the minimum
        is also below the out-of-transit median. This rejects isolated
        noise spikes that happen to be the deepest bin. Default True.

    Returns
    -------
    T0_refined : float
        Refined epoch in BTJD. Equal to T0_catalog if refinement failed.
    T0_offset_hours : float
        T0_refined - T0_catalog in hours. Zero if refinement failed.

    Notes
    -----
    The refined T0 is expressed as the nearest transit epoch to the
    midpoint of the observed time baseline, consistent with how
    corrected_epochs are computed elsewhere in the V5 workflow.

    The function is intentionally simple -- it uses a binned median rather
    than a Gaussian or trapezoid fit to avoid overfitting on noisy or
    sparsely sampled transit windows. For well-sampled transits (N_transit
    >> 10), the binned minimum is a robust estimator of the transit center.

    Example
    -------
    >>> T0_ref, offset_hrs = refine_t0_from_lc(
    ...     time=lc["Time"].values,
    ...     flux=lc["Detrended Flux"].values,
    ...     T0_catalog=1492.3,
    ...     period=3.14,
    ...     duration_days=0.08,
    ... )
    >>> print(f"T0 offset: {offset_hrs:.2f} hours")
    """
    # Phase-fold to (-0.5P, 0.5P) centered on T0
    phase_days = ((time - T0_catalog + period / 2.0) % period) - period / 2.0

    # Search window: +/- search_width_factor * duration
    half_window = duration_days * search_width_factor
    in_window   = np.abs(phase_days) <= half_window

    if in_window.sum() < min_cadences_in_transit:
        # Not enough cadences in window -- return catalog value unchanged
        return T0_catalog, 0.0

    phase_window = phase_days[in_window]
    flux_window  = flux[in_window]

    # Remove non-finite flux values inside window
    finite = np.isfinite(flux_window)
    if finite.sum() < min_cadences_in_transit:
        return T0_catalog, 0.0

    phase_window = phase_window[finite]
    flux_window  = flux_window[finite]

    # Bin the window into n_bins phase bins and find the minimum-flux bin.
    # Using a fixed small number of bins (default 5) ensures each bin spans
    # multiple cadences, making the median robust against individual noise
    # spikes. The previous dynamic n_bins = max(5, in_window.sum()//2) could
    # produce bins with only 1-2 cadences for typical transit windows, which
    # is too sensitive to per-cadence scatter in high-noise pipelines.
    edges   = np.linspace(phase_window.min(), phase_window.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(phase_window, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_medians = np.array([
        np.median(flux_window[bin_idx == i]) if (bin_idx == i).sum() > 0
        else np.nan
        for i in range(n_bins)
    ])

    valid_bins = np.isfinite(bin_medians)
    if valid_bins.sum() == 0:
        return T0_catalog, 0.0

    # Estimate out-of-transit baseline as the median of all bins excluding
    # the minimum. Used for depth and smoothness checks below.
    min_bin      = int(np.nanargmin(bin_medians))
    other_bins   = np.array([bin_medians[i] for i in range(n_bins)
                              if i != min_bin and np.isfinite(bin_medians[i])])
    baseline     = float(np.median(other_bins)) if other_bins.size > 0 else 1.0

    # Depth check: if the minimum is shallower than min_depth_fraction below
    # baseline, the dip is consistent with noise. Return catalog T0 unchanged.
    depth = baseline - bin_medians[min_bin]
    if depth < min_depth_fraction * baseline:
        return T0_catalog, 0.0

    # Smoothness check: a genuine transit produces a dip that spans multiple
    # bins, not just one. Require at least one neighbor of the minimum to
    # also be below the baseline. If the minimum is isolated, it is likely
    # a noise spike and catalog T0 is returned.
    if require_smooth_minimum:
        left_ok  = (min_bin > 0) and np.isfinite(bin_medians[min_bin - 1]) and (bin_medians[min_bin - 1] < baseline)
        right_ok = (min_bin < n_bins - 1) and np.isfinite(bin_medians[min_bin + 1]) and (bin_medians[min_bin + 1] < baseline)
        if not (left_ok or right_ok):
            return T0_catalog, 0.0

    # Phase offset of the minimum-flux bin center
    phase_offset  = float(centers[min_bin])   # days

    # Sanity check: if the offset is larger than the search window, something
    # is wrong (e.g. a flare or gap artifact is the minimum). Return unchanged.
    if np.abs(phase_offset) > half_window:
        return T0_catalog, 0.0

    # Convert phase offset to a corrected epoch at the nearest transit to the
    # midpoint of the observed time baseline, consistent with corrected_epochs.
    t_mid   = 0.5 * (time.min() + time.max())
    n_mid   = round((t_mid - T0_catalog) / period)
    T0_ref  = T0_catalog + n_mid * period + phase_offset

    offset_hours = phase_offset * 24.0
    return float(T0_ref), float(offset_hours)


import numpy as np
from typing import Any, Dict, Optional, Tuple, Union


def refine_t0_from_lc(
    time: np.ndarray,
    flux: np.ndarray,
    T0_catalog: float,
    period: float,
    duration_days: float,
    search_width_factor: float = 8.0,
    max_search_days: Optional[float] = None,
    min_cadences_in_transit: int = 3,
    n_bins: int = 11,
    min_depth_fraction: float = 0.001,
    require_smooth_minimum: bool = False,
    parabolic_refine: bool = True,
    return_diagnostics: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
    """
    Refine a catalog transit epoch by locating the flux minimum in a phase
    window centered on the catalog T0.

    Drop-in replacement for the previous refine_t0_from_lc. Behavioral
    changes vs the prior version are listed under "Changes" below.

    Algorithm
    ---------
    1. Phase-fold the LC to (-P/2, +P/2) centered on T0_catalog.
    2. Restrict to a half-window of
       min(search_width_factor * duration_days, max_search_days, 0.4 * P).
    3. Bin into n_bins phase bins centered on phase=0.
    4. Take per-bin median; reject if (baseline - min_bin) < min_depth_fraction.
    5. Optionally require a neighbor bin below baseline.
    6. Optionally parabolically interpolate the min and its two neighbors
       for sub-bin precision.
    7. Re-anchor T0_refined to the nearest transit epoch to the baseline
       midpoint (consistent with corrected_epochs elsewhere in V5).

    Parameters
    ----------
    time : np.ndarray
        Cadence timestamps (BTJD).
    flux : np.ndarray
        Detrended, normalized flux (out-of-transit median ~1.0).
    T0_catalog : float
        Catalog transit epoch (BTJD).
    period : float
        Orbital period in days.
    duration_days : float
        Transit duration in days.
    search_width_factor : float, default 8.0
        Half-width of the search window in units of duration_days. The
        previous default (2.0) was too narrow for TOIs with multi-hour T0
        drift; 8.0 covers approximately +/- 8 h for a 1 h transit while
        staying well within +/- 0.5 P for typical short-period planets.
    max_search_days : float or None, default None
        Optional hard cap on the half-window in days. If None, only the
        0.4 * period safety cap applies.
    min_cadences_in_transit : int, default 3
        Minimum cadences required in the search window.
    n_bins : int, default 11
        Number of phase bins across the (full) search window. Bins are
        centered on phase=0 so the recovered offset is symmetric.
    min_depth_fraction : float, default 0.001
        Minimum depth (relative to off-minimum baseline) below which the
        candidate is treated as noise. The prior 0.003 was too strict for
        shallow M-dwarf transits with ~0.1-0.2 % MAD scatter.
    require_smooth_minimum : bool, default False
        If True, require a neighboring bin below baseline. Prior default
        (True) rejected single-bin transits, which can occur for short
        durations with coarse binning.
    parabolic_refine : bool, default True
        If True, parabolically interpolate the bin minimum for sub-bin
        precision. Falls back to bin center on degenerate fits.
    return_diagnostics : bool, default False
        If True, return a third element with status and intermediate
        quantities for auditing per-pipeline behavior.

    Returns
    -------
    T0_refined : float
        Refined epoch (BTJD), re-anchored to the nearest transit epoch to
        the baseline midpoint. Equal to T0_catalog if refinement failed.
    T0_offset_hours : float
        In-period offset (T0_refined - T0_catalog mod P, signed) in hours.
        Zero if refinement failed.
    diagnostics : dict, optional
        Returned only if return_diagnostics is True. Keys:
        status, depth, baseline, half_window_days, n_in_window, n_bins,
        offset_days_pre_refine.

    Changes vs prior version
    ------------------------
    * search_width_factor default 2.0 -> 8.0
    * n_bins default 5 -> 11
    * min_depth_fraction default 0.003 -> 0.001
    * require_smooth_minimum default True -> False
    * Bins are now centered on phase=0 (was anchored to data extrema in
      the search window)
    * Added max_search_days kwarg and a 0.4*P safety cap
    * Added parabolic_refine kwarg (sub-bin interpolation)
    * Added return_diagnostics kwarg
    * Refined T0 is now actually re-anchored to the nearest transit epoch
      (the prior docstring claimed this; the prior code did not do it)

    Example
    -------
    >>> T0_ref, dt_hr, diag = refine_t0_from_lc(
    ...     time=lc["Time"].to_numpy(dtype=float),
    ...     flux=lc["Detrended Flux"].to_numpy(dtype=float),
    ...     T0_catalog=1492.30,
    ...     period=3.14,
    ...     duration_days=1.0 / 24.0,
    ...     return_diagnostics=True,
    ... )
    >>> print(f"dT0 = {dt_hr:+.2f} h, status = {diag['status']}")
    """
    def _ret(t0: float, dt_hr: float, status: str, **extra: Any):
        if return_diagnostics:
            return t0, dt_hr, {"status": status, **extra}
        return t0, dt_hr

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    finite_lc = np.isfinite(time) & np.isfinite(flux)
    if int(finite_lc.sum()) < min_cadences_in_transit:
        return _ret(T0_catalog, 0.0, "insufficient_finite_lc",
                    n_finite=int(finite_lc.sum()))

    time = time[finite_lc]
    flux = flux[finite_lc]

    # Phase-fold to (-P/2, +P/2) centered on catalog T0
    phase_days = ((time - T0_catalog + 0.5 * period) % period) - 0.5 * period

    # Effective half-window with hard caps
    half_window = duration_days * search_width_factor
    if max_search_days is not None:
        half_window = min(half_window, float(max_search_days))
    half_window = min(half_window, 0.4 * period)
    if half_window <= 0:
        return _ret(T0_catalog, 0.0, "invalid_window",
                    half_window_days=float(half_window))

    in_window = np.abs(phase_days) <= half_window
    n_in = int(in_window.sum())
    if n_in < min_cadences_in_transit:
        return _ret(T0_catalog, 0.0, "insufficient_cadences",
                    n_in_window=n_in,
                    half_window_days=float(half_window))

    phase_w = phase_days[in_window]
    flux_w = flux[in_window]

    # Bins centered on phase=0
    edges = np.linspace(-half_window, +half_window, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(phase_w, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_medians = np.full(n_bins, np.nan, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        sel = (bin_idx == i)
        bin_counts[i] = int(sel.sum())
        if bin_counts[i] > 0:
            bin_medians[i] = float(np.median(flux_w[sel]))

    if not np.any(np.isfinite(bin_medians)):
        return _ret(T0_catalog, 0.0, "no_valid_bins",
                    half_window_days=float(half_window))

    i_min = int(np.nanargmin(bin_medians))

    # Baseline: median of bins excluding min and its immediate neighbors
    excl = np.zeros(n_bins, dtype=bool)
    excl[max(0, i_min - 1): min(n_bins, i_min + 2)] = True
    if (~excl & np.isfinite(bin_medians)).any():
        baseline = float(np.nanmedian(bin_medians[~excl]))
    else:
        baseline = float(np.nanmedian(bin_medians))

    depth = baseline - float(bin_medians[i_min])
    if not np.isfinite(depth) or depth < min_depth_fraction:
        return _ret(
            T0_catalog, 0.0, "insufficient_depth",
            depth=float(depth) if np.isfinite(depth) else None,
            baseline=baseline,
            half_window_days=float(half_window),
            n_in_window=n_in,
        )

    if require_smooth_minimum:
        below = False
        for j in (i_min - 1, i_min + 1):
            if 0 <= j < n_bins and np.isfinite(bin_medians[j]) \
                    and bin_medians[j] < baseline:
                below = True
                break
        if not below:
            return _ret(T0_catalog, 0.0, "isolated_minimum",
                        depth=float(depth), baseline=baseline,
                        half_window_days=float(half_window),
                        n_in_window=n_in)

    # Sub-bin parabolic refinement
    offset_days = float(centers[i_min])
    if parabolic_refine and 0 < i_min < n_bins - 1:
        y0 = bin_medians[i_min - 1]
        y1 = bin_medians[i_min]
        y2 = bin_medians[i_min + 1]
        if np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2):
            denom = (y0 - 2.0 * y1 + y2)
            if denom > 0:  # convex (true minimum, not saddle)
                dx = float(centers[1] - centers[0])
                shift = 0.5 * (y0 - y2) / denom * dx
                if abs(shift) <= dx:
                    offset_days = float(centers[i_min]) + shift

    pre_refine_offset = offset_days

    # Re-anchor refined T0 to the nearest transit epoch to the baseline midpoint
    t_mid = 0.5 * (time.min() + time.max())
    T0_local = T0_catalog + offset_days
    n_periods = int(round((t_mid - T0_local) / period))
    T0_refined = T0_local + n_periods * period
    T0_offset_hours = offset_days * 24.0

    return _ret(
        T0_refined,
        T0_offset_hours,
        "ok",
        depth=float(depth),
        baseline=baseline,
        half_window_days=float(half_window),
        n_in_window=n_in,
        n_bins=int(n_bins),
        offset_days_pre_refine=float(pre_refine_offset),
    )

# ===========================================================================
# Section 5: Core V5 workflow
# ===========================================================================

def target_to_lightcurve_workflow_V5(
    target: Union[pd.Series, pd.DataFrame],
    pipelines: List[str],
    target_Sector: Union[int, None, Literal["all"], List[int]],
    DEFAULT_RADIUS: Any,
    DEFAULT_CADENCE: str,
    DEFAULT_DOWNLOADPATH: str,
    *,
    run_nemesis: bool = True,
    nemesis_settings: Optional[Dict[str, Any]] = None,
    nemesis_gap_threshold_days: float = GAP_THRESHOLD_DAYS,
    save_results: bool = True,
    results_savepath: Optional[str] = None,
    force_redownload: bool = False,
    run_leovetter: bool = True,
    leovetter_savepath: Optional[str] = None,
    refine_t0: bool = False,
    nemesis_from_cache: Optional[Dict[str, Any]] = None,  
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Download, standardize, concatenate, phase-fold, and vet TESS light curves
    across HLSPs and NEMESIS for a single target.

    NEMESIS runs first (before any HLSP downloads), is standardized, and
    injected into combined_results under the key "NEMESIS" with the same dict
    structure as every HLSP pipeline. The phase-fold loop, Apply_LEOVetter
    loop, and comparison figures all iterate over combined_results uniformly.

    For Apply_LEOVetter, HLSPs are converted via _standardized_to_nemesis_lc.
    NEMESIS passes its raw LC_df directly (already has the correct columns).

    Sector selection via target_Sector
    -----------------------------------
    int      Process that sector only.
    None     Earliest available sector (V4-compatible behavior).
    "all"    All available HLSP sectors, concatenated.
    list     Exactly those sectors, concatenated.

    Parameters
    ----------
    target : pd.Series or single-row pd.DataFrame
        Required columns: 'TIC ID', 'Orbital Period (days) Value',
        'Orbital Epoch Value', 'Transit Depth Value',
        'Transit Duration (hours) Value', plus stellar param columns
        for get_qld (Teff, logg, rad, mass, ID).
    pipelines : list of str
        HLSP pipeline names.
    target_Sector : int | None | "all" | list of int
    DEFAULT_RADIUS : astropy Quantity or float
    DEFAULT_CADENCE : str
    DEFAULT_DOWNLOADPATH : str
    run_nemesis : bool
        Run NEMESIS before HLSPs. Default True.
    nemesis_settings : dict or None
        Overrides for default_NEMESIS_pipeline_settings.
    nemesis_gap_threshold_days : float
        Gap threshold for NEMESIS sector boundary inference. Default 5.0.
    save_results : bool
        Persist per-sector HLSP Parquets. Default True.
    results_savepath : str or None
        Root for Parquet outputs. Defaults to DEFAULT_DOWNLOADPATH/saved_results.
    force_redownload : bool
        Skip all caches. Default False.
    run_leovetter : bool
        Run Apply_LEOVetter on all pipelines. Default True.
    leovetter_savepath : str or None
        Directory for LEO-Vetter PNGs.
    verbose : bool

    Returns
    -------
    combined_results : dict
        Pipeline-keyed dict including "NEMESIS" when run_nemesis=True.
        Each entry has: status, standardized, standardized_masked, sectors,
        n_raw, n_standardized, n_masked, pipeline,
        leo_vetter_metrics, leo_vetter_vetted.

    Example
    -------
    >>> results = target_to_lightcurve_workflow_V5(
    ...     target=nearby_TOI_MD_df.iloc[0],
    ...     pipelines=["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"],
    ...     target_Sector="all",
    ...     DEFAULT_RADIUS=DEFAULT_RADIUS,
    ...     DEFAULT_CADENCE=DEFAULT_CADENCE,
    ...     DEFAULT_DOWNLOADPATH=DEFAULT_DOWNLOADPATH,
    ... )
    >>> catalog = compile_leovetter_catalog(results, nearby_TOI_MD_df.iloc[0])
    >>> catalog[["pipeline"] + [c for c in catalog.columns if "LEOVetter" in c]]
    """
    t_start = clock.time()

    if isinstance(target, pd.DataFrame):
        target = target.iloc[0]

    ID         = target["TIC ID"].item()
    target_P   = target["Orbital Period (days) Value"].item()
    target_T0  = target["Orbital Epoch Value"].item()
    target_Dep = target["Transit Depth Value"].item() / 1e6

    if results_savepath is None:
        results_savepath = os.path.join(DEFAULT_DOWNLOADPATH, "saved_results")

    combined_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 1. NEMESIS                                                          #
    # ------------------------------------------------------------------ #
    
    if nemesis_from_cache is not None:
        # Cache injection path (used by force_releovetter=True).
        # The caller has already loaded the NEMESIS LC from disk via
        # _load_nemesis_from_cache. Inject it directly so the phase-fold
        # loop and LEO-Vetter loop both see it without re-running full_pipeline.
        combined_results["NEMESIS"] = nemesis_from_cache
        if verbose:
            st = nemesis_from_cache.get("status", "unknown")
            ns = nemesis_from_cache.get("n_standardized", 0)
            print(f"[V5] NEMESIS: loaded from cache, status={st}, cadences={ns}")

    elif run_nemesis:
        nemesis_outdir = os.path.join(results_savepath, "nemesis_output", f"TIC{ID}")
        os.makedirs(nemesis_outdir, exist_ok=True)
        if verbose:
            print(f"\n[V5] ── NEMESIS ──────────────────────────────────────────")
            print(f"[V5] TIC {ID}: running full_pipeline -> {nemesis_outdir}")

        try:
            nemesis_result = run_nemesis_and_standardize(
                tic_id=ID,
                nemesis_output_path=nemesis_outdir,
                cadence=DEFAULT_CADENCE,
                nemesis_settings=nemesis_settings,
                gap_threshold_days=nemesis_gap_threshold_days,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"[V5] NEMESIS raised: {exc}")
            nemesis_result = {
                "status": "error", "error": str(exc),
                "standardized": None, "standardized_masked": None,
                "sectors": [], "n_standardized": 0, "n_masked": 0,
                "raw": None, "nemesis_settings": {},
            }

        combined_results["NEMESIS"] = _build_nemesis_combined_entry(ID, nemesis_result)

        if verbose:
            st = combined_results["NEMESIS"]["status"]
            ns = combined_results["NEMESIS"]["n_standardized"]
            print(f"[V5] NEMESIS: status={st}, cadences={ns}, "
                  f"sectors(inferred)={combined_results['NEMESIS']['sectors']}")

    # ------------------------------------------------------------------ #
    # 2. Resolve HLSP sector list                                         #
    # ------------------------------------------------------------------ #
    if isinstance(target_Sector, list):
        sectors = sorted(int(s) for s in target_Sector)
    elif target_Sector == "all":
        sectors = get_available_sectors(
            tic_id=ID, exptime=DEFAULT_CADENCE,
            pipelines=pipelines, radius=DEFAULT_RADIUS, verbose=verbose,
        )
        if not sectors:
            print(f"[V5] TIC {ID}: no HLSP sectors found via MAST.")
            sectors = []
    elif target_Sector is None:
        _all = get_available_sectors(
            tic_id=ID, exptime=DEFAULT_CADENCE,
            pipelines=pipelines, radius=DEFAULT_RADIUS, verbose=verbose,
        )
        sectors = [_all[0]] if _all else [None]
    else:
        sectors = [int(target_Sector)]

    valid_sectors = [s for s in sectors if s is not None]
    if len(valid_sectors) == 0:
        sector_tag = "Sauto"
    elif len(valid_sectors) == 1:
        sector_tag = f"S{valid_sectors[0]:04d}"
    else:
        sector_tag = f"S{min(valid_sectors):04d}to{max(valid_sectors):04d}"

    if leovetter_savepath is None:
        leovetter_savepath = os.path.join(results_savepath, f"TIC{ID}_{sector_tag}")

    # ------------------------------------------------------------------ #
    # 3. Per-sector HLSP download / cache loop                            #
    # ------------------------------------------------------------------ #
    accum: Dict[str, Dict[str, Any]] = {
        p: {"std": [], "std_masked": [], "sectors_ok": [],
            "n_raw": 0, "n_std": 0, "n_masked": 0, "last_error": None}
        for p in pipelines
    }

    for sector in sectors:
        if verbose:
            print(f"\n[V5] ── Sector {sector} ──────────────────────────────────")

        sector_results = None
        if not force_redownload and sector is not None:
            try:
                sector_results = load_pipeline_results(
                    savepath=results_savepath, tic_id=ID, sector=sector,
                )
                if verbose:
                    print(f"[V5] Cache hit: TIC {ID}, sector {sector}")
            except FileNotFoundError:
                if verbose:
                    print(f"[V5] Cache miss: TIC {ID}, sector {sector} -- downloading.")

        if sector_results is None:
            try:
                sector_results = collect_lightcurves_for_target(
                    tic_id=ID, sector=sector, pipelines=pipelines,
                    downloadpath=DEFAULT_DOWNLOADPATH,
                    radius=DEFAULT_RADIUS, exptime=DEFAULT_CADENCE,
                    apply_quality_mask=True, verbose=verbose,
                )
            except Exception as exc:
                print(f"[V5] collect_lightcurves_for_target failed sector {sector}: {exc}")
                for p in pipelines:
                    accum[p]["last_error"] = str(exc)
                continue

            if save_results and sector is not None:
                try:
                    save_pipeline_results(
                        results=sector_results, savepath=results_savepath,
                        tic_id=ID, sector=sector, overwrite=True,
                    )
                except Exception as exc:
                    print(f"[V5] Warning: could not save sector {sector}: {exc}")

        for p in pipelines:
            info = sector_results.get(p, {})
            if info.get("status") != "ok":
                accum[p]["last_error"] = info.get("error", "unknown")
                continue
            df_std = info.get("standardized")
            if df_std is None or df_std.empty:
                continue
            df_msk = info.get("standardized_masked")
            
#             df_std = normalize_standardized_lc(df_std)
#             df_msk = _apply_quality_mask(df_std, pipeline=p)

            accum[p]["std"].append(df_std)
            accum[p]["n_raw"] += int(info.get("n_raw") or 0)
            accum[p]["n_std"] += int(info.get("n_standardized") or len(df_std))
            accum[p]["sectors_ok"].append(sector if sector is not None else -1)
            if df_msk is not None and not df_msk.empty:
                accum[p]["std_masked"].append(df_msk)
                accum[p]["n_masked"] += len(df_msk)

    # ------------------------------------------------------------------ #
    # 4. Concatenate HLSPs                                                #
    # ------------------------------------------------------------------ #
    for p in pipelines:
        a = accum[p]
        if not a["std"]:
            combined_results[p] = {
                "product": None, "raw": None,
                "standardized": None, "standardized_masked": None,
                "tic_id": ID, "sector": sectors, "sectors": [],
                "pipeline": p, "n_raw": 0, "n_standardized": 0, "n_masked": 0,
                "status": "error",
                "error": a["last_error"] or "no data across requested sectors",
                "leo_vetter_metrics": None, "leo_vetter_vetted": None,
            }
            continue

        std_concat = (
            pd.concat(a["std"], ignore_index=True)
            .sort_values("time").reset_index(drop=True)
        )
        msk_concat = None
        if a["std_masked"]:
            msk_concat = (
                pd.concat(a["std_masked"], ignore_index=True)
                .sort_values("time").reset_index(drop=True)
            )

        combined_results[p] = {
            "product": None, "raw": None,
            "standardized":        std_concat,
            "standardized_masked": msk_concat,
            "tic_id": ID, "sector": a["sectors_ok"], "sectors": a["sectors_ok"],
            "pipeline": p,
            "n_raw": a["n_raw"], "n_standardized": a["n_std"], "n_masked": a["n_masked"],
            "status": "ok", "error": None,
            "leo_vetter_metrics": None, "leo_vetter_vetted": None,
        }

    if verbose:
        for p, info in combined_results.items():
            print(f"[V5] {p}: status={info['status']}, "
                  f"cadences={info['n_standardized']}, sectors={info['sectors']}")

    # ------------------------------------------------------------------ #
    # 5. Phase-fold loop (NEMESIS first, then HLSPs)                      #
    # ------------------------------------------------------------------ #
    # all_pipeline_keys = (["NEMESIS"] if run_nemesis else []) + list(pipelines)
    # new
    all_pipeline_keys = (
        ["NEMESIS"] if (run_nemesis or nemesis_from_cache is not None) else []
    ) + list(pipelines)

    corrected_epochs:   Dict[str, Optional[float]]         = {}
    pipeline_data:      Dict[str, Optional[Dict[str, Any]]] = {}
    all_flux_unclipped: List[np.ndarray] = []
    all_flux_clipped:   List[np.ndarray] = []
        
    target_dur_d = float(target["Transit Duration (hours) Value"].item() / 24.0)
    


    for p in all_pipeline_keys:
        info = combined_results.get(p, {})
        if info.get("status") != "ok":
            corrected_epochs[p] = None
            pipeline_data[p]    = None
            continue
        
        # NEW
        sectors_used = info.get("sectors", valid_sectors)
        
        # ── Get the NEMESIS-schema LC for this pipeline ───────────────────
        if p == "NEMESIS":
            lc_nms = info.get("raw")
            if lc_nms is None or lc_nms.empty:
                if verbose:
                    print("[LEO-Vetter] NEMESIS: no raw LC_df -- skipping.")
                continue
        else:
            #lc_std = info.get("standardized_masked") or info.get("standardized")
            lc_std = info.get("standardized_masked")
            if lc_std is None or lc_std.empty:
                lc_std = info.get("standardized")
            if lc_std is None or lc_std.empty:
                if verbose:
                    print(f"[LEO-Vetter] {p}: no valid LC -- skipping.")
                continue
            lc_nms = _standardized_to_nemesis_lc(lc_std)
            
        # ── Refine T0 per pipeline ────────────────────────────────────────
        # The catalog T0 may have drifted relative to the actual transit
        # center observed in this sector baseline. This is common for TOIs
        # with few observed transits at catalog entry time. A local phase
        # minimum search within +/- 2 * duration of the catalog T0 gives a
        # fast, BLS-free correction that does not require a full period search.
        #
        # The refined T0 is used only for LEO-Vetter metric computation;
        # the catalog T0 is preserved everywhere else (phase-fold figure,
        # manifest, etc.).
        #
        # If no transit is detected in the search window (e.g. the sector
        # has no coverage at the expected phase), refine_t0_from_lc returns
        # the catalog T0 unchanged and offset_hrs = 0.
        if refine_t0:
            lv_T0, t0_offset_hrs, diag = refine_t0_from_lc(
                time          = lc_nms["Time"].to_numpy(dtype=float),
                flux          = lc_nms["Detrended Flux"].to_numpy(dtype=float),
                T0_catalog    = target_T0,
                period        = target_P,
                duration_days = target_dur_d,
                return_diagnostics=True,
            )
            print(f"offset = {t0_offset_hrs:+.2f} h, status = {diag['status']}, diag = {diag}")
        else:
            lv_T0, t0_offset_hrs = target_T0, 0.0
        
        if verbose and np.abs(t0_offset_hrs) > 0.5:
            print(
                f"[LEO-Vetter] {p} TIC {ID}: T0 refined by "
                f"{t0_offset_hrs:+.2f} h "
                f"(catalog={target_T0:.4f}, refined={lv_T0:.4f})"
            )

        # Store the offset so it can be examined post-hoc
        combined_results[p]["lv_T0_refined"]      = lv_T0
        combined_results[p]["lv_T0_offset_hours"] = t0_offset_hrs
        
        #NEW

        lc_to_plot = info["standardized_masked"]
        if lc_to_plot is None or lc_to_plot.empty:
            lc_to_plot = info["standardized"]
        if lc_to_plot is None or lc_to_plot.empty:
            corrected_epochs[p] = None
            pipeline_data[p]    = None
            continue

        flux    = lc_to_plot["flux_corr"].to_numpy()
        t_min   = lc_to_plot["time"].min()
        n_first = int(np.ceil((t_min - target_T0) / target_P))
#         corrected_epochs[p] = target_T0 + n_first * target_P


#         if lc_nms is not None and not lc_nms.empty:
#             if refine_t0:
#                 lv_T0, t0_offset_hrs, diag = refine_t0_from_lc(
#                 time          = lc_nms["Time"].to_numpy(dtype=float),
#                 flux          = lc_nms["Detrended Flux"].to_numpy(dtype=float),
#                 T0_catalog    = target_T0,
#                 period        = target_P,
#                 duration_days = target_dur_d,
#                 return_diagnostics=True,
#                 )
#                 print(f"offset = {t0_offset_hrs:+.2f} h, status = {diag['status']}, diag = {diag}")
#             else:
#                 lv_T0, t0_offset_hrs = target_T0, 0.0
#         else:
#             lv_T0, t0_offset_hrs = target_T0, 0.0
        
        combined_results[p]["lv_T0_refined"]      = lv_T0
        combined_results[p]["lv_T0_offset_hours"] = t0_offset_hrs
        
        corrected_epochs[p] = lv_T0

        flux_norm = flux / np.nanmedian(flux)
        clipped   = sigma_clip(flux_norm, sigma=3, sigma_lower=7, sigma_upper=3, maxiters=5)
        good      = ~clipped.mask
        lc_clean  = lc_to_plot.iloc[good].reset_index(drop=True)

        if verbose:
            print(f"[V5] {p}: {len(flux_norm)} cadences, "
                  f"{np.sum(clipped.mask)} removed by sigma-clip")

        pipeline_data[p] = {"lc": lc_to_plot, "lc_clean": lc_clean, "T0": corrected_epochs[p]}
        all_flux_unclipped.append(flux_norm)
        all_flux_clipped.append(flux_norm[good])

    # ------------------------------------------------------------------ #
    # 6. Phase-fold comparison figure                                      #
    # ------------------------------------------------------------------ #
    from astropy.stats import mad_std as _mad_std

    if all_flux_clipped:
        af     = np.concatenate(all_flux_clipped)
        finite = af[np.isfinite(af)]
        scatter = _mad_std(finite) if finite.size > 0 else 0.01
    else:
        scatter = 0.01

    y_min = (1.0 - target_Dep) - 10.0 * scatter
    y_max =  1.0               + 10.0 * scatter

#     hlsp_colors = _get_colors(len(pipelines))
#     color_map: Dict[str, Any] = {p: hlsp_colors[i] for i, p in enumerate(pipelines)}
#     color_map["NEMESIS"] = "black"
    
    color_map: dict[str, str] = {p: get_pipeline_color(p) for p in pipelines}
    # NEMESIS is already in _PIPELINE_COLORS as black; explicit override kept for
    # clarity but is now redundant:
    color_map["NEMESIS"] = get_pipeline_color("NEMESIS")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for p in all_pipeline_keys:
        if pipeline_data.get(p) is None:
            continue
        pdata = pipeline_data[p]
        C     = color_map.get(p, "grey")
        plot_phasefolded(ax=axes[0], target=target, lc=pdata["lc"],
                         color=C, label=p, T0=pdata["T0"])
        plot_phasefolded(ax=axes[1], target=target, lc=pdata["lc_clean"],
                         color=C, label=p, T0=pdata["T0"])

    if all_flux_unclipped:
        axes[0].set_ylim(y_min, y_max)
    if all_flux_clipped:
        axes[1].set_ylim(y_min, y_max)
    for ax in axes:
        ax.set_xlabel("Orbital Phase [Hours from transit center]")
        ax.set_ylabel("Normalized Relative Flux")
        ax.legend(loc="best", fontsize=9)
    axes[0].set_title(f"Unclipped: TIC {ID}, {sector_tag}")
    axes[1].set_title(f"Sigma-clipped: TIC {ID}, {sector_tag}")
    plt.tight_layout()

    os.makedirs(leovetter_savepath, exist_ok=True)
    phasefold_figpath = os.path.join(
        leovetter_savepath, f"TIC_{ID}_{sector_tag}_phasefold_comparison.png"
    )
    fig.savefig(phasefold_figpath, bbox_inches="tight", dpi=150)
    if verbose:
        print(f"[V5] Phase-fold figure saved -> {phasefold_figpath}")
    plt.show()

    # ------------------------------------------------------------------ #
    # 7. LEO-Vetter on all pipelines                                       #
    # ------------------------------------------------------------------ #
    if run_leovetter:
        target_df = (
            target.to_frame().T.reset_index(drop=True)
            if isinstance(target, pd.Series)
            else target.reset_index(drop=True)
        )

        for p in all_pipeline_keys:
            info = combined_results.get(p, {})
            if info.get("status") != "ok":
                continue
                
            sectors_used = info.get("sectors") or valid_sectors
            #lv_T0 = target_T0
            #NEW
            lv_T0 = combined_results[p].get("lv_T0_refined", target_T0)

            if p == "NEMESIS":
                lc_nms = info.get("raw")
                if lc_nms is None or lc_nms.empty:
                    if verbose:
                        print("[LEO-Vetter] NEMESIS: no raw LC_df -- skipping.")
                    continue
            else:
                lc_std = info.get("standardized_masked")
                if lc_std is None or lc_std.empty:
                    lc_std = info.get("standardized")
                if lc_std is None or lc_std.empty:
                    if verbose:
                        print(f"[LEO-Vetter] {p}: no valid LC -- skipping.")
                    continue
                lc_nms = _standardized_to_nemesis_lc(lc_std)

            if verbose:
                print(f"[LEO-Vetter] Running {p} for TIC {ID}, {sector_tag}")

            metrics = Apply_LEOVetter(
                ID=ID, target=target, sector=sectors_used,
                LC_df=lc_nms, pipeline=p, savepath=leovetter_savepath,
                corrected_T0=lv_T0, verbose=verbose,
            )

            if metrics is not None:
                combined_results[p]["leo_vetter_metrics"] = metrics
                combined_results[p]["leo_vetter_vetted"]  = pd.concat(
                    [target_df, metrics.add_prefix("LEOVetter_")], axis=1,
                ).reset_index(drop=True)

    if verbose:
        elapsed = clock.time() - t_start
        print(f"\n{'='*40}")
        print(f"target_to_lightcurve_workflow_V5 took: {elapsed:.1f} s")
        print(f"{'='*40}\n")

    return combined_results


# ===========================================================================
# Section 6: Persistence helpers
# ===========================================================================

def _persist_combined_results(
    combined_results: Dict[str, Any],
    outdir: str,
    logger: logging.Logger,
) -> None:
    """
    Save concat standardized LCs and LEO-Vetter metrics Parquets for all
    pipelines in combined_results (HLSPs + NEMESIS).

    Parameters
    ----------
    combined_results : dict
    outdir : str
    logger : logging.Logger
    """
    os.makedirs(outdir, exist_ok=True)
    for pipeline, info in combined_results.items():
        if info.get("status") != "ok":
            continue
        for key, suffix in [
            ("standardized",        "concat_standardized"),
            ("standardized_masked", "concat_standardized_masked"),
        ]:
            df = info.get(key)
            if df is not None and not df.empty:
                fpath = os.path.join(outdir, f"{pipeline}_{suffix}.parquet")
                df.to_parquet(fpath, index=False)
                logger.info(f"  saved {pipeline}_{suffix}.parquet ({len(df)} rows)")
        lv = info.get("leo_vetter_metrics")
        if lv is not None and not lv.empty:
            fpath = os.path.join(outdir, f"{pipeline}_leovetter_metrics.parquet")
            lv.to_parquet(fpath, index=False)
            logger.info(f"  saved {pipeline}_leovetter_metrics.parquet")


def _persist_leovetter_catalog(
    combined_results: Dict[str, Any],
    target: Union[pd.Series, pd.DataFrame],
    outdir: str,
    tic_id: Union[int, str],
    sector_tag: str,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """
    Build and save the stacked per-pipeline LEO-Vetter catalog Parquet.

    Output: {outdir}/TIC{tic_id}_{sector_tag}_leovetter_catalog.parquet

    Parameters
    ----------
    combined_results : dict
    target : pd.Series or pd.DataFrame
    outdir : str
    tic_id : int or str
    sector_tag : str
    logger : logging.Logger

    Returns
    -------
    pd.DataFrame or None
    """
    catalog = compile_leovetter_catalog(combined_results, target)
    if catalog.empty:
        logger.warning("  leovetter_catalog: no pipeline produced metrics, skipping.")
        return None
    fpath = os.path.join(outdir, f"TIC{tic_id}_{sector_tag}_leovetter_catalog.parquet")
    catalog.to_parquet(fpath, index=False)
    logger.info(
        f"  saved leovetter_catalog ({len(catalog)} rows, "
        f"pipelines={catalog['pipeline'].tolist()}) -> {fpath}"
    )
    return catalog


def _save_pipeline_comparison_figure(
    combined_results: Dict[str, Any],
    outdir: str,
    tic_id: Union[int, str],
    sector_tag: str,
    logger: logging.Logger,
) -> None:
    """
    Save a 2-panel raw vs. corrected flux comparison for all pipelines.

    HLSPs are plotted as colored dots (s=1). NEMESIS is plotted last in
    black (s=1, zorder=5) so it sits on top without visual clutter.

    Output: {outdir}/TIC{tic_id}_{sector_tag}_pipeline_comparison.png

    Parameters
    ----------
    combined_results : dict
    outdir : str
    tic_id : int or str
    sector_tag : str
    logger : logging.Logger
    """
    hlsp_keys   = [p for p in combined_results if p != "NEMESIS"]
#     hlsp_colors = _get_colors(len(hlsp_keys))
#     color_map: Dict[str, Any] = {p: hlsp_colors[i] for i, p in enumerate(hlsp_keys)}
#     color_map["NEMESIS"] = "black"
    
    
    color_map: dict[str, str] = {
    p: get_pipeline_color(p) for p in combined_results}

    ordered = hlsp_keys + (["NEMESIS"] if "NEMESIS" in combined_results else [])
    ok = [
        p for p in ordered
        if combined_results[p].get("status") == "ok"
        and combined_results[p].get("standardized_masked") is not None
        and not combined_results[p]["standardized_masked"].empty
    ]

    if not ok:
        logger.warning("  pipeline_comparison figure skipped: no masked LC data.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for p in ok:
        df         = combined_results[p]["standardized_masked"]
        c          = color_map.get(p, "grey")
        is_nemesis = (p == "NEMESIS")
        kw = dict(color=c, label=p, rasterized=True,
                  s=1, marker="o",
                  zorder=5 if is_nemesis else 2)
        ax1.scatter(df["time"], df["flux_raw"], **kw)
        norm = df["flux_corr"] / np.nanmedian(df["flux_corr"])
        ax2.scatter(df["time"], norm, **kw)

    ax1.set_title(f"TIC {tic_id}  {sector_tag} -- Raw flux", fontsize=10)
    ax2.set_title("Corrected flux (normalized per pipeline)", fontsize=10)
    ax2.set_xlabel("Time (BTJD)")
    for ax in (ax1, ax2):
        ax.set_ylabel("Flux")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0),
                  markerscale=5, fontsize=8)

    fig.tight_layout(pad=1)
    os.makedirs(outdir, exist_ok=True)
    figpath = os.path.join(outdir, f"TIC{tic_id}_{sector_tag}_pipeline_comparison.png")
    fig.savefig(figpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"  saved pipeline_comparison figure -> {figpath}")


# ===========================================================================
# Section 7: Manifest and logger helpers
# ===========================================================================

def _load_manifest(manifest_path: str = MANIFEST_PATH) -> pd.DataFrame:
    if os.path.isfile(manifest_path):
        return pd.read_csv(manifest_path, dtype=str)
    return pd.DataFrame(columns=MANIFEST_COLS)


def _upsert_manifest(
    row: Dict[str, Any],
    manifest_path: str = MANIFEST_PATH,
) -> None:
    df      = _load_manifest(manifest_path)
    tic_str = str(row["tic_id"])
    mask    = df["tic_id"].astype(str) == tic_str
    new_row = pd.DataFrame([{c: str(row.get(c, "")) for c in MANIFEST_COLS}])
    if mask.any():
        df = df[~mask]
    df = pd.concat([df, new_row], ignore_index=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df.to_csv(manifest_path, index=False)


def _make_target_logger(
    tic_id: Union[int, str],
    results_root: str = RESULTS_ROOT,
) -> logging.Logger:
    log_dir  = os.path.join(results_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"TIC{tic_id}.log")
    logger   = logging.getLogger(f"V5_TIC{tic_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger


def _close_target_logger(logger: logging.Logger) -> None:
    for h in logger.handlers:
        h.close()
    logger.handlers.clear()


# ===========================================================================
# Section 8: run_single_target
# ===========================================================================

def run_single_target(
    target: pd.Series,
    pipelines: List[str],
    results_root: str ,
    target_Sector: Union[int, None, Literal["all"], List[int]] = "all",
    run_nemesis: bool = True,
    nemesis_settings: Optional[Dict[str, Any]] = None,
    nemesis_gap_threshold_days: float = GAP_THRESHOLD_DAYS,
    run_leovetter: bool = True,
    force_rerun: bool = False,
    force_releovetter: bool = False,
    refine_t0: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run target_to_lightcurve_workflow_V5 for one target row, persist all
    outputs, and return a manifest-ready summary dict.

    The primary science output is:
        saved_results/TIC{ID}_{sector_tag}/
            TIC{ID}_{sector_tag}_leovetter_catalog.parquet

    Parameters
    ----------
    target : pd.Series
        One row of nearby_TOI_MD_df. Must contain 'TIC ID'.
    pipelines : list of str
        HLSP pipeline names.
    results_root : str
    target_Sector : int | None | "all" | list of int
        Default "all".
    run_nemesis : bool
    nemesis_settings : dict or None
    nemesis_gap_threshold_days : float
    run_leovetter : bool
    force_releovetter: bool
    force_rerun : bool
    verbose : bool

    Returns
    -------
    summary : dict
        Keys: tic_id, status, sectors_found, sector_tag,
              pipelines_ok, pipelines_failed, nemesis_status,
              nemesis_error, started_at, finished_at, error.

    Notes
    -----
    overall 'status' logic:
      "done"    -- at least one HLSP pipeline succeeded
      "partial" -- V5 returned results but all HLSP pipelines failed
      "failed"  -- V5 raised an exception or returned empty

    'nemesis_status' is tracked independently: ok | error | skipped.
    A NEMESIS failure does not affect overall 'status'.
    """
    tic_id = int(target["TIC ID"])
    logger = _make_target_logger(tic_id, results_root)

    summary: Dict[str, Any] = {
        "tic_id":           tic_id,
        "status":           "failed",
        "sectors_found":    "",
        "sector_tag":       "",
        "pipelines_ok":     "",
        "pipelines_failed": "",
        "nemesis_status":   "skipped",
        "nemesis_error":    "",
        "started_at":       datetime.utcnow().isoformat(timespec="seconds"),
        "finished_at":      "",
        "error":            "",
    }

    logger.info("=" * 60)
    logger.info(f"TIC {tic_id} -- starting")

    target           = pd.DataFrame([target]).iloc[0]
    results_savepath = os.path.join(results_root, "saved_results")
    
    
    # ── force_releovetter: pre-load NEMESIS from cache BEFORE calling V5 ────
    # This is the key fix: by loading the cache and passing it to V5 via
    # nemesis_from_cache, the NEMESIS LC is present in combined_results when
    # the phase-fold loop runs inside V5, so it appears in the comparison figure.
    #
    # The sector_tag needed to locate the cache directory is not yet known
    # (it depends on which HLSP sectors MAST returns). To resolve this, we
    # do a lightweight MAST sector query here -- it is fast (no download)
    # and uses the same logic V5 would use internally.
    nemesis_cache_entry: Optional[Dict[str, Any]] = None
    if force_releovetter and run_nemesis:
        target_P_tmp  = float(target["Orbital Period (days) Value"])
        target_ID_tmp = int(target["TIC ID"])

        # Resolve sector tag without downloading: query available sectors.
#         try:
#             _avail = get_available_sectors(
#                 tic_id=target_ID_tmp, exptime=DEFAULT_CADENCE,
#                 pipelines=pipelines, radius=DEFAULT_RADIUS, verbose=False,
#             )
#         except Exception:
#             _avail = []

#         if _avail:
#             _stag = (
#                 f"S{_avail[0]:04d}" if len(_avail) == 1
#                 else f"S{min(_avail):04d}to{max(_avail):04d}"
#             )
#         else:
#             _stag = "Sauto"
        # NEW
        _stag = find_cached_sector_tag(tic_id, results_savepath)
        if _stag is None:
            logger.warning(
                f"  force_releovetter: no cached output directory found for "
                f"TIC {tic_id} in {results_savepath}. "
                f"NEMESIS will be absent. Run without force_releovetter first.")
        #NEW

        nemesis_cache_entry = _load_nemesis_from_cache(
            tic_id=tic_id,
            results_savepath=results_savepath,
            sector_tag=_stag,
            verbose=verbose,
        )
        if nemesis_cache_entry["status"] == "ok":
            logger.info(
                f"  NEMESIS cache loaded: {nemesis_cache_entry['n_standardized']} "
                f"cadences from {_stag}"
            )
        else:
            logger.warning(
                f"  NEMESIS cache load failed ({_stag}): "
                f"{nemesis_cache_entry.get('error')}"
            )
            # Fall through with None -- NEMESIS will simply be absent from figure.
            nemesis_cache_entry = None    

    try:
        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            combined_results = target_to_lightcurve_workflow_V5(
                target=target,
                pipelines=pipelines,
                target_Sector=target_Sector,
                DEFAULT_RADIUS=DEFAULT_RADIUS,
                DEFAULT_CADENCE=DEFAULT_CADENCE,
                DEFAULT_DOWNLOADPATH=DEFAULT_DOWNLOADPATH,
                # When force_releovetter=True:
                #   - Skip full_pipeline (NEMESIS photometry). NEMESIS raw LC
                #     will be injected from cache below after V5 returns.
                #   - Never re-download HLSP data; always use parquet cache.
                #   - Do not overwrite existing photometry parquets.
                run_nemesis              = run_nemesis and not force_releovetter,
                nemesis_settings         = nemesis_settings,
                nemesis_gap_threshold_days = nemesis_gap_threshold_days,
                save_results             = not force_releovetter,
                results_savepath         = results_savepath,
                force_redownload         = False if force_releovetter else force_rerun,
                run_leovetter            = run_leovetter,
                refine_t0= refine_t0,
                
                nemesis_from_cache       = nemesis_cache_entry,
                verbose                  = verbose,
            )
        captured = stdout_buf.getvalue()
        if captured.strip():
            for line in captured.splitlines():
                logger.debug(f"[V5] {line}")
    except Exception as exc:
        logger.error(f"target_to_lightcurve_workflow_V5 raised:\n{traceback.format_exc()}")
        summary["error"] = str(exc)
        summary["finished_at"] = datetime.utcnow().isoformat(timespec="seconds")
        _close_target_logger(logger)
        return summary

    if not combined_results:
        msg = "V5 returned empty dict (no sectors found)"
        logger.warning(msg)
        summary["error"] = msg
        summary["finished_at"] = datetime.utcnow().isoformat(timespec="seconds")
        _close_target_logger(logger)
        return summary
    
    
    # ── force_releovetter: inject NEMESIS from cache and re-run LEO-Vetter ──
    # V5 was called with run_nemesis=False, so combined_results has no NEMESIS
    # entry. We now:
    #   1. Determine the sector_tag from the HLSP sectors V5 resolved.
    #   2. Load the cached NEMESIS standardized parquets from that directory.
    #   3. Reconstruct the raw LC_df that Apply_LEOVetter expects.
    #   4. Run refine_t0_from_lc and Apply_LEOVetter directly on the cached LC.
    if force_releovetter and run_nemesis:
        # Determine sector_tag from HLSP sectors already in combined_results.
        _sectors_tmp: List[int] = []
        for _p, _info in combined_results.items():
            _secs = _info.get("sectors") or []
            _sectors_tmp = sorted(
                set(_sectors_tmp + [s for s in _secs if s is not None])
            )
        sector_tag_tmp = (
            "Sunknown" if not _sectors_tmp
            else f"S{_sectors_tmp[0]:04d}" if len(_sectors_tmp) == 1
            else f"S{min(_sectors_tmp):04d}to{max(_sectors_tmp):04d}"
        )

        nemesis_entry = _load_nemesis_from_cache(
            tic_id=tic_id,
            results_savepath=results_savepath,
            sector_tag=sector_tag_tmp,
            verbose=verbose,
        )
        combined_results["NEMESIS"] = nemesis_entry

        if nemesis_entry["status"] == "ok":
            # Run T0 refinement and Apply_LEOVetter on the cached NEMESIS LC.
            target_P     = float(target["Orbital Period (days) Value"])
            target_T0    = float(target["Orbital Epoch Value"])
            target_dur_d = float(target["Transit Duration (hours) Value"]) / 24.0

            lc_nms = nemesis_entry["raw"]
            if lc_nms is not None and not lc_nms.empty:
                lv_T0, t0_offset = refine_t0_from_lc(
                    time          = lc_nms["Time"].to_numpy(dtype=float),
                    flux          = lc_nms["Detrended Flux"].to_numpy(dtype=float),
                    T0_catalog    = target_T0,
                    period        = target_P,
                    duration_days = target_dur_d,
                    search_width_factor = 2.0,
                )
                combined_results["NEMESIS"]["lv_T0_refined"]      = lv_T0
                combined_results["NEMESIS"]["lv_T0_offset_hours"] = t0_offset

                if verbose and abs(t0_offset) > 0.5:
                    print(
                        f"[releovetter] NEMESIS TIC {tic_id}: T0 refined by "
                        f"{t0_offset:+.2f} h "
                        f"(catalog={target_T0:.4f}, refined={lv_T0:.4f})"
                    )

                leovetter_savepath_nms = os.path.join(
                    results_savepath, f"TIC{tic_id}_{sector_tag_tmp}"
                )
                os.makedirs(leovetter_savepath_nms, exist_ok=True)

                target_df = target.to_frame().T.reset_index(drop=True)
                metrics = Apply_LEOVetter(
                    ID=tic_id,
                    target=target,
                    sector=nemesis_entry["sectors"] or [0],
                    LC_df=lc_nms,
                    pipeline="NEMESIS",
                    savepath=leovetter_savepath_nms,
                    corrected_T0=lv_T0,
                    verbose=verbose,
                )
                if metrics is not None:
                    combined_results["NEMESIS"]["leo_vetter_metrics"] = metrics
                    combined_results["NEMESIS"]["leo_vetter_vetted"]  = pd.concat(
                        [target_df, metrics.add_prefix("LEOVetter_")], axis=1,
                    ).reset_index(drop=True)
                    logger.info(f"  NEMESIS (cached): LEO-Vetter complete, T0 offset={t0_offset:+.2f}h")
                else:
                    logger.warning("  NEMESIS (cached): Apply_LEOVetter returned None")
            else:
                logger.warning("  NEMESIS (cached): raw LC_df is empty after loading")
        else:
            logger.warning(
                f"  NEMESIS cache load failed: {nemesis_entry.get('error')}"
            )    

    # Resolve sector tag from HLSP sectors only
    all_sectors: List[int] = []
    for p, info in combined_results.items():
        if p == "NEMESIS":
            continue
        secs = info.get("sectors") or []
        all_sectors = sorted(set(all_sectors + [s for s in secs if s is not None]))

    sector_tag = (
        "Sunknown"                                              if not all_sectors
        else f"S{all_sectors[0]:04d}"                           if len(all_sectors) == 1
        else f"S{min(all_sectors):04d}to{max(all_sectors):04d}"
    )
    outdir = os.path.join(results_savepath, f"TIC{tic_id}_{sector_tag}")
    
    # When force_releovetter, only overwrite the leovetter catalog and metrics;
    # do not overwrite the photometry parquets (save_results=False was passed
    # to V5 above, but _persist_combined_results would overwrite them here).
    # Guard by only persisting leovetter outputs.
    if force_releovetter:
        _persist_leovetter_catalog(
            combined_results=combined_results, target=target,
            outdir=outdir, tic_id=tic_id, sector_tag=sector_tag, logger=logger,
        )
        # Also persist updated LEO-Vetter metrics parquets for each pipeline.
        os.makedirs(outdir, exist_ok=True)
        for _p, _info in combined_results.items():
            _lv = _info.get("leo_vetter_metrics")
            if _lv is not None and not _lv.empty:
                _fpath = os.path.join(outdir, f"{_p}_leovetter_metrics.parquet")
                _lv.to_parquet(_fpath, index=False)
                logger.info(f"  [releovetter] saved {_p}_leovetter_metrics.parquet")
    else:
        _persist_combined_results(combined_results, outdir, logger)
        _persist_leovetter_catalog(
            combined_results=combined_results, target=target,
            outdir=outdir, tic_id=tic_id, sector_tag=sector_tag, logger=logger,
        )
        _save_pipeline_comparison_figure(
            combined_results=combined_results,
            outdir=outdir, tic_id=tic_id, sector_tag=sector_tag, logger=logger,
        )

    ok_pipelines     = [p for p, v in combined_results.items()
                        if v.get("status") == "ok" and p != "NEMESIS"]
    failed_pipelines = [p for p, v in combined_results.items()
                        if v.get("status") != "ok" and p != "NEMESIS"]

    nemesis_info   = combined_results.get("NEMESIS", {})
    nemesis_status = nemesis_info.get("status", "skipped") if run_nemesis else "skipped"
    nemesis_error  = nemesis_info.get("error") or ""

    for p in ok_pipelines:
        info = combined_results[p]
        logger.info(
            f"  {p}: ok | n_std={info.get('n_standardized', 0)} "
            f"n_masked={info.get('n_masked', 0)} sectors={info.get('sectors')}"
        )
    for p in failed_pipelines:
        logger.warning(f"  {p}: FAILED -- {combined_results[p].get('error', 'unknown')}")
    logger.info(f"  NEMESIS: {nemesis_status}")

    overall_status = "done" if ok_pipelines else "partial"
    summary.update({
        "status":           overall_status,
        "sectors_found":    json.dumps(all_sectors),
        "sector_tag":       sector_tag,
        "pipelines_ok":     json.dumps(ok_pipelines),
        "pipelines_failed": json.dumps(failed_pipelines),
        "nemesis_status":   nemesis_status,
        "nemesis_error":    nemesis_error,
        "finished_at":      datetime.utcnow().isoformat(timespec="seconds"),
        "error":            "",
    })

    logger.info(
        f"TIC {tic_id} -- {overall_status}  ({sector_tag}, "
        f"hlsp_ok={ok_pipelines}, nemesis={nemesis_status})"
    )
    _close_target_logger(logger)
    return summary


# ===========================================================================
# Section 9: run_batch
# ===========================================================================

def run_batch(
    catalog_df: pd.DataFrame,
    pipelines: List[str],
    results_root: str = RESULTS_ROOT,
    target_Sector: Union[int, None, Literal["all"], List[int]] = "all",
    run_nemesis: bool = True,
    nemesis_settings: Optional[Dict[str, Any]] = None,
    nemesis_gap_threshold_days: float = GAP_THRESHOLD_DAYS,
    run_leovetter: bool = True,
    force_rerun: bool = False,
    force_releovetter: bool = False,
    refine_t0: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run run_single_target for every row in catalog_df.

    Skips targets already marked 'done' in the manifest unless force_rerun=True.
    Writes the manifest after every target so the run is fully resumable.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Target list. Must contain 'TIC ID'.
    pipelines : list of str
        HLSP pipeline names.
    results_root : str
    target_Sector : int | None | "all" | list of int
        Applied uniformly to all targets. Default "all".
    run_nemesis : bool
    nemesis_settings : dict or None
        Applied uniformly to all targets.
    nemesis_gap_threshold_days : float
    run_leovetter : bool
    force_rerun : bool
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Final state of run_manifest.csv.

    Example
    -------
    >>> from nemesis_hlsp_pipeline import run_batch
    >>> pipelines = ["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]
    >>>
    >>> # Smoke test
    >>> manifest = run_batch(nearby_TOI_MD_df.iloc[:5].reset_index(drop=True),
    ...                      pipelines=pipelines)
    >>>
    >>> # Full batch
    >>> manifest = run_batch(nearby_TOI_MD_df, pipelines=pipelines)
    >>>
    >>> # HLSPs only
    >>> manifest = run_batch(nearby_TOI_MD_df, pipelines=pipelines,
    ...                      run_nemesis=False)
    >>>
    >>> # With NEMESIS caching enabled
    >>> manifest = run_batch(nearby_TOI_MD_df, pipelines=pipelines,
    ...                      nemesis_settings={"keep_FITS": True})
    >>>
    >>> # Inspect failures
    >>> manifest[manifest["status"] != "done"][
    ...     ["tic_id", "status", "nemesis_status", "error"]]
    """
    os.makedirs(results_root, exist_ok=True)
    manifest_path = os.path.join(results_root, "run_manifest.csv")

    manifest  = _load_manifest(manifest_path)
    done_tics: set = set()
    if not force_rerun and not force_releovetter:
        done_tics = set(
            manifest.loc[manifest["status"] == "done", "tic_id"].astype(str)
        )


    n_total = len(catalog_df)
    n_skip  = sum(
        1 for _, row in catalog_df.iterrows()
        if str(int(row["TIC ID"])) in done_tics
    )
    if force_releovetter:
        mode_str = "LEO-Vetter only (cached photometry, all targets)"
    elif run_nemesis:
        mode_str = "NEMESIS enabled"
    else:
        mode_str = "NEMESIS disabled"

    print(
        f"[batch] {n_total} targets total, {n_skip} already done, "
        f"{n_total - n_skip} to run.  Mode: {mode_str}."
        + (f"  LEO-Vetter {'enabled' if run_leovetter else 'disabled'}.")
    )

    for i, (_, target) in enumerate(catalog_df.iterrows()):
        tic_id  = str(int(target["TIC ID"]))
        elapsed = f"{i + 1}/{n_total}"

        if not force_rerun and not force_releovetter and tic_id in done_tics:
            print(f"[{elapsed}] TIC {tic_id} -- skipping (already done)")
            continue

        print(f"[{elapsed}] TIC {tic_id} -- running ...", flush=True)
        t0 = time.time()

        summary = run_single_target(
            target=target,
            pipelines=pipelines,
            results_root=results_root,
            target_Sector=target_Sector,
            run_nemesis=run_nemesis,
            nemesis_settings=nemesis_settings,
            nemesis_gap_threshold_days=nemesis_gap_threshold_days,
            run_leovetter=run_leovetter,
            force_rerun=force_rerun,
            force_releovetter=force_releovetter,
            refine_t0 = refine_t0,
            verbose=verbose,
        )
        wall = time.time() - t0

        _upsert_manifest(summary, manifest_path)

        print(
            f"[{elapsed}] TIC {tic_id} -- {summary['status'].upper()}  "
            f"({wall:.1f}s, sectors={summary['sector_tag']}, "
            f"hlsp_ok={summary['pipelines_ok']}, "
            f"nemesis={summary['nemesis_status']})"
        )
        if summary["status"] == "failed":
            print(f"         error: {summary['error']}")

    final_manifest = _load_manifest(manifest_path)
    n_done   = (final_manifest["status"] == "done").sum()
    n_failed = (final_manifest["status"] == "failed").sum()
    print(f"\n[batch] Complete -- {n_done} done, {n_failed} failed.")
    print(f"[batch] Manifest: {manifest_path}")
    return final_manifest