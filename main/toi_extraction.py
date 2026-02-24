# toi_extraction.py
# ------------------------------------------------------------
# Select a target TOI from a nearby M-dwarf TOI catalog DataFrame,
# then fetch HLSP light curves for multiple pipelines and return
# structured results (no globals()).
#
# Usage in notebook:
#   from toi_extraction import extract_tois_lcs
#   results = extract_tois_lcs(nearby_TOI_MD_df)
#   results["target"]         # target row as DataFrame (1 row)
#   results["meta"]           # dict of P, T0, Dur, Depth, TIC, Sector
#   results["results"]["TGLC"]["newLC"]
#   results["masked"]["QLP"]
#   results["errors"]         # pipelines that failed (if any)
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from TESS_pipeline import get_tess_LC


def _parse_first_sector(sectors_value: Union[str, int, float]) -> int:
    """Parse a sectors field like '3,4,5' (possibly with spaces) and return the smallest sector."""
    s = str(sectors_value)
    parts = [p.strip() for p in s.split(",")]
    ints = []
    for p in parts:
        # allow things like "3" or "3 " or "3.0"
        try:
            ints.append(int(float(p)))
        except Exception:
            continue
    if not ints:
        raise ValueError(f"Could not parse any sector integers from Sectors={sectors_value!r}")
    return int(min(ints))


def select_largest_radius_target(nearby_TOI_MD_df: pd.DataFrame) -> pd.DataFrame:
    """Return a 1-row DataFrame for the target with max Planet Radius Value (ignoring NaNs)."""
    if "Planet Radius Value" not in nearby_TOI_MD_df.columns:
        raise KeyError("Missing required column: 'Planet Radius Value'")

    col = nearby_TOI_MD_df["Planet Radius Value"]

    if not col.notna().any():
        raise ValueError("'Planet Radius Value' is all NaN; cannot select largest-radius target.")

    max_val = col.max(skipna=True)
    target = nearby_TOI_MD_df.loc[col == max_val].head(1).reset_index(drop=True)

    if len(target) == 0:
        raise ValueError("Target selection produced no rows (unexpected).")

    return target


def extract_target_meta(target: pd.DataFrame) -> Dict[str, Any]:
    """Extract period, epoch, depth, duration, TIC ID, and first sector from a 1-row target df."""
    # Safer .iloc[0] than .item() if columns ever contain arrays/objects
    row = target.iloc[0]

    required_cols = [
        "Orbital Period (days) Value",
        "Orbital Epoch Value",
        "Transit Depth Value",
        "Transit Duration (hours) Value",
        "Sectors",
        "TIC ID",
    ]
    missing = [c for c in required_cols if c not in target.columns]
    if missing:
        raise KeyError(f"Target is missing required columns: {missing}")

    target_P = float(row["Orbital Period (days) Value"])
    target_T0 = float(row["Orbital Epoch Value"])
    # Catalog is in ppm -> convert to fraction
    target_Dep = float(row["Transit Depth Value"]) / 1e6
    target_Dur = float(row["Transit Duration (hours) Value"])

    target_Sector = _parse_first_sector(row["Sectors"])
    tic_id = int(float(row["TIC ID"]))

    return {
        "P_days": target_P,
        "T0": target_T0,
        "Depth_frac": target_Dep,
        "Duration_hours": target_Dur,
        "TIC_ID": tic_id,
        "Sector": target_Sector,
    }


def fetch_pipeline_lcs(
    tic_id: Union[int, str],
    sector: Optional[Union[int, list[int]]],
    *,
    radius_arcsec: Union[int, float] = 21,
    exptime: Union[int, str, Tuple[float, float]] = 1800,
    pipelines: Optional[list[str]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Fetch LCs for multiple pipelines.

    Returns
    -------
    results : dict
        results[pipeline] = {"r": product, "LC": raw_df, "newLC": standardized_df}
    masked : dict
        masked[pipeline] = standardized_df where Quality == 0
    errors : dict
        errors[pipeline] = repr(exception) if that pipeline failed
    """
    if pipelines is None:
        pipelines = ["TGLC", "QLP", "TESS-SPOC", "GSFC-ELEANOR-LITE"]

    results: Dict[str, Dict[str, Any]] = {}
    masked: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    for P in pipelines:
        if verbose:
            print(f"Trying to get {P} LC for TIC {tic_id} in Sector {sector}")
        try:
            r, LC, newLC = get_tess_LC(
                TIC_ID=tic_id,
                radius=radius_arcsec,   # float => arcsec in your get_tess_LC normalization
                exptime=exptime,
                Sector=sector,
                pipeline=P,
                verbose=verbose,
            )

            results[P] = {"r": r, "LC": LC, "newLC": newLC}

            if "Quality" in newLC.columns:
                masked[P] = newLC.loc[newLC["Quality"] == 0].copy()
            else:
                # Some pipelines might not map quality; keep unmasked as fallback
                masked[P] = newLC.copy()

        except Exception as e:
            errors[P] = repr(e)
            if verbose:
                print(f" X {P} failed: {e}")

    return results, masked, errors


def extract_tois_lcs(
    nearby_TOI_MD_df: pd.DataFrame,
    *,
    pipelines: Optional[list[str]] = None,
    radius_arcsec: Union[int, float] = 21,
    exptime: Union[int, str, Tuple[float, float]] = 1800,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end helper:
      1) select max-radius TOI from the catalog df
      2) extract orbital/transit metadata
      3) download + standardize multi-pipeline LCs

    Returns a dict with keys:
      - "target"  : 1-row DataFrame
      - "meta"    : dict of target parameters
      - "results" : dict of per-pipeline {r, LC, newLC}
      - "masked"  : dict of per-pipeline newLC[Quality==0]
      - "errors"  : dict of per-pipeline failures (if any)
    """
    target = select_largest_radius_target(nearby_TOI_MD_df)
    meta = extract_target_meta(target)

    if verbose:
        print(
            meta["P_days"],
            meta["T0"],
            meta["Duration_hours"],
            meta["Depth_frac"],
            meta["TIC_ID"],
            meta["Sector"],
        )

    results, masked, errors = fetch_pipeline_lcs(
        meta["TIC_ID"],
        meta["Sector"],
        radius_arcsec=radius_arcsec,
        exptime=exptime,
        pipelines=pipelines,
        verbose=verbose,
    )

    return {
        "target": target,
        "meta": meta,
        "results": results,
        "masked": masked,
        "errors": errors,
    }