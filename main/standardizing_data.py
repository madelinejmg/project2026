from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Union, Tuple, Dict, List
import os, re

import numpy as np
import pandas as pd
import colorsys
import astropy.units as u
from astropy.io import fits
import lksearch as lk

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Default
#ID = 259377017 # NOTE: FIXED FOR NOW!
#SECTOR = 3 # NOTE: NOT FIXED!  
DEFAULT_RADIUS = 3 * 21 * u.arcsec
DEFAULT_CADENCE = "30 minute"
DEFAULT_DOWNLOADPATH = os.getcwd() + "/HLSP/"

cadence_map = {
        'long': (30*u.minute , 'FFI'),
        '30 minute': (30*u.minute , 'FFI'),
        '10 minute': (10*u.minute , 'FFI'),
        'short': ( 2*u.minute , 'TPF'),
        '2 minute': ( 2*u.minute , 'TPF'),
        '20 second': (20*u.second, 'TPF'),
        'fast': (20*u.second, 'TPF') }

if DEFAULT_CADENCE not in cadence_map:
    raise ValueError(f"Unrecognised cadence: {DEFAULT_CADENCE!r}")

exp_u, ffi_or_tpf = cadence_map[DEFAULT_CADENCE]
exptime  = int(exp_u.to(u.second).value)

# 1) Helpfer fucntion to download and standardize outputs from various TESS HLSP pipelines
def standardize_lc(
    lc: pd.DataFrame,
    pipeline: str,
    *,
    tic_id: Optional[int] = None,
    sector: Optional[int] = None,
    exptime: Optional[Union[int, float, str]] = None,
    strict: bool = False,
    case_insensitive: bool = True,
) -> pd.DataFrame:
    """
    Standardize a light-curve DataFrame from multiple TESS HLSP pipelines.

    Parameters
    ----------
    lc : pd.DataFrame
        Raw light-curve table from a given pipeline.
    pipeline : str
        Pipeline key (e.g. 'QLP', 'TESS-SPOC', 'TGLC', 'GSFC-ELEANOR-LITE').
    strict : bool, optional
        If True, raise KeyError on a missing required column.  Default False.
    case_insensitive : bool, optional
        Match input column names case-insensitively.  Default True.

    Returns
    -------
    out : pd.DataFrame
        Standardized DataFrame with columns:
        time, flux_raw, flux_raw_err, flux_corr, flux_corr_err,
        flux_bkg, flux_bkg_err, quality, quality2.

        ``quality2`` is populated only for TGLC (→ TESS_flags); all other
        pipelines produce an all-NaN column.  Use ``quality2`` together with
        ``quality`` when building quality masks for TGLC data.

    Notes
    -----
    Mapping is key-based: each schema column name maps to a raw column name
    (or None for an all-NaN placeholder).

    Examples
    --------
    >>> out = standardize_lc(raw_df, "TGLC")
    >>> list(out.columns)
    ['time', 'flux_raw', 'flux_raw_err', 'flux_corr', 'flux_corr_err',
     'flux_bkg', 'flux_bkg_err', 'quality', 'quality2']
    >>> out["quality2"].notna().any()   # TESS_flags populated
    True
    >>> out2 = standardize_lc(raw_df, "QLP")
    >>> out2["quality2"].isna().all()   # not used for QLP
    True
    """
    SCHEMA_COLUMNS: Tuple[str, ...] = (
        "time",
        "flux_raw",
        "flux_raw_err",
        "flux_corr",
        "flux_corr_err",
        "flux_bkg",
        "flux_bkg_err",
        "quality",
        "quality2",     # secondary flag; only TGLC populates this
    )

    PIPELINE_COLUMN_MAP: Dict[str, Dict[str, Optional[str]]] = {
        "QLP": {
            "time":          "TIME",
            "flux_raw":      "SAP_FLUX",
            "flux_raw_err":  None,
            "flux_corr":     "KSPSAP_FLUX",
            "flux_corr_err": "KSPSAP_FLUX_ERR",
            "flux_bkg":      "SAP_BKG",
            "flux_bkg_err":  "SAP_BKG_ERR",
            "quality":       "QUALITY",
            "quality2":      None,
        },
        "TESS-SPOC": {
            "time":          "TIME",
            "flux_raw":      "SAP_FLUX",
            "flux_raw_err":  "SAP_FLUX_ERR",
            "flux_corr":     "PDCSAP_FLUX",
            "flux_corr_err": "PDCSAP_FLUX_ERR",
            "flux_bkg":      "SAP_BKG",
            "flux_bkg_err":  "SAP_BKG_ERR",
            "quality":       "QUALITY",
            "quality2":      None,
        },
        "TGLC": {
            "time":          "time",
            "flux_raw":      "aperture_flux",
            "flux_raw_err":  None,
            "flux_corr":     "cal_aper_flux",
            "flux_corr_err": None,
            "flux_bkg":      "background",
            "flux_bkg_err":  None,
            "quality":       "TGLC_flags",
            "quality2":      "TESS_flags",
        },
        "GSFC-ELEANOR-LITE": {
            "time":          "TIME",
            "flux_raw":      "RAW_FLUX",
            "flux_raw_err":  "FLUX_ERR",
            "flux_corr":     "PCA_FLUX",
            "flux_corr_err": None,
            "flux_bkg":      "FLUX_BKG",
            "flux_bkg_err":  None,
            "quality":       "QUALITY",
            "quality2":      None,
        },
        "NEMESIS": {
            "time":          "TIME",
            "flux_raw":      "RAW_FLUX",
            "flux_raw_err":  "RAW_FLUX_ERR",
            "flux_corr":     "CORR_FLUX",
            "flux_corr_err": "CORR_FLUX_ERR",
            "flux_bkg":      "BKG_FLUX",
            "flux_bkg_err":  "BKG_FLUX_ERR",
            "quality":       "QUALITY",
            "quality2":      None,
        },
    }

    if pipeline not in PIPELINE_COLUMN_MAP:
        raise KeyError(
            f"Unknown pipeline '{pipeline}'. "
            f"Supported: {sorted(PIPELINE_COLUMN_MAP.keys())}"
        )

    old_cols = PIPELINE_COLUMN_MAP[pipeline]
    if len(old_cols) != len(SCHEMA_COLUMNS):
        raise ValueError(
            f"Mapping for pipeline='{pipeline}' has {len(old_cols)} entries, "
            f"expected {len(SCHEMA_COLUMNS)}."
        )

    if case_insensitive:
        col_lookup = {str(c).casefold(): c for c in lc.columns}
        def resolve(name: str) -> Optional[str]:
            return col_lookup.get(name.casefold())
    else:
        def resolve(name: str) -> Optional[str]:
            return name if name in lc.columns else None

    out = pd.DataFrame(index=lc.index)
    for new_name in SCHEMA_COLUMNS:
        old_name = old_cols[new_name]
        if old_name is None:
            out[new_name] = np.nan
            continue
        actual = resolve(old_name)
        if actual is None:
            if strict:
                raise KeyError(
                    f"Missing column '{old_name}' for pipeline='{pipeline}'. "
                    f"Available: {list(lc.columns)}"
                )
            out[new_name] = np.nan
            continue
        out[new_name] = lc[actual].to_numpy(copy=False)

    return out


# ---------------------------------------------------------------------------

def get_tess_lc(
    TIC_ID: Union[int, str],
    pipeline: str,
    radius: u.Quantity = DEFAULT_RADIUS,
    exptime: str = DEFAULT_CADENCE,
    Sector: Optional[Union[int, List[int]]] = None,
    downloadpath: str = DEFAULT_DOWNLOADPATH,
    *,
    verbose: bool = True,
    choose_first_timeseries: bool = True,
    preloaded_search: Optional[Any] = None,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Download one TESS HLSP light curve (FITS) and return raw + standardized
    DataFrames.

    Parameters
    ----------
    TIC_ID : int or str
        TIC identifier.
    pipeline : str
        HLSP pipeline name.
    radius : astropy.units.Quantity
        Cone-search radius.
    exptime : str
        Cadence/exptime string.
    Sector : int, list of int, or None
        TESS sector filter.
    downloadpath : str
        Download directory.
    verbose : bool
        Print selected-product summary.
    choose_first_timeseries : bool
        Sort by earliest time column before selecting best row.
    preloaded_search : lksearch.TESSSearch or None
        Pre-fetched ``.timeseries`` result.  When provided the MAST query is
        skipped and this object is filtered directly by pipeline.  Pass from
        ``collect_lightcurves_for_target`` to avoid redundant network calls.

    Returns
    -------
    product : lksearch.TESSSearch
        Single-row search object for the selected product.
    raw_df : pd.DataFrame
        Raw light-curve table from FITS.
    std_df : pd.DataFrame
        Standardized light-curve DataFrame.
    """
    os.makedirs(downloadpath, exist_ok=True)

    tic_str = str(TIC_ID).strip()
    try:
        tic_str = str(int(float(tic_str)))
    except Exception:
        pass

    search_radius = (
        float(radius) if isinstance(radius, (int, float, np.floating)) else radius
    )

    # ── MAST query (skipped when caller supplies a preloaded search) ──────────
    if preloaded_search is not None:
        ts = preloaded_search
    else:
        search = lk.TESSSearch(
            target=f"TIC {tic_str}",
            search_radius=search_radius,
            exptime=exptime,
            sector=Sector,
            hlsp=True,
        )
        try:
            ts = search.timeseries
        except Exception:
            ts = search

    # ── Filter to pipeline ────────────────────────────────────────────────────
    filtered = ts.filter_table(mission="HLSP", pipeline=pipeline)

    if filtered.table is None or len(filtered.table) == 0:
        table = getattr(ts, "table", None)
        if isinstance(table, pd.DataFrame) and len(table) > 0:
            hlsp_tbl = (
                table[table["mission"].astype(str).eq("HLSP")]
                if "mission" in table.columns else table
            )
            avail = (
                np.unique(hlsp_tbl["pipeline"].astype(str))
                if "pipeline" in hlsp_tbl.columns and len(hlsp_tbl) > 0
                else np.array([])
            )
            raise ValueError(
                f"No HLSP product for pipeline='{pipeline}' "
                f"(TIC={tic_str}, sector={Sector}, exptime={exptime}). "
                f"Available: {avail.tolist()}"
            )
        raise ValueError(
            f"No products for TIC={tic_str}, sector={Sector}, exptime={exptime}."
        )

    tbl = filtered.table.copy()

    # ── Select best row ───────────────────────────────────────────────────────
    sort_cols = []
    time_col_used = None

    if choose_first_timeseries:
        for candidate in ["t_min", "tstart", "t_min_btjd", "start_time", "year"]:
            if candidate in tbl.columns:
                sort_cols.append(candidate)
                time_col_used = candidate
                break
        if "distance" in tbl.columns:
            sort_cols.append("distance")
        if not sort_cols and "description" in tbl.columns:
            sort_cols = ["description"]
        elif not sort_cols and "distance" in tbl.columns:
            sort_cols = ["distance"]
    else:
        if "distance" in tbl.columns:
            sort_cols = ["distance"]
        elif "t_min" in tbl.columns:
            sort_cols = ["t_min"]

    if sort_cols:
        tbl = tbl.sort_values(sort_cols, ascending=True).reset_index(drop=True)

    best_tbl = tbl.iloc[[0]].copy()
    product  = lk.TESSSearch(table=best_tbl)

    if verbose:
        cols = [c for c in
                ["target_name", "pipeline", "mission", "sector",
                 "exptime", "distance", "year", "description"]
                if c in best_tbl.columns]
        print("Selected product row:")
        print(best_tbl[cols] if cols else best_tbl.head(1))
        if choose_first_timeseries:
            print(f"choose_first_timeseries=True; sorted using: {time_col_used}")

    # ── Download ──────────────────────────────────────────────────────────────
    manifest = product.download(download_dir=downloadpath)
    if not isinstance(manifest, pd.DataFrame) or len(manifest) == 0:
        raise ValueError("Download returned an empty manifest.")

    path_col = None
    for c in manifest.columns:
        if c.lower().replace(" ", "").replace("_", "") == "localpath":
            path_col = c
            break
    if path_col is None:
        raise ValueError(
            f"No Local Path column in manifest. Columns: {list(manifest.columns)}"
        )
    local_path = str(manifest[path_col].iloc[0])

    # ── Read FITS ─────────────────────────────────────────────────────────────
    with fits.open(local_path, memmap=False) as hdul:
        table_hdu = None
        for hdu in hdul[1:]:
            data    = getattr(hdu, "data", None)
            extname = str(getattr(hdu, "name", "")).upper()
            if data is None:
                continue
            if hasattr(data, "names") and extname in {
                "LIGHTCURVE", "LIGHTCURVES", "LC", "TIME_SERIES", "TIMESERIES"
            }:
                table_hdu = hdu
                break
            if table_hdu is None and hasattr(data, "names"):
                table_hdu = hdu

        if table_hdu is None:
            raise ValueError(f"No table-like FITS extension in: {local_path}")

        rec = np.array(table_hdu.data)
        if hasattr(rec.dtype, "isnative") and not rec.dtype.isnative:
            try:
                rec = rec.byteswap().newbyteorder()
            except AttributeError:
                rec = rec.byteswap().view(rec.dtype.newbyteorder("="))

        raw_df = pd.DataFrame.from_records(rec)
        std_df = standardize_lc(raw_df, pipeline)

    return product, raw_df, std_df


# ---------------------------------------------------------------------------




# NOTE: Generate N distinct colors for plots!
def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

# Multi-pipeline wrapper 
# ---------------------------------------------------------------------------
# New helper: per-sector flux normalization
# ---------------------------------------------------------------------------

def normalize_standardized_lc(std_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all flux columns in a standardized LC DataFrame by their
    per-sector nanmedian so that each flux type has a baseline of ~1.0.

    Each flux column is divided by ``nanmedian`` of that column's finite
    values.  The corresponding error column is divided by the **same**
    median (preserving the signal-to-noise ratio).  Background flux is
    normalized independently.

    Normalization pairs
    -------------------
    ``flux_raw``  / ``flux_raw_err``   → divided by ``nanmedian(flux_raw)``
    ``flux_corr`` / ``flux_corr_err``  → divided by ``nanmedian(flux_corr)``
    ``flux_bkg``  / ``flux_bkg_err``   → divided by ``nanmedian(flux_bkg)``

    Columns ``time`` and ``quality`` are left untouched.

    Parameters
    ----------
    std_df : pd.DataFrame
        Standardized DataFrame from ``standardize_lc``.  Expected columns:
        ``time``, ``flux_raw``, ``flux_raw_err``, ``flux_corr``,
        ``flux_corr_err``, ``flux_bkg``, ``flux_bkg_err``, ``quality``.

    Returns
    -------
    out : pd.DataFrame
        Copy of ``std_df`` with flux/error columns normalized in-place.
        Returns the input unchanged if a median is zero, non-finite, or the
        column is absent (safe no-op).

    Notes
    -----
    - Normalization is computed only from finite values to guard against
      NaN-dominated columns (common for pipelines that do not populate
      certain flux types).
    - A median of exactly 0.0 is treated as invalid and that pair is skipped.
    - This function must be applied **per sector** before concatenation;
      applying it to an already-concatenated multi-sector LC would
      normalize across the joint baseline, defeating the purpose.

    Examples
    --------
    >>> std_df = standardize_lc(raw_df, "QLP")
    >>> std_norm = normalize_standardized_lc(std_df)
    >>> float(np.nanmedian(std_norm["flux_corr"]))   # ≈ 1.0
    1.0
    """
    NORM_PAIRS = [
        ("flux_raw",  "flux_raw_err"),
        ("flux_corr", "flux_corr_err"),
        ("flux_bkg",  "flux_bkg_err"),
    ]

    out = std_df.copy()

    for flux_col, err_col in NORM_PAIRS:
        if flux_col not in out.columns:
            continue

        vals = out[flux_col].to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]

        if finite_vals.size == 0:
            continue  # entire column is NaN — nothing to normalize

        median = np.nanmedian(finite_vals)

        if not np.isfinite(median) or median == 0.0:
            continue  # degenerate; leave column as-is

        out[flux_col] = out[flux_col] / median

        if err_col in out.columns:
            out[err_col] = out[err_col] / median

    return out


# ---------------------------------------------------------------------------
# Full replacement: collect_lightcurves_for_target
# (only change: normalize_standardized_lc called after standardize_lc)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _apply_quality_mask(std_df: pd.DataFrame, pipeline: str = "") -> pd.DataFrame:
    """
    Apply a quality mask to a standardized LC DataFrame and validate the result.

    Mask logic
    ----------
    - All pipelines : ``quality == 0``
    - TGLC          : ``quality == 0`` AND ``quality2 == 0``

    ``quality2`` NaN values (non-TGLC pipelines) are filled with 0 so they
    pass the secondary check unconditionally.

    Parameters
    ----------
    std_df : pd.DataFrame
        Normalized, standardized LC with at least a ``quality`` column.
    pipeline : str, optional
        Pipeline name used in error messages only.

    Returns
    -------
    masked : pd.DataFrame
        Subset of ``std_df`` where all quality flags are zero, index reset.

    Raises
    ------
    ValueError
        If any non-zero ``quality`` or ``quality2`` rows survive the mask.

    Examples
    --------
    >>> masked = _apply_quality_mask(std_df, pipeline="TGLC")
    >>> (masked["quality"]  == 0).all()
    True
    >>> (masked["quality2"] == 0).all()
    True
    """
    q_mask = std_df["quality"] == 0

    if "quality2" in std_df.columns:
        q2 = pd.to_numeric(std_df["quality2"], errors="coerce").fillna(0)
        q_mask = q_mask & (q2 == 0)

    masked = std_df.loc[q_mask].reset_index(drop=True)

    # ── Validation ────────────────────────────────────────────────────────────
    bad_q = int((masked["quality"] != 0).sum())
    if bad_q:
        raise ValueError(
            f"[{pipeline}] _apply_quality_mask: {bad_q} non-zero 'quality' "
            f"rows survived the mask — check flag values."
        )

    if "quality2" in masked.columns:
        q2_check = pd.to_numeric(masked["quality2"], errors="coerce").fillna(0)
        bad_q2 = int((q2_check != 0).sum())
        if bad_q2:
            raise ValueError(
                f"[{pipeline}] _apply_quality_mask: {bad_q2} non-zero 'quality2' "
                f"rows survived the mask — check TESS_flags values."
            )

    return masked


def collect_lightcurves_for_target(
    tic_id: Union[int, str],
    sector: int,
    pipelines: List[str],
    downloadpath: str = DEFAULT_DOWNLOADPATH,
    *,
    radius: u.Quantity = DEFAULT_RADIUS,
    exptime: str = DEFAULT_CADENCE,
    apply_quality_mask: bool = True,
    verbose: bool = True,
    choose_first_timeseries: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Download, standardize, and normalize light curves from multiple TESS HLSP
    pipelines for one target in one sector.

    Issues a **single** ``lk.TESSSearch`` call shared across all pipelines,
    then filters per pipeline before downloading.  Reduces MAST round-trips
    from N_pipelines to 1.

    Quality masking is pipeline-aware:

    - All pipelines: ``quality == 0``
    - TGLC only:     ``quality == 0`` AND ``quality2 == 0``
      (i.e. both TGLC_flags and TESS_flags must be zero)

    The mask is applied to the already-normalized standardized DataFrame so
    both ``standardized`` and ``standardized_masked`` share the same
    fractional-flux baseline.

    Parameters
    ----------
    tic_id : int or str
        TIC identifier.
    sector : int
        TESS sector.
    pipelines : list of str
        Pipeline names.
    downloadpath : str, optional
        Download directory.
    radius : astropy.units.Quantity, optional
        Cone-search radius.
    exptime : str, optional
        Cadence/exptime string.
    apply_quality_mask : bool, optional
        Produce ``standardized_masked``.  Default True.
    verbose : bool, optional
        Print progress.  Default True.
    choose_first_timeseries : bool, optional
        Select earliest time-series product.  Default True.

    Returns
    -------
    results : dict
        Pipeline-keyed dict with keys: ``product``, ``raw``,
        ``standardized``, ``standardized_masked``, ``tic_id``, ``sector``,
        ``pipeline``, ``n_raw``, ``n_standardized``, ``n_masked``,
        ``status``, ``error``.

    Examples
    --------
    >>> results = collect_lightcurves_for_target(
    ...     tic_id=259377017, sector=3,
    ...     pipelines=["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"],
    ... )
    >>> results["TGLC"]["status"]
    'ok'
    >>> # Both TGLC flag columns used for masking:
    >>> (results["TGLC"]["standardized_masked"]["quality"]  == 0).all()
    True
    >>> (results["TGLC"]["standardized_masked"]["quality2"] == 0).all()
    True
    """
    tic_str = str(tic_id).strip()
    try:
        tic_str = str(int(float(tic_str)))
    except Exception:
        pass

    search_radius = (
        float(radius) if isinstance(radius, (int, float, np.floating)) else radius
    )

    # ── Single MAST query shared across all pipelines ─────────────────────────
    if verbose:
        print(f"[collect] TESSSearch TIC {tic_str}, sector {sector}, "
              f"exptime={exptime!r}  (1 query for {len(pipelines)} pipelines)")

    search = lk.TESSSearch(
        target=f"TIC {tic_str}",
        search_radius=search_radius,
        exptime=exptime,
        sector=sector,
        hlsp=True,
    )
    try:
        ts = search.timeseries
    except Exception:
        ts = search

    # ── Per-pipeline: filter → download → standardize → normalize → mask ──────
    results: Dict[str, Dict[str, Any]] = {}

    for pipeline in pipelines:
        try:
            if verbose:
                print(f"  [{pipeline}] downloading ...")

            product, raw_df, std_df = get_tess_lc(
                TIC_ID=tic_id,
                pipeline=pipeline,
                radius=radius,
                exptime=exptime,
                Sector=sector,
                downloadpath=downloadpath,
                verbose=verbose,
                choose_first_timeseries=choose_first_timeseries,
                preloaded_search=ts,
            )

            std_df = normalize_standardized_lc(std_df)

            if apply_quality_mask:
                std_masked_df = _apply_quality_mask(std_df, pipeline=pipeline)
            else:
                std_masked_df = None

            results[pipeline] = {
                "product":             product,
                "raw":                 raw_df,
                "standardized":        std_df,
                "standardized_masked": std_masked_df,
                "tic_id":              tic_id,
                "sector":              sector,
                "pipeline":            pipeline,
                "n_raw":               len(raw_df),
                "n_standardized":      len(std_df),
                "n_masked":            len(std_masked_df) if std_masked_df is not None else None,
                "status":              "ok",
                "error":               None,
            }

        except Exception as exc:
            results[pipeline] = {
                "product":             None,
                "raw":                 None,
                "standardized":        None,
                "standardized_masked": None,
                "tic_id":              tic_id,
                "sector":              sector,
                "pipeline":            pipeline,
                "n_raw":               None,
                "n_standardized":      None,
                "n_masked":            None,
                "status":              "failed",
                "error":               str(exc),
            }

    return results