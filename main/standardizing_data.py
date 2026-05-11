from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Union, Tuple, Dict, List
import os, re, json

import numpy as np
import pandas as pd
import colorsys
import astropy.units as u
from astropy.io import fits
import lksearch as lk
import lksearch as _lk

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

# Module-level constant
_CACHE_INDEX_FILENAME = "hlsp_fits_cache.json"

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


# Cache helpers
def _load_fits_cache(downloadpath: str) -> Dict[str, str]:
    """
    Load the local FITS cache index from ``{downloadpath}/hlsp_fits_cache.json``.

    Returns an empty dict if the file does not yet exist.

    Parameters
    ----------
    downloadpath : str
        Directory where HLSP files are (or will be) stored.

    Returns
    -------
    index : dict
        Mapping of cache_key (str) -> absolute local file path (str).

    Examples
    --------
    >>> idx = _load_fits_cache("/data/HLSP")
    >>> idx.get("259377017__QLP__3__30_minute")
    '/data/HLSP/mastDownload/HLSP/.../file.fits'
    """
    index_path = os.path.join(downloadpath, _CACHE_INDEX_FILENAME)
    if os.path.exists(index_path):
        with open(index_path, "r") as fh:
            return json.load(fh)
    return {}


def _update_fits_cache(downloadpath: str, key: str, local_path: str) -> None:
    """
    Add or update one entry in the FITS cache index and write it to disk.

    Parameters
    ----------
    downloadpath : str
        Directory where the cache index lives.
    key : str
        Cache key for this product (see ``_build_fits_cache_key``).
    local_path : str
        Absolute path to the downloaded FITS file.

    Examples
    --------
    >>> _update_fits_cache("/data/HLSP", "259377017__QLP__3__30_minute",
    ...                    "/data/HLSP/mastDownload/HLSP/qlp.fits")
    """
    os.makedirs(downloadpath, exist_ok=True)
    index = _load_fits_cache(downloadpath)
    index[key] = local_path
    index_path = os.path.join(downloadpath, _CACHE_INDEX_FILENAME)
    with open(index_path, "w") as fh:
        json.dump(index, fh, indent=2)


def _build_fits_cache_key(
    best_tbl: pd.DataFrame,
    tic_str: str,
    pipeline: str,
    exptime: str,
) -> str:
    """
    Build a deterministic cache key for a single HLSP product.

    Strategy (priority order):
    1. Use ``dataURI`` / ``obs_id`` / ``productFilename`` from the product table
       — these are MAST-assigned unique identifiers.
    2. Fall back to ``(tic_str, pipeline, sector, exptime)`` constructed from
       the selected product row.

    Parameters
    ----------
    best_tbl : pd.DataFrame
        Single-row product table as selected inside ``get_tess_lc``.
    tic_str : str
        Normalised TIC identifier string (no leading zeros or ".0").
    pipeline : str
        HLSP pipeline name.
    exptime : str
        Cadence/exposure-time string (e.g. ``"30 minute"``).

    Returns
    -------
    key : str
        Cache key string safe for use as a JSON dict key.

    Examples
    --------
    >>> key = _build_fits_cache_key(best_tbl, "259377017", "QLP", "30 minute")
    >>> key
    '259377017__QLP__3__30_minute'
    """
    # Attempt to derive key from MAST product-level identifiers first
    for uri_col in ("dataURI", "dataurl", "obs_id", "obsid", "productFilename"):
        for col in best_tbl.columns:
            if col.lower().replace("_", "") == uri_col.lower().replace("_", ""):
                val = str(best_tbl[col].iloc[0]).strip()
                if val and val.lower() not in {"nan", "none", ""}:
                    # Sanitise for use as a JSON key (no path separators)
                    safe = val.replace("/", "__").replace("\\", "__")
                    return safe

    # Fallback: construct from (tic, pipeline, sector, exptime)
    sector_val = "None"
    for s_col in ("sector", "sequence_number", "year"):
        if s_col in best_tbl.columns:
            raw = str(best_tbl[s_col].iloc[0]).strip()
            if raw.lower() not in {"nan", "none", ""}:
                sector_val = raw
                break

    safe_exptime = exptime.replace(" ", "_")
    return f"{tic_str}__{pipeline}__{sector_val}__{safe_exptime}"

# 1) Download and standardize outputs from various TESS HLSP pipelines
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
    lc : pandas.DataFrame
        Input light-curve table from a given pipeline.
    pipeline : str
        Pipeline key used in the mapping (e.g., 'QLP', 'TESS-SPOC', 'TGLC', 'GSFC-ELEANOR-LITE').
    strict : bool, optional
        If True, raise KeyError when a required mapped input column is missing.
        If False, missing/None columns become NaN-filled outputs.
    case_insensitive : bool, optional
        If True, allows matching input columns ignoring case.

    Returns
    -------
    out : pandas.DataFrame
        DataFrame with standardized columns in a fixed order:
        ['Time', 'Raw Flux', 'Raw Flux Error', 'BKG Flux', 'BKG Flux Error',
         'Corrected Flux', 'Corrected Flux Error', 'Quality']

    Notes
    -----
    - Mapping is positional: the i-th old column maps to the i-th standardized column.
    - Any `None` in the mapping produces an all-NaN output column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"time":[1.0, 2.0], "cal_psf_flux":[10.0, 11.0], "background":[0.1, 0.2], "TGLC_flags":[0, 1]})
    >>> out = standardize_lc(df, "TGLC")
    >>> list(out.columns)
    ['Time', 'Raw Flux', 'Raw Flux Error', 'BKG Flux', 'BKG Flux Error', 'Corrected Flux', 'Corrected Flux Error', 'Quality']
    >>> float(out["Raw Flux"].iloc[0])
    10.0
    >>> np.isnan(out["Raw Flux Error"].iloc[0])
    True
    """
    # Standard name conventions for each pipeline
    SCHEMA_COLUMNS: Tuple[str,...] = (
        "time",
        "flux_raw",
        "flux_raw_err",
        "flux_corr",
        "flux_corr_err",
        "flux_bkg",
        "flux_bkg_err",
        "quality",
    )

    # PIPELINE COLUMN NAMES
    PIPELINE_COLUMN_MAP: Dict[str, Dict[str, Optional[str]]] = {
        "QLP": {
            "time": "TIME",
            "flux_raw": "SAP_FLUX",
            "flux_raw_err": None, 
            "flux_corr": "KSPSAP_FLUX",
            "flux_corr_err": "KSPSAP_FLUX_ERR",
            "flux_bkg": "SAP_BKG",
            "flux_bkg_err": "SAP_BKG_ERR",
            "quality": "QUALITY",
        },
        "TESS-SPOC": { 
            "time": "TIME",
            "flux_raw": "SAP_FLUX",
            "flux_raw_err": "SAP_FLUX_ERR",
            "flux_corr": "PDCSAP_FLUX",
            "flux_corr_err": "PDCSAP_FLUX_ERR",
            "flux_bkg": "SAP_BKG",
            "flux_bkg_err": "SAP_BKG_ERR",
            "quality": "QUALITY",
        },
        "TGLC": {
            "time": "time",
            "flux_raw": "aperture_flux",
            "flux_raw_err": None,
            "flux_corr": "cal_aper_flux",
            "flux_corr_err": None,
            "flux_bkg": "background",
            "flux_bkg_err": None,
            "quality": "TGLC_flags",
        },
        "GSFC-ELEANOR-LITE": {
            "time": "TIME",
            "flux_raw": "RAW_FLUX",
            "flux_raw_err": "FLUX_ERR",
            "flux_corr": "PCA_FLUX",
            "flux_corr_err": None, 
            "flux_bkg": "FLUX_BKG",
            "flux_bkg_err": None,
            "quality": "QUALITY",
        },

        # NOTE: These are placeholders only.
        "NEMESIS": {
            "time": "TIME",
            "flux_raw": "RAW_FLUX",
            "flux_raw_err": "RAW_FLUX_ERR",
            "flux_corr": "CORR_FLUX",
            "flux_corr_err": "CORR_FLUX_ERR",
            "flux_bkg": "BKG_FLUX",
            "flux_bkg_err": "BKG_FLUX_ERR",
            "quality": "QUALITY",
        },
    }

    if pipeline not in PIPELINE_COLUMN_MAP:
        raise KeyError(f"Unknown pipeline '{pipeline}'. Supported: {sorted(PIPELINE_COLUMN_MAP.keys())}")

    old_cols = PIPELINE_COLUMN_MAP[pipeline]
    if len(old_cols) != len(SCHEMA_COLUMNS):
        raise ValueError(
            f"Mapping for pipeline='{pipeline}' has {len(old_cols)} columns, "
            f"expected {len(SCHEMA_COLUMNS)}."
        )

    # Build a resolver for case-insensitive matching, if requested
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
                    f"Available columns: {list(lc.columns)}"
                )
            out[new_name] = np.nan
            continue

        out[new_name] = lc[actual].to_numpy(copy=False)

    return out

# full replacement with caching helper functions
def get_tess_lc(
    TIC_ID: Union[int, str],
    pipeline: str,
    radius: u.Quantity = DEFAULT_RADIUS,
    exptime: str = DEFAULT_CADENCE,
    Sector: Optional[Union[int, list[int]]] = None,
    downloadpath: str = DEFAULT_DOWNLOADPATH,
    *,
    verbose: bool = True,
    choose_first_timeseries: bool = True,
    use_cache: bool = True,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Download one TESS HLSP light curve (FITS) via lksearch and return it 
    as a pair of DataFrames (raw, standardized). Previously downloaded
    products are served from a local JSON-backed cache — the FITS file is
    not re-fetched as long as it still exists on disk.

    Parameters
    ----------
    TIC_ID : int | str
        TIC identifier (e.g., 123456789).
    pipeline : str
        HLSP pipeline name to filter on (e.g., "QLP", "TASOC", "TESS-SPOC", etc.)
    radius : float | astropy.units.Quantity
        Cone-search radius
    exptime : str 
        Exposure-time / cadence key (e.g. ``"30 minute"``).
    Sector : int | list[int] | None
        TESS sector(s) to filter on.
    
    downloadpath : str
        Directory where products will be downloaded.
    verbose : bool
        Print progress / selected-product summary.
    choose_first_timeseries : bool
        If True, prefer the earliest time-series product when multiple rows
        match.
    use_cache : bool
        If True (default), consult the local JSON cache before calling
        ``product.download()``.  Set to False to force a fresh download.

    Returns
    -------
    product : lksearch.TESSSearch (single-row)
        A TESSSearch object containing exactly one selected product row.
    raw_df : pd.DataFrame
        Light-curve table read directly from the FITS BinTable extension.
    std_df : pd.DataFrame
        Standardized DataFrame via ``standardize_lc``.

    Raises
    ------
    ValueError
        If no matching HLSP timeseries product is found, or if the
        download manifest is empty.

    Notes
    -----
    Cache index is written to ``{downloadpath}/hlsp_fits_cache.json``.
    A cache entry is invalidated automatically if the recorded file path
    no longer exists on disk (stale entry).

    Examples
    --------
    >>> product, raw_df, std_df = get_tess_lc(259377017, "QLP", Sector=3)
    >>> product, raw_df, std_df = get_tess_lc(259377017, "QLP", Sector=3)
    # Second call reads from cache — no download.
    """
    os.makedirs(downloadpath, exist_ok=True)

    # Normalize "123.0" -> "123" if user passed a float-like string
    tic_str = str(TIC_ID).strip()
    try:
        tic_str = str(int(float(tic_str)))
    except Exception:
        pass
    
    # lksearch treats float search_radius as arcseconds by default.
    search_radius = float(radius) if isinstance(radius, (int, float, np.floating)) else radius

    # 1) MAST search (always run — it is fast and determines which file
    #  to serve from cache).   
    search = lk.TESSSearch(
        target=f"TIC {tic_str}",
        search_radius=search_radius,
        exptime=exptime,
        sector=Sector,
        hlsp=True,
    )
    # 2) Restrict to time-series products, then filter to HLSP + pipeline
    try:
        ts = search.timeseries
    except:
        # Very defensive fallback; docs/tutorials show .timeseries exists.
        ts = search
    
    # Filter to HLSP + pipeline (and keep exptime/sector constraints via ctor inputs)
    filtered = ts.filter_table(mission="HLSP", pipeline=pipeline)

    if filtered.table is None or len(filtered.table) == 0:
        table = getattr(ts, "table", None)
        if isinstance(table, pd.DataFrame) and len(table) > 0:
            hlsp_tbl = (
                table[table["mission"].astype(str).eq("HLSP")]
                if "mission" in table.columns
                else table
            )
            avail = (
                np.unique(hlsp_tbl["pipeline"].astype(str))
                if "pipeline" in hlsp_tbl.columns and len(hlsp_tbl) > 0
                else np.array([])
            )
            raise ValueError(
                f"No HLSP timeseries product found for pipeline='{pipeline}' "
                f"(TIC={tic_str}, sector={Sector}, exptime={exptime}, "
                f"radius={radius}). Available HLSP pipelines: {avail.tolist()}"
            )
        raise ValueError(
            f"No products returned at all for TIC={tic_str}, sector={Sector}, "
            f"exptime={exptime}, radius={radius}."
        )

    tbl = filtered.table.copy()

    # 3) Row selection (identical logic to original) 

    sort_cols = List[str] = []
    time_col_used = None

    if choose_first_timeseries:
        # Case 1: Prefer true time-like columns first (highly preferred)
        for candidate in ["t_min", "tstart", "t_min_btjd", "start_time", "year"]:
            if candidate in tbl.columns:
                sort_cols.append(candidate)
                time_col_used = candidate
                break
        # Case 2: What if two rows have the same t_min (or very close)? 
        if "distance" in tbl.columns:
            sort_cols.append("distance")

        # Case 3: If none of the preferred options exist, use something else so
        # the code doesn't break
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
    

##########################################################################################


    # Create a single-row TESSSearch object so download() only pulls one file.
    product = _lk.TESSSearch(table=best_tbl)

    if verbose:
        cols = [
            c for c in ["target_name", "pipeline", "mission", "sector", "exptime", 
                        "distance", "year", "description"] 
                        if c in best_tbl.columns
            ]
        print("Selected product row:")
        print(best_tbl[cols] if cols else best_tbl.head(1))

        if choose_first_timeseries:
            print(f"choose_first_timeseries=True; sorted using: {time_col_used}")

    #  Cache lookup — skip download if FITS already on disk   
    cache_key = _build_fits_cache_key(best_tbl, tic_str, pipeline, exptime)
    local_path: Optional[str] = None

    if use_cache:
        cache_index = _load_fits_cache(downloadpath)
        cached_path = cache_index.get(cache_key)
        if cached_path and os.path.isfile(cached_path):
            if verbose:
                print(f"[cache hit]  {pipeline} TIC={tic_str} → {cached_path}")
            local_path = cached_path
        elif cached_path:
            if verbose:
                print(
                    f"[cache stale] recorded path no longer exists: {cached_path}\n"
                    f"              Re-downloading..."
                )

    # 4) Download (only if no valid cached path found)  
    if local_path is None:
        manifest = product.download(download_dir=downloadpath)

        # 5) Extract local path robustly
        if not isinstance(manifest, pd.DataFrame) or len(manifest) == 0:
            raise ValueError("Download returned an empty manifest; nothing was downloaded.")

    # lksearch manifest uses 'Local Path' in tutorials.
    if path_col is None:
        for c in manifest.columns:
            canon = c.lower().replace(" ", "").replace("_", "")
            if canon == "localpath":
                path_col = c
                break

    if path_col is None:
        raise ValueError(
            f"Could not find Local Path column in manifest. " 
            f"Columns: {list(manifest.columns)}"
        )

    local_path = str(manifest[path_col].iloc[0])

    if use_cache:
            _update_fits_cache(downloadpath, cache_key, local_path)
            if verbose:
                print(f"[cache write] {cache_key} → {local_path}")

    # 6) Read FITS → DataFrame → standardize  
    with fits.open(local_path, memmap=False) as hdul:
        table_hdu = None
        for hdu in hdul[1:]:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            # Prefer a named LIGHTCURVE extension if present; else first BinTable-like HDU
            extname = str(getattr(hdu, "name", "")).upper()
            if hasattr(data, "names") and extname in {
                "LIGHTCURVE", "LIGHTCURVES", "LC", "TIME_SERIES", "TIMESERIES"
            }:
                table_hdu = hdu
                break
            if table_hdu is None and hasattr(data, "names"):
                table_hdu = hdu
        
        if table_hdu is None:
            raise ValueError(
                f"No table-like FITS extension found in fike: {local_path}"
            )
        
        rec = np.array(table_hdu.data)

        # Ensure native endianness (handles some FITS tables cleanly)
        if hasattr(rec.dtype, "isnative") and not rec.dtype.isnative:
            try:
                rec = rec.byteswap().newbyteorder()
            except AttributeError:
                # numpy>=2 compatibility path for newbyteorder changes
                rec = rec.byteswap().view(rec.dtype.newbyteorder("="))

        raw_df = pd.DataFrame.from_records(rec)
        new_df = standardize_lc(new_df, pipeline)

    return product, raw_df, new_df

# Persist / reload results from collect_lightcurves_for_target
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

# NOTE: Generate N distinct colors for plots!
def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

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

# Full replacement: collect_lightcurves_for_target
# (only change: normalize_standardized_lc called after standardize_lc)
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

    # Validation
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

# Multi-pipeline wrapper 
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
    Download, standardize, and organize light curves from multiple TESS HLSP
    pipelines for one target star in one TESS sector.

    This is a convenience wrapper around `get_tess_lc(...)`. For each requested
    pipeline, the function attempts to:

    1. Search for the target in the requested sector and cadence
    2. Download the selected HLSP light-curve FITS product
    3. Read the raw light-curve table into a pandas DataFrame
    4. Standardize the raw table into the common schema used by this project
    5. Optionally create a quality-masked standardized light curve
       (`quality == 0`)

    The function never stops the full multi-pipeline run just because one
    pipeline fails. Instead, it records the failure and continues to the next
    pipeline. This makes it useful for large comparison runs where some
    pipelines may be missing data for a given TIC/sector combination.

    Parameters
    ----------
    tic_id : int or str
        TIC identifier of the target star. This is passed directly into
        `get_tess_lc(...)` and may be given as an integer or string.

    sector : int
        TESS sector to search. This wrapper is written for one sector at a
        time. If you want to run across many sectors, call this function inside
        a loop over sector numbers.

    pipelines : list of str
        List of pipeline names to try, for example:
        `["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]`.

        Each name is passed directly to `get_tess_lc(...)` and must match a
        pipeline name supported by your ingestion code.

    downloadpath : str, optional
        Directory where downloaded HLSP files should be stored.
        Defaults to `DEFAULT_DOWNLOADPATH`.

    radius : astropy.units.Quantity, optional
        Search radius passed to `get_tess_lc(...)`.
        Defaults to `DEFAULT_RADIUS`.

    exptime : str, optional
        Exposure-time / cadence selection passed to `get_tess_lc(...)`.
        Defaults to `DEFAULT_CADENCE`.

    apply_quality_mask : bool, optional
        If True, also create a masked standardized light curve using only rows
        with `quality == 0`.

        If False, the `"standardized_masked"` entry is set to None.

        This masking is applied only to the standardized light curve, not the
        raw light curve.

    verbose : bool, optional
        If True, print a short progress message before each pipeline fetch and
        allow `get_tess_lc(...)` to print its own selected-product summary.

    choose_first_timeseries : bool
        If true, prints the first time series of matching pipelines.

    Returns
    -------
    results : dict
        Dictionary keyed by pipeline name. Each entry is itself a dictionary
        with the following fields:

        - `"product"` :
            The single-row `lksearch.TESSSearch` object returned by
            `get_tess_lc(...)`, or None if the pipeline failed.

        - `"raw"` :
            Raw pandas DataFrame read directly from the FITS table, or None if
            the pipeline failed.

        - `"standardized"` :
            Standardized pandas DataFrame with the project schema
            (`time`, `flux_raw`, `flux_raw_err`, `flux_corr`,
            `flux_corr_err`, `flux_bkg`, `flux_bkg_err`, `quality`),
            or None if the pipeline failed.

        - `"standardized_masked"` :
            If `apply_quality_mask=True`, a filtered version of the
            standardized light curve containing only rows with `quality == 0`,
            with the index reset. If masking is disabled or the pipeline
            failed, this is None.

        - `"tic_id"` :
            The TIC identifier used for the query.

        - `"sector"` :
            The sector used for the query.

        - `"pipeline"` :
            The pipeline name for that entry.

        - `"n_raw"` :
            Number of rows in the raw light curve, or None if failed.

        - `"n_standardized"` :
            Number of rows in the standardized light curve, or None if failed.

        - `"n_masked"` :
            Number of rows in the masked standardized light curve if masking is
            enabled, otherwise None.

        - `"status"` :
            `"ok"` if the pipeline completed successfully, otherwise `"failed"`.

        - `"error"` :
            None if successful, otherwise the exception message as a string.

    Notes
    -----
    - This function is designed for bookkeeping across pipelines.
    - It does not compute vetting metrics by itself.
    - It is useful as the ingestion layer for building a unified results table
      with one row per `(tic_id, sector, pipeline)`.
    - Rows in `"standardized_masked"` are reset with `drop=True` so the masked
      table starts at index 0.

    Examples
    --------
    >>> pipelines = ["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]
    >>> results = collect_lightcurves_for_target(
    ...     tic_id=259377017,
    ...     sector=3,
    ...     pipelines=pipelines,
    ...     verbose=True
    ... )
    >>> results["QLP"]["status"]
    'ok'
    >>> results["QLP"]["standardized"].head()
    >>> results["QLP"]["standardized_masked"].head()

    Example of checking which pipelines succeeded:

    >>> for p in results:
    ...     print(p, results[p]["status"], results[p]["error"])

    Example of building one summary row per pipeline:

    >>> rows = []
    >>> for p, info in results.items():
    ...     rows.append({
    ...         "tic_id": info["tic_id"],
    ...         "sector": info["sector"],
    ...         "pipeline": info["pipeline"],
    ...         "status": info["status"],
    ...         "error": info["error"],
    ...         "n_raw": info["n_raw"],
    ...         "n_standardized": info["n_standardized"],
    ...         "n_masked": info["n_masked"],
    ...     })
    >>> summary_df = pd.DataFrame(rows)
    """
    results: Dict[str, Dict[str, Any]] = {}

    for pipeline in pipelines:
        try:
            if verbose:
                print(f"Fetching {pipeline} for TIC {tic_id}, sector {sector}")

            product, raw_df, std_df = get_tess_lc(
                TIC_ID=tic_id,
                pipeline=pipeline,
                radius=radius,
                exptime=exptime,
                Sector=sector,
                downloadpath=downloadpath,
                verbose=verbose,
            )

            if apply_quality_mask:
                if "quality" not in std_df.columns:
                    raise KeyError(
                        f"Standardized light curve for pipeline '{pipeline}' "
                        "does not contain a 'quality' column."
                    )
                std_masked_df = (
                    std_df.loc[std_df["quality"] == 0]
                    .reset_index(drop=True)
                )
            else:
                std_masked_df = None

            results[pipeline] = {
                "product": product,
                "raw": raw_df,
                "standardized": std_df,
                "standardized_masked": std_masked_df,
                "tic_id": tic_id,
                "sector": sector,
                "pipeline": pipeline,
                "n_raw": len(raw_df),
                "n_standardized": len(std_df),
                "n_masked": len(std_masked_df) if std_masked_df is not None else None,
                "status": "ok",
                "error": None,
            }

        except Exception as e:
            results[pipeline] = {
                "product": None,
                "raw": None,
                "standardized": None,
                "standardized_masked": None,
                "tic_id": tic_id,
                "sector": sector,
                "pipeline": pipeline,
                "n_raw": None,
                "n_standardized": None,
                "n_masked": None,
                "status": "failed",
                "error": str(e),
            }

    return results

# NOTE: NEW
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


######
def target_to_lightcurve_workflow_V2(
    target,
    pipelines,
    target_Sector,
    DEFAULT_RADIUS,
    DEFAULT_CADENCE,
    DEFAULT_DOWNLOADPATH,
    *,
    save_results: bool = True,
    results_savepath: str = None,
    force_redownload: bool = False,
):
    """
    Phase-fold and compare TESS HLSP light curves across multiple pipelines
    for a single target, with disk-backed caching of both FITS files and
    standardized DataFrames.

    On first call for a given (TIC, sector) pair, the function downloads all
    requested pipelines via ``collect_lightcurves_for_target`` and optionally
    persists the standardized results to Parquet on disk.  On subsequent
    calls, it attempts to load results from disk — skipping all network I/O —
    and only falls back to a live download if the saved data are not found.

    Parameters
    ----------
    target : pd.Series or single-row pd.DataFrame
        Row from the nearby TOI / M-dwarf catalog.  Must contain:
        ``'TIC ID'``, ``'Orbital Period (days) Value'``,
        ``'Orbital Epoch Value'``, ``'Transit Depth Value'``,
        ``'Transit Duration (hours) Value'``, and ``'Sectors'``.
    pipelines : list of str
        Pipeline names to compare (e.g. ``["QLP", "TESS-SPOC", "TGLC",
        "GSFC-ELEANOR-LITE"]``).
    target_Sector : int or None
        TESS sector to use.  If None, the first (earliest) sector listed in
        ``target['Sectors']`` is selected automatically.
    DEFAULT_RADIUS : astropy.units.Quantity
        Cone-search radius passed to ``collect_lightcurves_for_target``.
    DEFAULT_CADENCE : str
        Cadence/exptime string passed to ``collect_lightcurves_for_target``.
    DEFAULT_DOWNLOADPATH : str
        Root directory for FITS downloads and the FITS cache index.
    save_results : bool, optional
        If True (default), persist standardized DataFrames to Parquet after a
        live download run.  Has no effect when results are loaded from disk.
    results_savepath : str or None, optional
        Root directory for saved Parquet results.  Defaults to
        ``DEFAULT_DOWNLOADPATH + "/saved_results"`` if not provided.
    force_redownload : bool, optional
        If True, skip the disk cache entirely and re-run
        ``collect_lightcurves_for_target`` from scratch.  Useful when the
        on-disk data are stale or a pipeline has been updated.

    Returns
    -------
    sector_results : dict
        The results dict as returned by ``collect_lightcurves_for_target``
        (or reconstructed from disk by ``load_pipeline_results``).

    Examples
    --------
    >>> target = nearby_TOI_MD_df.loc[
    ...     nearby_TOI_MD_df['TIC ID'].astype(int) == 259377017
    ... ].reset_index(drop=True).iloc[0]
    >>> pipelines = ["QLP", "TESS-SPOC", "TGLC", "GSFC-ELEANOR-LITE"]
    >>> pipeline_colors = _get_colors(len(pipelines))
    >>> results = target_to_lightcurve_workflow(
    ...     target=target,
    ...     pipelines=pipelines,
    ...     target_Sector=None,
    ...     DEFAULT_RADIUS=DEFAULT_RADIUS,
    ...     DEFAULT_CADENCE=DEFAULT_CADENCE,
    ...     DEFAULT_DOWNLOADPATH=DEFAULT_DOWNLOADPATH,
    ... )
    """
    import time as clock
    import matplotlib.pyplot as plt
    t_start = clock.time()
    # ------------------------------------------------------------------ #
    #  Unpack target metadata                                             #
    # ------------------------------------------------------------------ #
    ID          = target['TIC ID'].item()
    target_P    = target['Orbital Period (days) Value'].item()
    target_T0   = target['Orbital Epoch Value'].item()
    target_Dep  = target['Transit Depth Value'].item() / 1e6
    target_Dur  = target['Transit Duration (hours) Value'].item()

    if target_Sector is None:
        try:
            target_Sector = np.min(
                list(map(int, target['Sectors'].to_list()[0].split(',')))
            )
        except AttributeError:
            target_Sector = np.min(
                list(map(int, target['Sectors'].split(',')))
            )

    if results_savepath is None:
        results_savepath = os.path.join(DEFAULT_DOWNLOADPATH, "saved_results")

    # ------------------------------------------------------------------ #
    #  Cache layer: try loading from disk first                           #
    # ------------------------------------------------------------------ #
    sector_results = None

    if not force_redownload:
        try:
            sector_results = load_pipeline_results(
                savepath=results_savepath,
                tic_id=ID,
                sector=target_Sector,
            )
            print(
                f"[cache hit] Loaded saved results for TIC {ID}, "
                f"sector {target_Sector} from {results_savepath}"
            )
        except FileNotFoundError:
            print(
                f"[cache miss] No saved results found for TIC {ID}, "
                f"sector {target_Sector} — running live download."
            )

    # ------------------------------------------------------------------ #
    #  Live download (first run, or force_redownload=True)               #
    # ------------------------------------------------------------------ #
    if sector_results is None:
        sector_results = collect_lightcurves_for_target(
            tic_id=ID,
            sector=target_Sector,
            pipelines=pipelines,
            downloadpath=DEFAULT_DOWNLOADPATH,
            radius=DEFAULT_RADIUS,
            exptime=DEFAULT_CADENCE,
            apply_quality_mask=True,
            verbose=True,
        )

        if save_results:
            try:
                outdir = save_pipeline_results(
                    results=sector_results,
                    savepath=results_savepath,
                    tic_id=ID,
                    sector=target_Sector,
                    overwrite=force_redownload,
                )
                print(f"[saved] Results written to {outdir}")
            except FileExistsError:
                # Results directory already exists and overwrite=False.
                # This branch is only reachable if save_results=True but
                # force_redownload=False and the saved dir exists without a
                # readable metadata.json (edge case: partial prior write).
                print(
                    "[warning] Could not save results — directory exists. "
                    "Pass force_redownload=True to overwrite."
                )

def timer(start,end,message):
    """
    Print the wall-clock runtime of a pipeline step in human-readable units.

    Automatically selects seconds, minutes, or hours based on the elapsed
    time so that log output is always legible regardless of step duration.
    Designed for inline use at the end of each named pipeline step.

    Parameters
    ----------
    start : float
        Start timestamp in seconds, as returned by ``time.time()``.
    end : float
        End timestamp in seconds, as returned by ``time.time()``.
    message : str
        Label printed before the runtime, e.g. ``'SAP took:'``.

    Returns
    -------
    None
        Prints to stdout; does not return a value.

    Notes
    -----
    Uses ``astropy.units`` for unit conversion.  Output format is:

        ``<message> <value> seconds|minutes|hours``

    The unit boundaries are: < 1 min → seconds; 1–60 min → minutes;
    ≥ 60 min → hours.

    Examples
    --------
    >>> import time
    >>> t0 = time.time()
    >>> # ... some computation ...
    >>> timer(t0, time.time(), 'SAP took:')
    SAP took: 4.231 seconds
    """
    runtime = (end-start)*u.second
    if runtime.to(u.minute) < 1*u.minute:
        print(message, np.round(runtime.value,3),'seconds \n')
    if (runtime.to(u.minute) >= 1*u.minute) & (runtime.to(u.minute) < 60*u.minute):
        print(message, np.round((runtime.to(u.minute)).value,3),'minutes \n')        
    if (runtime.to(u.minute) >= 60*u.minute):
        print(message, np.round((runtime.to(u.hour)).value,3),'hours \n')                
