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

def get_tess_lc(
    TIC_ID: Union[int, str],
    pipeline: str,
    radius: u.Quantity = DEFAULT_RADIUS,
    exptime: str = DEFAULT_CADENCE,
    Sector: Optional[Union[int, list[int]]] = None,
    downloadpath: str = DEFAULT_DOWNLOADPATH,
    *,
    verbose: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    """
    Download one TESS HLSP light curve (FITS) via lksearch and return it as a DataFrame.

    Parameters
    ----------
    TIC_ID : int | str
        TIC identifier (e.g., 123456789).
    radius : float | astropy.units.Quantity
        Search radius. If float, lksearch interprets it as arcseconds. :contentReference[oaicite:4]{index=4}
        You may also pass an explicit Quantity (e.g., 30*u.arcsec).
    exptime : str | int | (float, float)
        Exposure time filter passed to lksearch (e.g. "shortest", 120, (100, 500)). :contentReference[oaicite:5]{index=5}
    Sector : int | list[int] | None
        TESS sector(s) to filter on.
    pipeline : str
        HLSP pipeline name to filter on (e.g., "QLP", "TASOC", "TESS-SPOC", etc.). :contentReference[oaicite:6]{index=6}
    downloadpath : str
        Directory where products will be downloaded.
    verbose : bool, optional (keyword-only)
        If True, prints a compact summary of matching pipelines and the selected product.

    Returns
    -------
    product : lksearch.TESSSearch (single-row)
        A TESSSearch object containing exactly one selected product row.
    df : pandas.DataFrame
        Light curve table from the first table-like FITS extension.

    Raises
    ------
    ValueError
        If no matching HLSP timeseries products are found for the requested pipeline.
    """
    os.makedirs(downloadpath, exist_ok=True)

    tic_str = str(TIC_ID).strip()
    # Normalize "123.0" -> "123" if user passed a float-like string
    try:
        tic_str = str(int(float(tic_str)))
    except Exception:
        pass
    
    # lksearch treats float search_radius as arcseconds by default. :contentReference[oaicite:7]{index=7}
    search_radius = float(radius) if isinstance(radius, (int, float, np.floating)) else radius

    # 1) Query
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
        # Very defensive fallback; docs/tutorials show .timeseries exists. :contentReference[oaicite:8]{index=8}
        ts = search
    
    # Filter to HLSP + pipeline (and keep exptime/sector constraints via ctor inputs)
    filtered = ts.filter_table(mission="HLSP", pipeline=pipeline)

    if filtered.table is None or len(filtered.table) == 0:
        # Build helpful debug context from whatever we *did* get back
        table = getattr(ts, "table", None)
        if isinstance(table, pd.DataFrame) and len(table) > 0:
            if "mission" in table.columns:
                hlsp_tbl = table[table["mission"].astype(str).eq("HLSP")]
            else:
                hlsp_tbl = table

            avail = (
                np.unique(hlsp_tbl["pipeline"].astype(str))
                if "pipeline" in hlsp_tbl.columns and len(hlsp_tbl) > 0
                else np.array([])
            )
            raise ValueError(
                f"No HLSP timeseries product found for pipeline='{pipeline}' "
                f"(TIC={tic_str}, sector={Sector}, exptime={exptime}, radius={radius}). "
                f"Available HLSP pipelines: {avail.tolist()}"
            )

        raise ValueError(
            f"No products returned at all for TIC={tic_str}, sector={Sector}, exptime={exptime}, radius={radius}."
        )

    tbl = filtered.table.copy()

    # 3) Pick best row:
    #    (a) prefer rows whose target_name contains the TIC id
    #    (b) then smallest distance (closest on-sky match)
    if "target_name" in tbl.columns:
        pat = re.compile(rf"(^|\D){re.escape(tic_str)}(\D|$)")
        mask = tbl["target_name"].astype(str).apply(lambda s: bool(pat.search(s)))
        if mask.any():
            tbl = tbl[mask]

    if "distance" in tbl.columns:
        tbl = tbl.sort_values("distance", ascending=True)

    best_tbl = tbl.iloc[[0]].reset_index(drop=True)

    # Create a single-row TESSSearch object so download() only pulls one file.
    product = lk.TESSSearch(table=best_tbl)

    if verbose:
        cols = [c for c in ["target_name", "pipeline", "mission", "sector", "exptime", "distance", "year", "description"] if c in best_tbl.columns]
        print("Selected product row:")
        print(best_tbl[cols] if cols else best_tbl.head(1))

    # 4) Download
    manifest = product.download(download_dir=downloadpath)

    # 5) Extract local path robustly
    if not isinstance(manifest, pd.DataFrame) or len(manifest) == 0:
        raise ValueError("Download returned an empty manifest; nothing was downloaded.")

    # lksearch manifest uses 'Local Path' in tutorials. :contentReference[oaicite:9]{index=9}
    path_col = None
    for c in manifest.columns:
        canon = c.lower().replace(" ", "").replace("_", "")
        if canon in {"localpath"}:
            path_col = c
            break

    if path_col is None:
        raise ValueError(f"Could not find Local Path column in manifest. Columns: {list(manifest.columns)}")

    local_path = str(manifest[path_col].iloc[0])

    # 6) Read FITS, grab first table-like extension, convert to DataFrame
    with fits.open(local_path, memmap=False) as hdul:
        table_hdu = None
        for hdu in hdul[1:]:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            # Prefer a named LIGHTCURVE extension if present; else first BinTable-like HDU
            extname = str(getattr(hdu, "name", "")).upper()
            if hasattr(data, "names") and extname in {"LIGHTCURVE, LIGHTCURVES", "LC", "TIME_SERIES", "TIMESERIES"}:
                table_hdu = hdu
                break
            if table_hdu is None and hasattr(data, "names"):
                table_hdu = hdu
        
        if table_hdu is None:
            raise ValueError(f"No table-like FITS extension found in fike: {local_path}")
        
        rec = np.array(table_hdu.data)

        # Ensure native endianness (handles some FITS tables cleanly)
        if hasattr(rec.dtype, "isnative") and not rec.dtype.isnative:
            try:
                rec = rec.byteswap().newbyteorder()
            except AttributeError:
                # numpy>=2 compatibility path for newbyteorder changes
                rec = rec.byteswap().view(rec.dtype.newbyteorder("="))

        df = pd.DataFrame.from_records(rec)
        
        newdf = standardize_lc(df,pipeline)

    return product, df, newdf


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