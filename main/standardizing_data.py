from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Union, Tuple, Dict, List
import os
import re

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
import lksearch as lk


# helper function to get HLSP from TESS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# Default
DEFAULT_RADIUS = 3 * 21 * u.arcsec
DEFAULT_CADENCE = "30 minute"
DEFAULT_DOWNLOADPATH = os.getcwd() + "/HLSP1/"

SCHEMA_COLUMNS = [
    "time",
    "flux_raw",
    "flux_raw_err",
    "flux_corr",
    "flux_corr_err",
    "flux_bkg",
    "flux_bkg_err",
    "quality",
    "flux_use",
    "flux_use_err",
    "flux_norm",
    "flux_norm_err",
    "quality_mask",
]

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
        "flux_corr_err": None, # ASK ABOUT TASOC
        "flux_bkg": "FLUX_BKG",
        "flux_bkg_err": None,
        "quality": "QUALITY",
    },

    # IMPORTANT: These are placeholders only.
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

@dataclass
class LightCurveMeta:
    tic_id: Optional[int]
    sector: Optional[int]
    pipeline: str
    exptime: Optional[Union[int, float, str]]
    normalization_method: str = "median"
    used_corrected_flux: bool = True
    quality_mask_applied: bool = False
    sigma_clip_applied: bool = False
    sigma_clip_nsigma: Optional[float] = None
    n_input: int = 0
    n_output: int = 0
    notes: Optional[str] = None

# helper functions for the pipeline

# case sensitive
def _resolve_column(df: pd.DataFrame, name: Optional[str], case_insensitive: bool = True) -> Optional[str]:
    """
    Resolve a column name in a DataFrame, optionally ignoring case.

    This helper function attempts to locate a column in a pandas DataFrame
    using the provided name. If `case_insensitive` is True, the function
    searches for a column whose lowercase representation matches the
    lowercase representation of the requested name.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the light-curve data.

    name : str or None
        Expected column name from the pipeline mapping. If None, the
        function returns None.

    case_insensitive : bool, optional
        If True (default), match column names ignoring capitalization.

    Returns
    -------
    str or None
        The resolved column name present in the DataFrame. Returns None
        if the column is not found.
    """
    if name is None:
        return None

    if not case_insensitive:
        return name if name in df.columns else None

    lookup = {str(c).casefold(): c for c in df.columns}
    return lookup.get(str(name).casefold())


def _safe_array(df: pd.DataFrame, col: Optional[str], length: int) -> np.ndarray:
    """
    Safely extract a numeric array from a DataFrame column.

    This function retrieves values from a specified DataFrame column and
    converts them into a NumPy array suitable for numerical analysis.
    If the column does not exist or contains non-numeric values, the
    function returns an array of NaNs of the specified length.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the light-curve data.

    col : str or None
        Column name to extract. If None, an array of NaNs is returned.

    length : int
        Length of the output array.

    Returns
    -------
    numpy.ndarray
        Array containing the column values converted to numeric form.
        Non-numeric values are coerced to NaN.
    """
    if col is None:
        return np.full(length, np.nan)

    values = df[col].to_numpy(copy=False)

    # convert booleans / ints / strings to numeric if possible
    if pd.api.types.is_numeric_dtype(df[col]):
        return values

    return pd.to_numeric(df[col], errors="coerce").to_numpy()


def _nanmedian_positive(x: np.ndarray) -> float:
    """
    Compute the median of an array while safely ignoring invalid values.

    This function calculates the median of an array while ignoring NaN
    values. It is used primarily when normalizing light curves by their
    median flux. If the array contains no finite values, NaN is returned.

    Parameters
    ----------
    x : numpy.ndarray
        Input array of flux values.

    Returns
    -------
    float
        Median of the finite values in the array. Returns NaN if no
        valid values are present.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.nan
    return np.nanmedian(x[finite])


def standardize_lc(
    lc: pd.DataFrame,
    pipeline: str,
    *,
    tic_id: Optional[int] = None,
    sector: Optional[int] = None,
    exptime: Optional[Union[int, float, str]] = None,
    prefer_corrected: bool = True,
    apply_quality_mask: bool = False,
    strict: bool = False,
    case_insensitive: bool = True,
) -> Tuple[pd.DataFrame, LightCurveMeta]:
    """
    Convert a pipeline-specific light curve table into a standardize schema.
    """
    if pipeline not in PIPELINE_COLUMN_MAP:
        raise KeyError(f"Unknown pipeline '{pipeline}'. Supported: {sorted(PIPELINE_COLUMN_MAP)}")

    mapping = PIPELINE_COLUMN_MAP[pipeline]
    n = len(lc)

    meta = LightCurveMeta(
        tic_id=tic_id,
        sector=sector,
        pipeline=pipeline,
        exptime=exptime,
        n_input=n,
    )

    resolved = {}
    for key, source_col in mapping.items():
        actual = _resolve_column(lc, source_col, case_insensitive=case_insensitive)
        resolved[key] = actual
        if strict and source_col is not None and actual is None:
            raise KeyError(
                f"Missing required column '{source_col}' for pipeline '{pipeline}'. "
                f"Available columns: {list(lc.columns)}"
            )

    out = pd.DataFrame(index=lc.index)

    out["time"] = _safe_array(lc, resolved["time"], n)
    out["flux_raw"] = _safe_array(lc, resolved["flux_raw"], n)
    out["flux_raw_err"] = _safe_array(lc, resolved["flux_raw_err"], n)
    out["flux_corr"] = _safe_array(lc, resolved["flux_corr"], n)
    out["flux_corr_err"] = _safe_array(lc, resolved["flux_corr_err"], n)
    out["flux_bkg"] = _safe_array(lc, resolved["flux_bkg"], n)
    out["flux_bkg_err"] = _safe_array(lc, resolved["flux_bkg_err"], n)

    # keep quality integer-like if possible
    q = _safe_array(lc, resolved["quality"], n)
    out["quality"] = q

    # Decide which flux enters downstream analysis
    has_corr = np.isfinite(out["flux_corr"]).any()
    if prefer_corrected and has_corr:
        out["flux_use"] = out["flux_corr"]
        out["flux_use_err"] = out["flux_corr_err"]
        meta.used_corrected_flux = True
    else:
        out["flux_use"] = out["flux_raw"]
        out["flux_use_err"] = out["flux_raw_err"]
        meta.used_corrected_flux = False

    # Normalize relative flux
    norm_ref = _nanmedian_positive(out["flux_use"].to_numpy())
    if not np.isfinite(norm_ref) or norm_ref == 0:
        out["flux_norm"] = np.nan
        out["flux_norm_err"] = np.nan
        meta.notes = "Normalization failed because median science flux was invalid."
    else:
        out["flux_norm"] = out["flux_use"] / norm_ref
        out["flux_norm_err"] = out["flux_use_err"] / norm_ref

    # Quality mask convention: True = usable
    # Default assumption from your notebook: quality == 0 is good
    if np.isfinite(out["quality"]).any():
        out["quality_mask"] = (out["quality"].fillna(np.inf) == 0)
    else:
        out["quality_mask"] = True

    # Optional masking
    keep_mask = np.ones(n, dtype=bool)

    if apply_quality_mask:
        keep_mask &= out["quality_mask"].to_numpy(dtype=bool)
        meta.quality_mask_applied = True

    out = out.loc[keep_mask].reset_index(drop=True)
    meta.n_output = len(out)

    # Ensure column order
    out = out[SCHEMA_COLUMNS]

    return out, meta

def read_first_table_hdu(local_path: str) -> pd.DataFrame:
    with fits.open(local_path, memmap=False) as hdul:
        table_hdu = None

        for hdu in hdul[1:]:
            data = getattr(hdu, "data", None)
            if data is None:
                continue

            extname = str(getattr(hdu, "name", "")).upper()
            if hasattr(data, "names") and extname in {"LIGHTCURVE", "LIGHTCURVES", "LC", "TIME_SERIES", "TIMESERIES"}:
                table_hdu = hdu
                break

            if table_hdu is None and hasattr(data, "names"):
                table_hdu = hdu

        if table_hdu is None:
            raise ValueError(f"No table-like FITS extension found in file: {local_path}")

        rec = np.array(table_hdu.data)

        if hasattr(rec.dtype, "isnative") and not rec.dtype.isnative:
            try:
                rec = rec.byteswap().newbyteorder()
            except AttributeError:
                rec = rec.byteswap().view(rec.dtype.newbyteorder("="))

        return pd.DataFrame.from_records(rec)


def get_tess_lc(
    TIC_ID: Union[int, str],
    pipeline: str,
    radius: u.Quantity = DEFAULT_RADIUS,
    exptime: str = DEFAULT_CADENCE,
    Sector: Optional[Union[int, List[int]]] = None,
    downloadpath: str = DEFAULT_DOWNLOADPATH,
    *,
    prefer_corrected: bool = True,
    apply_quality_mask: bool = False,
    verbose: bool = True,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, LightCurveMeta]:
    """
    Download one HLSP light curve, read FITS table, and return both raw and standardized versions.
    """
    os.makedirs(downloadpath, exist_ok=True)

    tic_str = str(TIC_ID).strip()
    try:
        tic_str = str(int(float(tic_str)))
    except Exception:
        pass

    search_radius = float(radius) if isinstance(radius, (int, float, np.floating)) else radius

    search = lk.TESSSearch(
        target=f"TIC {tic_str}",
        search_radius=search_radius,
        exptime=exptime,
        sector=Sector,
        hlsp=True,
    )

    ts = getattr(search, "timeseries", search)
    filtered = ts.filter_table(mission="HLSP", pipeline=pipeline)

    if filtered.table is None or len(filtered.table) == 0:
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

    # prefer exact TIC match in target name
    if "target_name" in tbl.columns:
        pat = re.compile(rf"(^|\D){re.escape(tic_str)}(\D|$)")
        mask = tbl["target_name"].astype(str).apply(lambda s: bool(pat.search(s)))
        if mask.any():
            tbl = tbl[mask]

    if "distance" in tbl.columns:
        tbl = tbl.sort_values("distance", ascending=True)

    best_tbl = tbl.iloc[[0]].reset_index(drop=True)
    product = lk.TESSSearch(table=best_tbl)

    if verbose:
        cols = [c for c in ["target_name", "pipeline", "mission", "sector", "exptime", "distance", "year", "description"] if c in best_tbl.columns]
        print("Selected product row:")
        print(best_tbl[cols] if cols else best_tbl.head(1))

    manifest = product.download(download_dir=downloadpath)
    if not isinstance(manifest, pd.DataFrame) or len(manifest) == 0:
        raise ValueError("Download returned an empty manifest.")

    path_col = None
    for c in manifest.columns:
        canon = c.lower().replace(" ", "").replace("_", "")
        if canon == "localpath":
            path_col = c
            break

    if path_col is None:
        raise ValueError(f"Could not find Local Path column in manifest. Columns: {list(manifest.columns)}")

    local_path = str(manifest[path_col].iloc[0])

    raw_df = read_first_table_hdu(local_path)
    std_df, meta = standardize_lc(
        raw_df,
        pipeline=pipeline,
        tic_id=int(float(tic_str)),
        sector=Sector if isinstance(Sector, int) else None,
        exptime=exptime,
        prefer_corrected=prefer_corrected,
        apply_quality_mask=apply_quality_mask,
    )

    return product, raw_df, std_df, meta

# Multi-pipeline wrapper
def collect_lightcurves_for_target(
    tic_id: Union[int, str],
    sector: int,
    pipelines: List[str],
    downloadpath: str = DEFAULT_DOWNLOADPATH,
    *,
    radius: u.Quantity = DEFAULT_RADIUS,
    exptime: str = DEFAULT_CADENCE,
    prefer_corrected: bool = True,
    apply_quality_mask: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Collect standardized light curves from multiple pipelines for one TIC/sector.

    Returns
    -------
    results[pipeline] = {
        "product": ...,
        "raw": raw_df,
        "standardized": std_df,
        "meta": meta,
        "status": "ok" or "failed",
        "error": None or str
    }
    """
    results: Dict[str, Dict[str, Any]] = {}

    for pipeline in pipelines:
        try:
            if verbose:
                print(f"Fetching {pipeline} for TIC {tic_id}, sector {sector}")

            product, raw_df, std_df, meta = get_tess_lc(
                TIC_ID=tic_id,
                radius=radius,
                exptime=exptime,
                Sector=sector,
                pipeline=pipeline,
                downloadpath=downloadpath,
                prefer_corrected=prefer_corrected,
                apply_quality_mask=apply_quality_mask,
                verbose=verbose,
            )

            results[pipeline] = {
                "product": product,
                "raw": raw_df,
                "standardized": std_df,
                "meta": asdict(meta),
                "status": "ok",
                "error": None,
            }

        except Exception as e:
            results[pipeline] = {
                "product": None,
                "raw": None,
                "standardized": None,
                "meta": None,
                "status": "failed",
                "error": str(e),
            }

    return results