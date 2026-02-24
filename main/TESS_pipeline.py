from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import os
import re

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
import lksearch as lk

from typing import Dict, Iterable, Optional, Sequence, Tuple
import matplotlib.pyplot as plt

# helper function to get HLSP from TESS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Practice with TOI-270 (3 planet system)
ID= 259377017
Sector=3
radius=3*21*u.arcsec
cadence='30 minute'
DOWNLOADPATH = os.getcwd() + "/HLSP/"

cadence_map = {
        'long': (30*u.minute , 'FFI'),
        '30 minute': (30*u.minute , 'FFI'),
        '10 minute': (10*u.minute , 'FFI'),
        'short': ( 2*u.minute , 'TPF'),
        '2 minute': ( 2*u.minute , 'TPF'),
        '20 second': (20*u.second, 'TPF'),
        'fast': (20*u.second, 'TPF') }

if cadence not in cadence_map:
    raise ValueError(f"Unrecognised cadence: {cadence!r}")
exp_u, ffi_or_tpf = cadence_map[cadence]
exptime     = int(exp_u.to(u.second).value)

def standardize_lc(
    lc: pd.DataFrame,
    pipeline: str,
    *,
    strict: bool = False,
    case_insensitive: bool = True,
) -> pd.DataFrame:
    """Standardize a light-curve DataFrame from multiple TESS HLSP pipelines.

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
    lc_map: Dict[str, Tuple[Optional[str], ...]] = {
        "QLP": (
            "TIME",
            "SAP_FLUX",
            "KSPSAP_FLUX_ERR",
            "SAP_BKG",
            "SAP_BKG_ERR",
            "KSPSAP_FLUX",
            "KSPSAP_FLUX_ERR",
            "QUALITY",
        ),
        "TESS-SPOC": (
            "TIME",
            "SAP_FLUX",
            "SAP_FLUX_ERR",
            "SAP_BKG",
            "SAP_BKG_ERR",
            "PDCSAP_FLUX",
            "PDCSAP_FLUX_ERR",
            "QUALITY",
        ),
        "TGLC": (
            "time",
            "aperture_flux",
            None,
            "background",
            None,
            "cal_aper_flux",
            None,
            "TGLC_flags",
        ),
        "GSFC-ELEANOR-LITE": (
            "TIME",
            "RAW_FLUX",
            "FLUX_ERR",
            "FLUX_BKG",
            None,
            "PCA_FLUX",
            "FLUX_ERR",
            "QUALITY",
        ),
    }

    standard_cols: Tuple[str, ...] = (
        "Time",
        "Raw Flux",
        "Raw Flux Error",
        "BKG Flux",
        "BKG Flux Error",
        "Corrected Flux",
        "Corrected Flux Error",
        "Quality",
    )

    if pipeline not in lc_map:
        raise KeyError(
            f"Unknown pipeline='{pipeline}'. Supported: {sorted(lc_map.keys())}"
        )

    old_cols = lc_map[pipeline]
    if len(old_cols) != len(standard_cols):
        raise ValueError(
            f"Mapping for pipeline='{pipeline}' has {len(old_cols)} columns, "
            f"expected {len(standard_cols)}."
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

    for old_name, new_name in zip(old_cols, standard_cols):
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

downloadpath = DOWNLOADPATH
os.makedirs(downloadpath, exist_ok=True)

def get_tess_LC(
    TIC_ID: Union[int, str],
    radius: Union[float, u.Quantity],
    exptime: Union[str, int, Tuple[float, float]],
    Sector: Optional[Union[int, list[int]]],
    pipeline: str,
    *,
    verbose: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    """Download one TESS HLSP light curve (FITS) via lksearch and return it as a DataFrame.

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
    except AttributeError:
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
            msg = (
                f"No HLSP timeseries products found for pipeline='{pipeline}' "
                f"(TIC={tic_str}, sector={Sector}, exptime={exptime}, radius={radius}). "
                f"Available HLSP pipelines in this search: {avail.tolist()}"
            )
        else:
            msg = (
                f"No products returned at all (TIC={tic_str}, sector={Sector}, "
                f"exptime={exptime}, radius={radius})."
            )
        raise ValueError(msg)

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
        raise ValueError(f"Could not find a Local Path column in download manifest. Columns: {list(manifest.columns)}")

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
            if hasattr(data, "names") and extname in {"LIGHTCURVE", "LIGHTCURVES", "LC", "TIME_SERIES", "TIMESERIES"}:
                table_hdu = hdu
                break
            if table_hdu is None and hasattr(data, "names"):
                table_hdu = hdu

        if table_hdu is None:
            raise ValueError(f"No table-like FITS extension found in file: {local_path}")

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

