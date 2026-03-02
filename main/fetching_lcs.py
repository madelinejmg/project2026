import lightkurve as lk

def fetch_tess_ffi_lcs(target, sector_min=1, sector_max=26, 
                       authors=("SPOC", "QLP", "GSFC-ELEANOR-LITE", "TGLC"),
                       exptime="long", mission="TESS"):
    """
    target: TIC ID (e.g., 'TIC 123...') or coordinates string.
    exptime='long' is typically the ~30-min cadence in prime mission FFIs.
    Returns: dict author -> LightCurveCollection (downloaded + cached)
    """
    out = {}
    for author in authors:
        sr = lk.search_lightcurve(
            target=target,
            mission="TESS",
            author=author,
            exptime=exptime
        )

        # keep only sectors 1–26
        if len(sr) == 0:
            out[author] = None
            continue

        sr = sr[sr.table["sequence_number"] >= sector_min]
        sr = sr[sr.table["sequence_number"] <= sector_max]

        if len(sr) == 0:
            out[author] = None
            continue

        out[author] = sr.download_all()  # cached in ~/.lightkurve-cache by default

    return out

if __name__ == "__main__":
    lcs = fetch_tess_ffi_lcs("TIC 302105449")

    print(lcs.keys())
    print({k: (None if v is None else len(v)) for k, v in lcs.items()})