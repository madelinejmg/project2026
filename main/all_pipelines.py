from TESS_pipeline import get_tess_LC
import astropy.units as u

def get_all_pipelines(
    TIC_ID,
    radius,
    exptime,
    Sector,
    pipelines,
    downloadpath,
):
    """
    Run get_tess_LC for multiple pipelines.

    Returns
    -------
    results : dict
        results[pipeline]["r"]      -> product object
        results[pipeline]["LC"]     -> raw LC DataFrame
        results[pipeline]["newLC"]  -> standardized LC DataFrame

    masked_results : dict
        masked_results[pipeline] -> newLC where Quality == 0
    """

    results = {}
    masked_results = {}

    for P in pipelines:
        r, LC, newLC = get_tess_LC(
            TIC_ID=TIC_ID,
            radius=radius,
            exptime=exptime,
            Sector=Sector,
            pipeline=P,
            downloadpath=downloadpath,
        )

        results[P] = {
            "r": r,
            "LC": LC,
            "newLC": newLC,
        }

        masked_results[P] = newLC[newLC["Quality"] == 0]

    return results, masked_results