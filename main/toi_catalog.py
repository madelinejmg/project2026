import pandas as pd

# hardcoded absolute path
code_path = "/Users/madelinejmg/Desktop/PHYS39002/research2026/Catalog/nearby_TOI_MDs.csv"

def load_catalog(csv_path=code_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
    return df

def get_target_by_tic(
    df: pd.DataFrame,
    tic_id: int,
    sector: int | None = None
) -> pd.DataFrame:
    """
    Return ALL rows for a TIC ID.
    If sector is provided, return all rows where that sector appears
    inside the comma-separated 'Sectors' string.
    """

    q = df[df["TIC ID"] == int(tic_id)].copy()

    if q.empty:
        raise ValueError(f"TIC ID {tic_id} not found.")

    if sector is not None:
        def row_has_sector(cell):
            if pd.isna(cell):
                return False
            for token in str(cell).split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    if int(float(token)) == int(sector):
                        return True
                except ValueError:
                    continue
            return False

        q = q[q["Sectors"].apply(row_has_sector)].copy()

        if q.empty:
            raise ValueError(f"Sector {sector} not found for TIC {tic_id}.")

    q["depth_frac"] = q["Transit Depth Value"] / 1e6
    q["pipelines"] = q["Detection Pipeline(s)"].apply(
        lambda x: [p.strip() for p in str(x).split(",") if p.strip()]
    )

    q["sectors"] = q["Sectors"].astype(str).str.replace(" ", "", regex=False)
    q["n_sectors"] = q["sectors"].apply(lambda s: len([t for t in s.split(",") if t]))


    out = q[[
        "TIC ID",
        "sectors",
        "n_sectors",
        "pipelines",
        "Orbital Period (days) Value",
        "Orbital Epoch Value",
        "depth_frac",
        "Transit Duration (hours) Value",
    ]].rename(columns={
        "TIC ID": "tic_id",
        "Orbital Period (days) Value": "P_days",
        "Orbital Epoch Value": "T0",
        "Transit Duration (hours) Value": "duration_hr",
    }).reset_index(drop=True)

    return out