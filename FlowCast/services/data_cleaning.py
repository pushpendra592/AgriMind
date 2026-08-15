"""
services/data_cleaning.py - Physical-bounds cleaning for raw NASA POWER data.

Extracted here (rather than living in ablation_study.py or weather_service.py)
so that both can import it without creating circular dependencies.

Design rules:
  - No imports from other FlowCast services.
  - Rows are DROPPED, not clipped. Clipping piles spurious mass onto the cap
    value and distorts the target distribution.
  - Bounds are deliberately generous, not tight statistical cutoffs:
      EVLAND <= 3.0 mm/hr     real-world ET tops out ~1-1.5 mm/hr; 3.0 gives margin
      PRECTOTCORR <= 150 mm/hr  intense Punjab monsoon bursts can hit 50-100 mm/hr
"""

import pandas as pd


def clean_physical_outliers(
    df_raw: pd.DataFrame,
    evland_max: float = 3.0,
    prectotcorr_max: float = 150.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Drop rows with physically implausible EVLAND / PRECTOTCORR values.

    Call this on the raw DataFrame returned by fetch_nasa_data_parallel(),
    BEFORE preprocess_data() computes Kc multiplication or the target column.

    Args:
        df_raw:           Raw NASA POWER DataFrame.
        evland_max:       Upper bound for EVLAND (mm/hr). Default 3.0.
        prectotcorr_max:  Upper bound for PRECTOTCORR (mm/hr). Default 150.0.
        verbose:          Print a summary of rows dropped. Default True.

    Returns:
        Cleaned copy of df_raw with outlier rows removed.
    """
    before = len(df_raw)
    mask = pd.Series(True, index=df_raw.index)

    if "EVLAND" in df_raw.columns:
        bad_evland = df_raw["EVLAND"] > evland_max
        mask &= ~bad_evland
        if verbose and bad_evland.sum() > 0:
            print(
                f"  [clean] EVLAND > {evland_max}: "
                f"{bad_evland.sum()} rows ({100 * bad_evland.sum() / before:.3f}%)"
            )

    if "PRECTOTCORR" in df_raw.columns:
        bad_precip = df_raw["PRECTOTCORR"] > prectotcorr_max
        mask &= ~bad_precip
        if verbose and bad_precip.sum() > 0:
            print(
                f"  [clean] PRECTOTCORR > {prectotcorr_max}: "
                f"{bad_precip.sum()} rows ({100 * bad_precip.sum() / before:.3f}%)"
            )

    df_clean = df_raw[mask].copy()
    after = len(df_clean)

    if verbose:
        dropped = before - after
        if dropped:
            print(
                f"  [clean] Dropped {dropped} physically implausible rows "
                f"({100 * dropped / before:.3f}%) - {after} rows remain."
            )
        else:
            print(f"  [clean] No physically implausible rows found ({after} rows clean).")

    return df_clean