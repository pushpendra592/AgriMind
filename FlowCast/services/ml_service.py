"""
Machine Learning service for Agri-Smart AI.
Handles model training, prediction, and crop calculations.
"""

import datetime
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config import config, CROPS, CropConfig
from services.data_cleaning import clean_physical_outliers

class MLService:
    """Handles all machine learning operations."""
    
    def __init__(self):
        self.model: Optional[XGBRegressor] = None
        self.features = config.MODEL_FEATURES
        self.crop_name: Optional[str] = None
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        crop_name: str,
        lat: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Preprocess weather data for model training.

        Args:
            df:        Raw weather DataFrame (straight from fetch_nasa_data_parallel).
            crop_name: Name of the crop.
            lat:       Latitude of the location.  Used for hemisphere-aware sowing
                       date calculation.  Pass None to assume Northern Hemisphere
                       (preserves backward compatibility).

        Returns:
            Preprocessed DataFrame with features and target column.
        """
        if crop_name not in CROPS:
            raise ValueError(f"Unknown crop: {crop_name}")

        crop = CROPS[crop_name]
        self.lat = lat  # store for downstream use (e.g. calculate_current_kc)

        # ── 1. Physical-bounds cleaning ──────────────────────────────────────
        # Raw EVLAND/PRECTOTCORR must be filtered BEFORE Kc multiplication,
        # otherwise a single bad row can produce a target value orders of
        # magnitude above the physical range and silently dominate RMSE.
        # verbose=False because weather_service already prints the summary when
        # data comes through the normal fetch path; this call is a silent safety
        # net for any caller that passes raw data directly to preprocess_data.
        df = clean_physical_outliers(df, verbose=False)

        df = df.copy()
        southern = (lat is not None) and (lat < 0)

        # ── 2. Kc coefficient ────────────────────────────────────────────────
        # Track silently-swallowed errors so we can surface them.
        _kc_error_count = [0]

        def _kc_with_counter(row):
            val, had_error = self._calculate_kc_for_row(row, crop, southern)
            if had_error:
                _kc_error_count[0] += 1
            return val

        df['Kc'] = df.apply(_kc_with_counter, axis=1)

        if _kc_error_count[0] > 0:
            print(
                f"  [warn] _calculate_kc_for_row raised {_kc_error_count[0]} "
                f"date errors ({100 * _kc_error_count[0] / len(df):.2f}% of rows). "
                f"Those rows will have Kc=0 and be excluded from training. "
                f"Check YEAR/MO/DY columns for malformed values."
            )

        # ── 3. Target column ─────────────────────────────────────────────────
        df['Irrigation_Need'] = (
            (df['EVLAND'] * df['Kc']) - df['PRECTOTCORR']
        ).clip(lower=0)

        # ── 4. Lag features ──────────────────────────────────────────────────
        df['Rain_Last_24h'] = df['PRECTOTCORR'].rolling(
            window=24, min_periods=1
        ).sum()
        df['Soil_Moisture_Prev_1h'] = df['GWETTOP'].shift(1)
        df['Evap_Prev_1h'] = df['EVLAND'].shift(1)

        # ── 5. Growing-season filter ─────────────────────────────────────────
        train_df = df[df['Kc'] > 0].copy()
        train_df.dropna(inplace=True)

        print(f"  Preprocessed {len(train_df):,} training samples for {crop_name}")

        # ── 6. Target distribution summary ────────────────────────────────────
        # Checks whether the target distribution is physically plausible before
        # any model sees it.  A strongly right-skewed distribution (skew > 1)
        # means RMSE will be dominated by a small number of extreme rows —
        # look at MAE (reported by train_model) for a more representative error.
        if not train_df.empty:
            y = train_df['Irrigation_Need']
            p99 = y.quantile(0.99)
            n_extreme = int((y > p99).sum())
            skew = float(y.skew())
            print(
                f"  [target] mean={y.mean():.4f}  std={y.std():.4f}  "
                f"max={y.max():.4f}  p99={p99:.4f}  skew={skew:.2f}"
            )
            if skew > 1.0:
                print(
                    f"  [target] Skew={skew:.2f} > 1 — RMSE will be outlier-dominated. "
                    f"({n_extreme} rows above p99={p99:.4f}). "
                    f"Use MAE from train_model for a more representative error estimate."
                )

        return train_df
    
    def _calculate_kc_for_row(
        self,
        row: pd.Series,
        crop: CropConfig,
        southern_hemisphere: bool = False,
    ) -> Tuple[float, bool]:
        """
        Calculate Kc coefficient for a single data row.

        Args:
            row:                  A single row from the NASA POWER DataFrame.
            crop:                 CropConfig for the selected crop.
            southern_hemisphere:  If True, flip the sowing-year logic so that a
                                  crop sown in November (e.g. Australian wheat)
                                  is correctly assigned to the current year rather
                                  than the previous one.

        Returns:
            (kc_value, had_error) — had_error is True only when a genuine date
            parsing problem occurred (not just an off-season row).
        """
        try:
            curr_date = datetime.date(
                int(row['YEAR']),
                int(row['MO']),
                int(row['DY'])
            )

            # ── Hemisphere-aware sowing year ─────────────────────────────────
            # Northern Hemisphere (default): a Jan wheat row was sown the
            # previous November, so sowing_year = curr_year - 1 when
            # curr_month < sowing_month.
            #
            # Southern Hemisphere: seasons are inverted. A Nov sowing date
            # means the crop was sown *this* year when curr_month >= sowing_month,
            # or next year otherwise — i.e. the opposite of NH logic.
            if not southern_hemisphere:
                sowing_year = (
                    curr_date.year - 1
                    if curr_date.month < crop.sowing_month
                    else curr_date.year
                )
            else:
                sowing_year = (
                    curr_date.year
                    if curr_date.month >= crop.sowing_month
                    else curr_date.year - 1
                )

            sowing_date = datetime.date(sowing_year, crop.sowing_month, crop.sowing_day)
            days_grown = (curr_date - sowing_date).days

            # Off-season: not an error, just not growing
            if days_grown < 0 or days_grown > crop.total_duration:
                return 0.0, False

            # Find current growth stage
            cum_days = 0
            for i, stage_duration in enumerate(crop.stage_days):
                cum_days += stage_duration
                if days_grown <= cum_days:
                    return crop.kc[i], False

            return 0.0, False

        except (ValueError, OverflowError, KeyError):
            # Only genuine date/key problems reach here — off-season rows
            # are handled above.  Return had_error=True so the caller can
            # accumulate a count and surface it rather than swallowing silently.
            return 0.0, True
    
    def train_model(
        self, 
        df: pd.DataFrame,
        test_size: float = 0.2,
        **model_params
    ) -> Dict[str, Any]:
        """
        Train XGBoost regression model.
        
        Args:
            df: Preprocessed training DataFrame
            test_size: Fraction of data for testing
            **model_params: Additional XGBoost parameters
            
        Returns:
            Dictionary with model metrics
        """
        from sklearn.metrics import mean_absolute_error

        # Default model parameters
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_jobs': -1
        }
        default_params.update(model_params)
        
        # Prepare features and target
        X = df[self.features]
        y = df['Irrigation_Need']
        
        # Train-test split (preserve time order).
        # random_state is set even though shuffle=False so that any future
        # change to shuffle=True doesn't silently produce non-reproducible metrics.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=False,
            random_state=42,
        )
        
        # Train model
        self.model = XGBRegressor(**default_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae  = float(mean_absolute_error(y_test, predictions))
        r2   = r2_score(y_test, predictions)
        
        metrics = {
            'rmse': rmse,
            'mae':  mae,
            'r2':   r2,
            'feature_importance': dict(zip(
                self.features,
                self.model.feature_importances_
            ))
        }
        
        print(f"  Model trained  │  RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")

        # ── Walk-forward cross-validation ──────────────────────────────────────────
        # Confirms whether the single 80/20 split result is stable across all
        # year boundaries, or a lucky cut.  std(R²) < 0.05 = stable.
        cv_metrics = self._walk_forward_cv(df, default_params, mean_absolute_error)
        metrics['walk_forward_cv'] = cv_metrics

        # ── Kc sensitivity check ─────────────────────────────────────────────
        # Hold every other feature at its training-set median, sweep Kc across
        # all observed values, and check that predicted irrigation rises with Kc.
        # A flat or non-monotonic curve means the model is not learning crop-stage
        # scaling despite Kc appearing in the feature importance list.
        if 'Kc' in self.features:
            median_row = X_train.median()
            kc_values  = sorted(df['Kc'].unique())
            kc_preds   = []
            for kc_val in kc_values:
                row = median_row.copy()
                row['Kc'] = kc_val
                pred = float(self.model.predict(pd.DataFrame([row]))[0])
                kc_preds.append((kc_val, round(pred, 4)))

            # Check monotonicity: count inversions (Kc rises but prediction drops)
            inversions = sum(
                1 for i in range(1, len(kc_preds))
                if kc_preds[i][1] < kc_preds[i - 1][1]
            )
            kc_str = "  ".join(f"Kc={k:.2f}→{p:.3f}" for k, p in kc_preds)
            print(f"  [Kc check] {kc_str}")
            if inversions > 0:
                print(
                    f"  [Kc check] WARNING: {inversions} non-monotonic step(s) detected. "
                    f"Model may not be learning crop-stage scaling correctly."
                )
            else:
                print(f"  [Kc check] Monotonic ✔ — model correctly scales prediction with crop stage.")

            metrics['kc_sensitivity'] = kc_preds

        return metrics

    def _walk_forward_cv(
        self,
        df: pd.DataFrame,
        model_params: dict,
        mae_fn,
        min_train_years: int = 2,
    ) -> Dict[str, Any]:
        """
        Walk-forward cross-validation across every year boundary in the dataset.

        For each test year Y (from the 3rd year onward), trains on all rows
        where YEAR < Y and tests on rows where YEAR == Y.  This forces the
        model to prove it generalises across every seasonal boundary in the
        data, not just the single 80/20 cut used by train_model.

        Decision rule:
          std(R²) < 0.05  →  single-split number is stable, trust it.
          std(R²) ≥ 0.05  →  result is split-sensitive; treat single-split R²
                             as an upper bound, not a reliable estimate.

        Args:
            df:              Preprocessed DataFrame (must contain a YEAR column).
            model_params:    XGBRegressor kwargs (same as used by train_model).
            mae_fn:          mean_absolute_error callable (passed in to avoid
                             re-importing inside the loop).
            min_train_years: Minimum number of years required in the training
                             window before the first fold is evaluated.

        Returns:
            Dict with keys: folds (list of per-fold dicts), mean_r2, std_r2,
            mean_rmse, mean_mae, n_folds.
        """
        if 'YEAR' not in df.columns:
            print("  [CV] YEAR column not found — skipping walk-forward CV.")
            return {}

        years      = sorted(df['YEAR'].unique())
        X_all      = df[self.features]
        y_all      = df['Irrigation_Need']
        folds      = []
        skipped    = 0

        for i, test_year in enumerate(years):
            train_years = years[:i]          # all years strictly before test_year
            if len(train_years) < min_train_years:
                skipped += 1
                continue

            train_mask = df['YEAR'].isin(train_years)
            test_mask  = df['YEAR'] == test_year

            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            X_te, y_te = X_all[test_mask],  y_all[test_mask]

            if len(X_te) == 0:
                skipped += 1
                continue

            m = XGBRegressor(**model_params)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)

            fold_r2   = float(r2_score(y_te, preds))
            fold_rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
            fold_mae  = float(mae_fn(y_te, preds))
            folds.append({
                'test_year': int(test_year),
                'n_train':   int(train_mask.sum()),
                'n_test':    int(test_mask.sum()),
                'r2':        round(fold_r2,   4),
                'rmse':      round(fold_rmse, 4),
                'mae':       round(fold_mae,  4),
            })

        if not folds:
            print("  [CV] Not enough yearly data for walk-forward CV (need ≥ 3 years).")
            return {}

        r2_vals   = [f['r2']   for f in folds]
        rmse_vals = [f['rmse'] for f in folds]
        mae_vals  = [f['mae']  for f in folds]
        mean_r2   = round(float(np.mean(r2_vals)),   4)
        std_r2    = round(float(np.std(r2_vals)),    4)
        mean_rmse = round(float(np.mean(rmse_vals)), 4)
        mean_mae  = round(float(np.mean(mae_vals)),  4)

        # ── Print per-fold table ───────────────────────────────────────────────
        print(f"\n  [CV] Walk-forward cross-validation  ({len(folds)} folds, {skipped} skipped)")
        print(f"  {'Year':>6}  {'Train rows':>10}  {'Test rows':>9}  {'R²':>7}  {'RMSE':>8}  {'MAE':>8}")
        print(f"  {'':->6}  {'':->10}  {'':->9}  {'':->7}  {'':->8}  {'':->8}")
        for f in folds:
            print(
                f"  {f['test_year']:>6}  {f['n_train']:>10,}  {f['n_test']:>9,}  "
                f"{f['r2']:>7.4f}  {f['rmse']:>8.4f}  {f['mae']:>8.4f}"
            )
        print(f"  {'':->6}  {'':->10}  {'':->9}  {'':->7}  {'':->8}  {'':->8}")
        print(f"  {'MEAN':>6}  {'':>10}  {'':>9}  {mean_r2:>7.4f}  {mean_rmse:>8.4f}  {mean_mae:>8.4f}")
        print(f"  {'STD':>6}  {'':>10}  {'':>9}  {std_r2:>7.4f}")

        # ── Stability verdict ─────────────────────────────────────────────────
        if std_r2 < 0.05:
            print(
                f"  [CV] ✔ std(R²)={std_r2:.4f} < 0.05 — single-split R²={r2_score(y_all, self.model.predict(X_all)):.4f} "
                f"is stable across year boundaries. Trust it."
            )
        else:
            print(
                f"  [CV] ⚠ std(R²)={std_r2:.4f} ≥ 0.05 — result is split-sensitive. "
                f"Use CV mean R²={mean_r2:.4f} as the reliable estimate, not the single-split number."
            )

        return {
            'folds':     folds,
            'mean_r2':   mean_r2,
            'std_r2':    std_r2,
            'mean_rmse': mean_rmse,
            'mean_mae':  mean_mae,
            'n_folds':   len(folds),
        }

    def predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            input_data: DataFrame with features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        return self.model.predict(input_data[self.features])
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        if self.model is None:
            return None
        
        return dict(zip(self.features, self.model.feature_importances_))


def calculate_current_kc(
    crop_name: str,
    lat: Optional[float] = None,
) -> Tuple[float, str, int, int]:
    """
    Calculate the current Kc coefficient for a crop based on today's date.

    Args:
        crop_name: Name of the crop.
        lat:       Latitude of the location. Used for hemisphere-aware sowing
                   year calculation. Pass None to assume Northern Hemisphere.

    Returns:
        Tuple of (kc_value, stage_name, days_grown, total_duration).
    """
    if crop_name not in CROPS:
        return 0.0, "Unknown", 0, 0

    crop = CROPS[crop_name]
    today = datetime.date.today()
    southern = (lat is not None) and (lat < 0)

    # ── Hemisphere-aware sowing year ─────────────────────────────────────────
    if not southern:
        sowing_year = (
            today.year - 1
            if today.month < crop.sowing_month
            else today.year
        )
    else:
        sowing_year = (
            today.year
            if today.month >= crop.sowing_month
            else today.year - 1
        )

    sowing_date = datetime.date(sowing_year, crop.sowing_month, crop.sowing_day)
    days_grown = (today - sowing_date).days
    total_duration = crop.total_duration

    if days_grown < 0:
        return 0.0, "Pre-Sowing", days_grown, total_duration
    elif days_grown > total_duration:
        return 0.0, "Harvested", days_grown, total_duration

    cum_days = 0
    for i, stage_duration in enumerate(crop.stage_days):
        cum_days += stage_duration
        if days_grown <= cum_days:
            return crop.kc[i], crop.stage_names[i], days_grown, total_duration

    return 0.0, "Harvested", days_grown, total_duration


def preprocess_data(df: pd.DataFrame, crop_name: str) -> pd.DataFrame:
    """Convenience function for preprocessing."""
    service = MLService()
    return service.preprocess_data(df, crop_name)