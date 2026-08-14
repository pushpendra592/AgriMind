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
from sklearn.metrics import mean_squared_error

from config import config, CROPS, CropConfig

class MLService:
    """Handles all machine learning operations."""
    
    def __init__(self):
        self.model: Optional[XGBRegressor] = None
        self.features = config.MODEL_FEATURES
        self.crop_name: Optional[str] = None
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        crop_name: str
    ) -> pd.DataFrame:
        """
        Preprocess weather data for model training.
        
        Args:
            df: Raw weather DataFrame
            crop_name: Name of the crop
            
        Returns:
            Preprocessed DataFrame with features
        """
        if crop_name not in CROPS:
            raise ValueError(f"Unknown crop: {crop_name}")
        
        crop = CROPS[crop_name]
        df = df.copy()
        
        # Calculate Kc coefficient for each row
        df['Kc'] = df.apply(
            lambda row: self._calculate_kc_for_row(row, crop), 
            axis=1
        )
        
        # Calculate irrigation need
        df['Irrigation_Need'] = (
            (df['EVLAND'] * df['Kc']) - df['PRECTOTCORR']
        ).clip(lower=0)
        
        # Create lag features
        df['Rain_Last_24h'] = df['PRECTOTCORR'].rolling(
            window=24, min_periods=1
        ).sum()
        df['Soil_Moisture_Prev_1h'] = df['GWETTOP'].shift(1)
        df['Evap_Prev_1h'] = df['EVLAND'].shift(1)
        
        # Filter to growing season only
        train_df = df[df['Kc'] > 0].copy()
        train_df.dropna(inplace=True)
        
        print(f"Preprocessed {len(train_df)} training samples for {crop_name}")
        
        return train_df
    
    def _calculate_kc_for_row(
        self, 
        row: pd.Series, 
        crop: CropConfig
    ) -> float:
        """Calculate Kc coefficient for a single row."""
        try:
            curr_date = datetime.date(
                int(row['YEAR']), 
                int(row['MO']), 
                int(row['DY'])
            )
            
            # Determine sowing year
            if curr_date.month < crop.sowing_month:
                sowing_year = curr_date.year - 1
            else:
                sowing_year = curr_date.year
            
            sowing_date = datetime.date(
                sowing_year, 
                crop.sowing_month, 
                crop.sowing_day
            )
            
            days_grown = (curr_date - sowing_date).days
            
            # Check if within growing season
            if days_grown < 0 or days_grown > crop.total_duration:
                return 0.0
            
            # Find current growth stage
            cum_days = 0
            for i, stage_duration in enumerate(crop.stage_days):
                cum_days += stage_duration
                if days_grown <= cum_days:
                    return crop.kc[i]
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating Kc: {e}")
            return 0.0
    
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
        
        # Train-test split (preserve time order)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            shuffle=False
        )
        
        # Train model
        self.model = XGBRegressor(**default_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        metrics = {
            'rmse': rmse,
            'y_test': y_test.values,
            'predictions': predictions,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': self.features,
            'feature_importance': dict(zip(
                self.features, 
                self.model.feature_importances_
            ))
        }
        
        print(f"Model trained with RMSE: {rmse:.4f}")
        
        return metrics
    
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


def calculate_current_kc(crop_name: str) -> Tuple[float, str, int, int]:
    """
    Calculate current Kc coefficient for a crop.
    
    Args:
        crop_name: Name of the crop
        
    Returns:
        Tuple of (kc_value, stage_name, days_grown, total_duration)
    """
    if crop_name not in CROPS:
        return 0.0, "Unknown", 0, 0
    
    crop = CROPS[crop_name]
    today = datetime.date.today()
    
    # Determine sowing year
    if today.month < crop.sowing_month:
        sowing_year = today.year - 1
    else:
        sowing_year = today.year
    
    sowing_date = datetime.date(
        sowing_year, 
        crop.sowing_month, 
        crop.sowing_day
    )
    
    days_grown = (today - sowing_date).days
    total_duration = crop.total_duration
    
    # Check season status
    if days_grown < 0:
        return 0.0, "Pre-Sowing", days_grown, total_duration
    elif days_grown > total_duration:
        return 0.0, "Harvested", days_grown, total_duration
    
    # Find current stage
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