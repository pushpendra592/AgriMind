"""
Configuration settings and constants for Agri-Smart AI.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import datetime


@dataclass
class AppConfig:
    """Main application configuration."""
    
    NASA_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    WEATHER_API_URL: str = "https://api.open-meteo.com/v1/forecast"
    
    START_YEAR: int = 2001
    API_TIMEOUT: int = 30
    
    NASA_PARAMETERS: str = "T2M,T2MWET,TS,PRECTOTCORR,WS2M,GWETTOP,GWETROOT,EVLAND,RH2M,ALLSKY_SFC_SW_DWN,T2MDEW"
    
    MODEL_FEATURES: List[str] = field(default_factory=lambda: [
        'T2M', 'T2MWET', 'TS', 'WS2M', 'GWETTOP', 'GWETROOT', 'RH2M',
        'ALLSKY_SFC_SW_DWN', 'T2MDEW', 'Kc', 'Rain_Last_24h',
        'Soil_Moisture_Prev_1h', 'Evap_Prev_1h'
    ])
    
    @property
    def end_year(self) -> int:
        return datetime.datetime.now().year


@dataclass
class CropConfig:
    """Configuration for a crop type."""
    name: str
    sowing_month: int
    sowing_day: int
    stage_days: List[int]
    kc: List[float]
    
    @property
    def total_duration(self) -> int:
        return sum(self.stage_days)
    
    @property
    def stage_names(self) -> List[str]:
        return ["Initial", "Development", "Mid-Season", "Late-Season"]


CROPS: Dict[str, CropConfig] = {
    # Cereals
    "Wheat": CropConfig("Wheat", 11, 15, [20, 30, 40, 30], [0.3, 0.75, 1.15, 0.4]),
    "Rice": CropConfig("Rice", 6, 15, [30, 30, 60, 30], [1.05, 1.20, 1.20, 0.90]),
    "Maize": CropConfig("Maize", 6, 1, [20, 35, 40, 30], [0.3, 0.75, 1.20, 0.5]),
    "Barley": CropConfig("Barley", 11, 1, [15, 25, 50, 30], [0.3, 0.75, 1.15, 0.25]),
    
    # Cash Crops
    "Cotton": CropConfig("Cotton", 5, 1, [30, 50, 60, 55], [0.35, 0.75, 1.15, 0.60]),
    "Sugarcane": CropConfig("Sugarcane", 2, 15, [35, 60, 190, 120], [0.40, 0.75, 1.25, 0.75]),
    "Tobacco": CropConfig("Tobacco", 12, 1, [25, 35, 50, 30], [0.35, 0.75, 1.10, 0.80]),
    
    # Pulses
    "Chickpea": CropConfig("Chickpea", 10, 15, [25, 35, 40, 20], [0.40, 0.80, 1.00, 0.35]),
    "Lentil": CropConfig("Lentil", 11, 1, [20, 30, 60, 25], [0.40, 0.80, 1.10, 0.30]),
    "Soybean": CropConfig("Soybean", 6, 15, [20, 30, 60, 25], [0.40, 0.80, 1.15, 0.50]),
    
    # Vegetables
    "Tomato": CropConfig("Tomato", 1, 15, [30, 40, 45, 30], [0.60, 0.85, 1.15, 0.80]),
    "Potato": CropConfig("Potato", 10, 15, [25, 30, 45, 30], [0.50, 0.80, 1.15, 0.75]),
    "Onion": CropConfig("Onion", 11, 1, [15, 25, 70, 40], [0.70, 0.85, 1.05, 0.75]),
    "Cabbage": CropConfig("Cabbage", 9, 15, [20, 30, 40, 25], [0.70, 0.85, 1.05, 0.95]),
    
    # Fruits
    "Banana": CropConfig("Banana", 6, 1, [120, 90, 120, 60], [0.50, 0.80, 1.10, 1.00]),
    "Grapes": CropConfig("Grapes", 3, 1, [20, 40, 120, 60], [0.30, 0.60, 0.85, 0.45]),
    "Mango": CropConfig("Mango", 2, 1, [60, 90, 120, 90], [0.40, 0.70, 0.80, 0.60]),
}

# Singleton instance — import this in all services
config = AppConfig()