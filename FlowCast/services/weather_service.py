"""
Weather data service with parallel downloads.
"""

import time
import datetime
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

from config import config
from services.data_cleaning import clean_physical_outliers

# Physical plausibility bounds — same values used by clean_physical_outliers.
# Printed in the validation summary so the operator knows what was screened.
_EVLAND_WARN_THRESHOLD    = 1.5   # mm/hr: values above this are suspicious but kept
_PRECTOTCORR_MAX_PHYSICAL = 150.0  # mm/hr: hard cap (already enforced by cleaner)


class WeatherService:
    """Handles all weather data operations."""
    
    def __init__(self):
        self.nasa_url = config.NASA_BASE_URL
        self.weather_url = config.WEATHER_API_URL
        self.parameters = config.NASA_PARAMETERS
        self.timeout = config.API_TIMEOUT
    
    def _fetch_year_data(self, year: int, lat: float, lon: float) -> List[Dict]:
        """Fetch data for a single year."""
        params = {
            "parameters": self.parameters,
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": f"{year}0101",
            "end": f"{year}1231",
            "format": "JSON"
        }
        
        try:
            response = requests.get(
                self.nasa_url,
                params=params,
                timeout=60  # Longer timeout for large requests
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_nasa_response(data), year
            
        except Exception as e:
            print(f"Error fetching {year}: {e}")
            return [], year
    
    def fetch_nasa_data_parallel(
        self,
        lat: float,
        lon: float,
        start_year: int = None,
        end_year: int = None,
        max_workers: int = 5  # Number of parallel downloads
    ) -> pd.DataFrame:
        """
        Fetch NASA data using parallel downloads.
        
        Speed improvement: ~3-5x faster than sequential
        """
        start_year = start_year or config.START_YEAR
        end_year = end_year or config.end_year
        
        years = list(range(start_year, end_year + 1))
        all_data = []
        
        # Progress tracking
        print("**Downloading data**")
        
        completed = 0
        total = len(years)
        
        # Parallel download using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_year = {
                executor.submit(self._fetch_year_data, year, lat, lon): year
                for year in years
            }
            
            # Process completed tasks
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    year_data, _ = future.result()
                    all_data.extend(year_data)
                    completed += 1
                    
                    print(f"Downloading Satalite data for {year}...")
                    
                except Exception as e:
                    print(f"Error processing {year}: {e}")
                    completed += 1
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            df.replace(-999, float('nan'), inplace=True)
            df.dropna(inplace=True)
            # Physical-bounds cleaning: drop rows with implausible EVLAND /
            # PRECTOTCORR values before any caller can use raw data.
            df = clean_physical_outliers(df, verbose=True)
            # Integrity check: warn if duplicate timestamps or suspicious
            # field ranges are detected (runs after cleaning, on what survives).
            self._validate_raw_data(df)
            # Sort by date
            df = df.sort_values(['YEAR', 'MO', 'DY', 'HR']).reset_index(drop=True)
            
        print(f"Total records fetched: {len(df)}")
        
        return df
    
    def _parse_nasa_response(self, data: Dict) -> List[Dict]:
        """Parse NASA API response."""
        records = []
        
        try:
            parameters = data['properties']['parameter']
            first_param = list(parameters.keys())[0]
            timestamps = sorted(parameters[first_param].keys())
            
            for ts in timestamps:
                row = {
                    'YEAR': int(ts[:4]),
                    'MO': int(ts[4:6]),
                    'DY': int(ts[6:8]),
                    'HR': int(ts[8:10])
                }
                
                for param in self.parameters.split(','):
                    row[param] = parameters.get(param, {}).get(ts, -999)
                
                records.append(row)
                
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
        
        return records
    
    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """
        Run basic integrity checks on the fetched NASA POWER data and print a
        concise warning summary. Checks two independent concerns:

        1. Duplicate timestamps (YEAR/MO/DY/HR) — caused by a parsing or
           concat bug in the fetch layer, not a units problem.
        2. Suspicious EVLAND / PRECTOTCORR ranges — EVLAND physically tops out
           at ~1–1.5 mm/hr; values above that after cleaning indicate a unit
           conversion issue in the ingestion layer, not the ML model.

        Does NOT raise; prints warnings so the pipeline can continue while the
        operator investigates.
        """
        sep = "─" * 60

        # ── 1. Duplicate timestamp check ────────────────────────────────────
        key_cols = ["YEAR", "MO", "DY", "HR"]
        if all(c in df.columns for c in key_cols):
            dupes = df.duplicated(subset=key_cols, keep=False)
            if dupes.sum() > 0:
                print(f"\n  [WARN] {sep}")
                print(f"  [WARN] DUPLICATE TIMESTAMPS: {dupes.sum()} rows share a (YEAR,MO,DY,HR) key.")
                print(f"  [WARN] This points to a concat/merge bug in the fetch layer, not a units issue.")
                print(f"  [WARN] Top offenders (first 5 duplicate groups):")
                sample = (
                    df[dupes]
                    .sort_values(key_cols)
                    .head(10)
                    [key_cols + [c for c in ["EVLAND", "PRECTOTCORR"] if c in df.columns]]
                )
                for line in sample.to_string(index=False).splitlines():
                    print(f"  [WARN]   {line}")
                print(f"  [WARN] {sep}")

        # ── 2. Field-range sanity check (post-cleaning) ──────────────────────
        warnings_issued = False
        for col, warn_threshold, label in [
            ("EVLAND",      _EVLAND_WARN_THRESHOLD,    "mm/hr (physical max ~1–1.5)"),
            ("PRECTOTCORR", _PRECTOTCORR_MAX_PHYSICAL, "mm/hr (hard cap)"),
        ]:
            if col not in df.columns:
                continue
            s = df[col]
            p999 = s.quantile(0.999)
            vmax = s.max()
            if p999 > warn_threshold:
                if not warnings_issued:
                    print(f"\n  [WARN] {sep}")
                    print(f"  [WARN] RAW DATA RANGE WARNING (after cleaning):")
                    warnings_issued = True
                pct_above = 100 * (s > warn_threshold).sum() / len(s)
                print(
                    f"  [WARN]   {col}: max={vmax:.4f}, p99.9={p999:.4f} {label}  "
                    f"| {pct_above:.3f}% of rows above threshold."
                )
                if col == "EVLAND" and p999 > warn_threshold:
                    print(
                        "  [WARN]   EVLAND above threshold after cleaning — check unit"
                        " conversion in _parse_nasa_response (should be mm/hr, not kg/m²/s)."
                    )
        if warnings_issued:
            print(f"  [WARN] {sep}")

    def get_live_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetch current weather and forecast."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m,shortwave_radiation,soil_moisture_0_to_1cm",
            "daily": "precipitation_sum",
            "past_days": 1,
            "forecast_days": 6
        }
        
        try:
            response = requests.get(self.weather_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            current = data['current']
            daily = data['daily']['precipitation_sum']
            
            return {
                't2m': current['temperature_2m'],
                'humidity': current['relative_humidity_2m'],
                'rain': current['rain'],
                'wind': current['wind_speed_10m'] * 0.75,
                'sun': current['shortwave_radiation'],
                'soil': current['soil_moisture_0_to_1cm'],
                'rain_yst': daily[0] if daily else 0,
                'future_rain': daily[2:7] if len(daily) >= 7 else daily[2:]
            }
            
        except Exception as e:
            print(f"Error fetching live weather: {e}")
            return None


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

_weather_service = None

def _get_weather_service() -> WeatherService:
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service


def fetch_nasa_data(lat: float, lon: float) -> pd.DataFrame:
    """Fetch NASA data with parallel downloads."""
    return _get_weather_service().fetch_nasa_data_parallel(lat, lon)


def get_live_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Get live weather (cached)."""
    return _get_weather_service().get_live_weather(lat, lon)
