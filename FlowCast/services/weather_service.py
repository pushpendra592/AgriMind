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
