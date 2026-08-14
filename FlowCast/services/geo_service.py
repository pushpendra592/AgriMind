"""
Geocoding service for Agri-Smart AI.
Handles location lookup and coordinate conversion.
"""
from typing import Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from config import config

class GeoService:
    """Handles geocoding operations."""
    
    def __init__(self, user_agent: str = "FlowCast"):
        self.user_agent = user_agent
        self._geolocator = None
    
    @property
    def geolocator(self) -> Nominatim:
        """Lazy initialization of geolocator."""
        if self._geolocator is None:
            self._geolocator = Nominatim(user_agent=self.user_agent)
        return self._geolocator
    
    def geocode(self, city_name: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Convert city name to coordinates.
        
        Args:
            city_name: City and country (e.g., "Jalandhar, India")
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if not found
        """
        try:
            location = self.geolocator.geocode(city_name, timeout=10)
            
            if location:
                print(f"Geocoded {city_name}: ({location.latitude}, {location.longitude})")
                return location.latitude, location.longitude
            
            print(f"Location not found: {city_name}")
            return None, None
            
        except GeocoderTimedOut:
            print(f"Geocoding timeout for {city_name}")
            return None, None
            
        except GeocoderServiceError as e:
            print(f"Geocoder service error for {city_name}: {e}")
            return None, None
            
        except Exception as e:
            print(f"Unexpected geocoding error for {city_name}: {e}")
            return None, None
    
    def reverse_geocode(self, lat: float, lon: float) -> Optional[str]:
        """
        Convert coordinates to location name.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Location name or None if not found
        """
        try:
            location = self.geolocator.reverse(f"{lat}, {lon}", timeout=10)
            
            if location:
                return location.address
            return None
            
        except Exception as e:
            print(f"Reverse geocoding error for ({lat}, {lon}): {e}")
            return None


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

_geo_service = None

def _get_geo_service() -> GeoService:
    """Get or create singleton geo service."""
    global _geo_service
    if _geo_service is None:
        _geo_service = GeoService()
    return _geo_service


def get_lat_lon(city_name: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Get coordinates for a city name (cached).
    
    Args:
        city_name: City and country
        
    Returns:
        Tuple of (latitude, longitude) or (None, None)
    """
    return _get_geo_service().geocode(city_name)