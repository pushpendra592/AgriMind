"""
Input validation utilities for Agri-Smart AI.
"""

import re
from typing import Tuple


def validate_city_input(city_name: str) -> bool:
    """
    Validate city name input.
    
    Args:
        city_name: City name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not city_name or len(city_name) > 100:
        return False
    
    # Allow letters, spaces, commas, periods, hyphens, apostrophes
    # Also allow common international characters
    pattern = r'^[a-zA-ZÀ-ÿ\s,.\-\']+$'
    return bool(re.match(pattern, city_name))


def validate_coordinates(lat: float, lon: float) -> Tuple[bool, str]:
    """
    Validate geographic coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if lat is None or lon is None:
        return False, "Coordinates cannot be None"
    
    if not (-90 <= lat <= 90):
        return False, f"Latitude must be between -90 and 90 (got {lat})"
    
    if not (-180 <= lon <= 180):
        return False, f"Longitude must be between -180 and 180 (got {lon})"
    
    return True, ""


def sanitize_input(text: str, max_length: int = 100) -> str:
    """
    Sanitize user text input.
    
    Args:
        text: Input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Strip whitespace and limit length
    sanitized = text.strip()[:max_length]
    
    # Remove any potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', sanitized)
    
    return sanitized


def validate_crop_name(crop_name: str, valid_crops: list) -> bool:
    """
    Validate crop name against list of valid crops.
    
    Args:
        crop_name: Name to validate
        valid_crops: List of valid crop names
        
    Returns:
        True if valid, False otherwise
    """
    return crop_name in valid_crops


def validate_dataframe(df, required_columns: list) -> Tuple[bool, str]:
    """
    Validate that a DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    
    return True, ""