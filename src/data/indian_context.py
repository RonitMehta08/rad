"""Indian context enrichment — holiday calendar, monsoon flags, festival multipliers.

Owner: P1 (Data & Config Lead)
Consumers: generator.py (demand multipliers), preprocessor.py (feature enrichment)
Reference: MASTER_PROMPT §6.3
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.utils.constants import WeatherCategory

# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path("config")
_HOLIDAYS_CONFIG: dict[str, Any] | None = None


def _load_holidays_config() -> dict[str, Any]:
    """Load and cache the Indian holidays YAML config.

    Returns:
        Parsed holiday configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    global _HOLIDAYS_CONFIG
    if _HOLIDAYS_CONFIG is not None:
        return _HOLIDAYS_CONFIG

    config_path = _CONFIG_DIR / "indian_holidays.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Holiday config not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        _HOLIDAYS_CONFIG = yaml.safe_load(f)
    return _HOLIDAYS_CONFIG


# ---------------------------------------------------------------------------
# Holiday Detection
# ---------------------------------------------------------------------------


def get_national_holidays(year: int) -> list[dict[str, Any]]:
    """Return list of national holidays for a given year.

    Args:
        year: Calendar year.

    Returns:
        List of dicts with keys: name, date, demand_multiplier.
    """
    config = _load_holidays_config()
    holidays: list[dict[str, Any]] = []

    for h in config.get("national_holidays", []):
        try:
            holiday_date = date(year, h["month"], h["day"])
        except ValueError:
            # Handle dates that don't exist (e.g., Feb 30)
            continue
        holidays.append({
            "name": h["name"],
            "date": holiday_date,
            "demand_multiplier": h.get("demand_multiplier", 0.6),
        })

    return holidays


def is_holiday(target_date: date) -> bool:
    """Check whether a date is a national holiday.

    Args:
        target_date: Date to check.

    Returns:
        True if the date is a national holiday.
    """
    holidays = get_national_holidays(target_date.year)
    holiday_dates = {h["date"] for h in holidays}
    return target_date in holiday_dates


def get_demand_multiplier(target_date: date) -> float:
    """Compute the demand multiplier for a given date.

    Combines holiday effects, post-holiday surges, pre-holiday rushes,
    monsoon effects, and weekly patterns.

    Args:
        target_date: Date to compute multiplier for.

    Returns:
        Multiplicative factor for daily patient volume (1.0 = normal day).
    """
    config = _load_holidays_config()
    multiplier = 1.0

    # 1. Check if it's a holiday
    holidays = get_national_holidays(target_date.year)
    holiday_dates = {h["date"]: h for h in holidays}

    if target_date in holiday_dates:
        multiplier = holiday_dates[target_date]["demand_multiplier"]
        return multiplier

    # 2. Post-holiday surge
    surge_mult = config.get("post_holiday_surge_multiplier", 1.3)
    surge_days = config.get("post_holiday_surge_days", 2)
    for h_date in holiday_dates:
        days_after = (target_date - h_date).days
        if 1 <= days_after <= surge_days:
            multiplier = max(multiplier, surge_mult)
            break

    # 3. Pre-holiday rush
    rush_mult = config.get("pre_holiday_rush_multiplier", 1.15)
    rush_days = config.get("pre_holiday_rush_days", 1)
    for h_date in holiday_dates:
        days_before = (h_date - target_date).days
        if 1 <= days_before <= rush_days:
            multiplier = max(multiplier, rush_mult)
            break

    # 4. Weekly pattern
    weekly = config.get("weekly_patterns", {})
    dow = target_date.weekday()  # 0=Monday
    if dow == 0:
        multiplier *= weekly.get("monday_multiplier", 1.25)
    elif dow == 5:
        multiplier *= weekly.get("saturday_multiplier", 0.75)
    elif dow == 6:
        multiplier *= weekly.get("sunday_multiplier", 0.30)

    # 5. Monsoon season
    monsoon = config.get("monsoon_season", {})
    monsoon_start = monsoon.get("start_month", 6)
    monsoon_end = monsoon.get("end_month", 9)
    if monsoon_start <= target_date.month <= monsoon_end:
        multiplier *= monsoon.get("demand_multiplier", 0.85)

    return round(multiplier, 3)


# ---------------------------------------------------------------------------
# Weather / Season Classification
# ---------------------------------------------------------------------------


def get_weather_category(target_date: date) -> WeatherCategory:
    """Classify a date into an Indian weather category.

    Args:
        target_date: Date to classify.

    Returns:
        WeatherCategory enum value.
    """
    month = target_date.month
    if 6 <= month <= 9:
        return WeatherCategory.MONSOON
    elif month in (3, 4, 5, 10):
        return WeatherCategory.SUMMER
    else:
        return WeatherCategory.WINTER


def get_monsoon_emergency_multiplier(target_date: date) -> float:
    """Return emergency arrival multiplier during monsoon season.

    Args:
        target_date: Date to check.

    Returns:
        Multiplier (>1.0 during monsoon, 1.0 otherwise).
    """
    config = _load_holidays_config()
    monsoon = config.get("monsoon_season", {})
    monsoon_start = monsoon.get("start_month", 6)
    monsoon_end = monsoon.get("end_month", 9)

    if monsoon_start <= target_date.month <= monsoon_end:
        return monsoon.get("emergency_multiplier", 1.15)
    return 1.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def generate_holiday_features(target_date: date) -> dict[str, Any]:
    """Generate all Indian-context features for a given date.

    Args:
        target_date: Date to generate features for.

    Returns:
        Dictionary with keys: is_holiday, is_weekend, is_monday,
        weather_category, demand_multiplier, day_of_week.
    """
    dow = target_date.weekday()
    return {
        "is_holiday": is_holiday(target_date),
        "is_weekend": dow >= 5,
        "is_monday": dow == 0,
        "weather_category": get_weather_category(target_date).value,
        "demand_multiplier": get_demand_multiplier(target_date),
        "day_of_week": dow,
    }
