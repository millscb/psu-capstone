"""Pytest configuration and shared fixtures for all test modules."""

from datetime import datetime
from typing import Any

import pytest

from htmrl.encoder_layer.category_encoder import CategoryParameters
from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters

# ============================================================================
# DateEncoder Fixtures
# ============================================================================


@pytest.fixture
def date_params_year_only() -> DateEncoderParameters:
    """DateEncoderParameters for year-only encoding with RDSE."""
    return DateEncoderParameters(
        year_size=2048,
        year_active_bits=42,
        year_sparsity=0.0,
        year_radius=0.0,
        year_resolution=1.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )


@pytest.fixture
def date_params_season_only() -> DateEncoderParameters:
    """DateEncoderParameters for season-only encoding with RDSE."""
    return DateEncoderParameters(
        year_active_bits=0,
        season_size=2048,
        season_active_bits=42,
        season_sparsity=0.0,
        season_radius=91.5,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )


@pytest.fixture
def date_params_day_of_week_only() -> DateEncoderParameters:
    """DateEncoderParameters for day-of-week only encoding with RDSE."""
    return DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_size=2048,
        day_of_week_active_bits=42,
        day_of_week_radius=292.57,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )


@pytest.fixture
def date_params_weekend_only() -> DateEncoderParameters:
    """DateEncoderParameters for weekend-only encoding with RDSE."""
    return DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_size=2048,
        weekend_active_bits=42,
        weekend_radius=39.38,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )


@pytest.fixture
def date_params_custom_only() -> DateEncoderParameters:
    """DateEncoderParameters for custom-days only encoding with RDSE."""
    return DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_size=2048,
        custom_active_bits=42,
        custom_radius=409.6,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["mon,tue,wed,thu,fri"],
        rdse_used=True,
    )


@pytest.fixture
def date_params_holiday_only() -> DateEncoderParameters:
    """DateEncoderParameters for holiday-only encoding with RDSE."""
    return DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_size=2048,
        holiday_active_bits=42,
        holiday_dates=[[2020, 1, 1], [7, 4], [12, 25]],
        holiday_radius=682.67,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )


@pytest.fixture
def date_params_time_of_day_only() -> DateEncoderParameters:
    """DateEncoderParameters for time-of-day only encoding with RDSE."""
    return DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_size=2048,
        time_of_day_active_bits=42,
        time_of_day_radius=85.33,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_active_bits=0,
        rdse_used=True,
    )


@pytest.fixture
def date_params_all_combined() -> DateEncoderParameters:
    """DateEncoderParameters with all temporal components enabled (no year)."""
    return DateEncoderParameters(
        year_active_bits=0,
        season_size=100,
        season_active_bits=2,
        season_sparsity=0.0,
        season_radius=25.0,
        season_resolution=0.0,
        day_of_week_size=100,
        day_of_week_active_bits=2,
        day_of_week_radius=14.28,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_size=100,
        weekend_active_bits=2,
        weekend_radius=1.92,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_size=100,
        holiday_active_bits=2,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=9.09,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_size=100,
        time_of_day_active_bits=2,
        time_of_day_radius=0.0278,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_size=100,
        custom_active_bits=2,
        custom_radius=25.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Monday", "Mon, Wed, Fri"],
        rdse_used=True,
    )


@pytest.fixture
def date_params_all_combined_with_year() -> DateEncoderParameters:
    """DateEncoderParameters with all temporal components including year enabled (RDSE)."""
    return DateEncoderParameters(
        year_size=2048,
        year_active_bits=42,
        year_sparsity=0.0,
        year_radius=0.0,
        year_resolution=1.0,
        season_size=1024,
        season_active_bits=22,
        season_sparsity=0.0,
        season_radius=256.0,
        season_resolution=0.0,
        day_of_week_size=1024,
        day_of_week_active_bits=22,
        day_of_week_radius=146.28,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_size=1024,
        weekend_active_bits=22,
        weekend_radius=512.0,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_size=1024,
        holiday_active_bits=22,
        holiday_dates=[[1, 1], [7, 4], [12, 25]],
        holiday_radius=341.33,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_size=2048,
        time_of_day_active_bits=42,
        time_of_day_radius=0.023704,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_size=1024,
        custom_active_bits=22,
        custom_radius=204.8,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Mon, Tue, Wed, Thu, Fri"],
        rdse_used=True,
    )


# ============================================================================
# RDSE Fixtures
# ============================================================================


@pytest.fixture
def rdse_params_basic() -> RDSEParameters:
    """Basic RDSE parameters for general testing."""
    return RDSEParameters(
        size=2048,
        active_bits=40,
        sparsity=0.0,
        resolution=0.0,
        radius=10.0,
        category=False,
        seed=42,
    )


@pytest.fixture
def rdse_params_small() -> RDSEParameters:
    """Small RDSE parameters for quick tests."""
    return RDSEParameters(
        size=100,
        active_bits=2,
        sparsity=0.0,
        resolution=0.0,
        radius=1.0,
        category=False,
        seed=1,
    )


# ============================================================================
# Category Encoder Fixtures
# ============================================================================


@pytest.fixture
def category_params_basic() -> CategoryParameters:
    """Basic category encoder parameters."""
    return CategoryParameters(
        w=3,
        category_list=["A", "B", "C"],
        rdse_used=False,
    )


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def decoder_test_dates() -> list[list[int]]:
    """Common test dates [year, month, day, hour, minute] for decoder tests."""
    return [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
    ]


@pytest.fixture
def sample_batch_data() -> list[dict[str, Any]]:
    """Sample batch data for encoder handler tests."""
    return [
        {"float_col": 3.14, "int_col": 42, "str_col": "B", "date_col": datetime(2023, 12, 25)},
        {"float_col": 5.4, "int_col": 21, "str_col": "C", "date_col": datetime(2023, 12, 26)},
        {"float_col": 6.7, "int_col": 10, "str_col": "D", "date_col": datetime(2023, 12, 27)},
        {"float_col": 12.4, "int_col": 5, "str_col": "E", "date_col": datetime(2023, 12, 28)},
    ]
