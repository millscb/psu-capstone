"""This module implements a geospatial encoder that encodes GPS coordinates and speed into a sparse binary representation. It uses the CoordinateEncoder for encoding the spatial information and includes logic for handling different coordinate reference systems and speed-based radius calculation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Optional, override

import numpy as np
from pyproj import CRS, Transformer

from htmrl.encoder_layer.base_encoder import BaseEncoder, ParameterMarker
from htmrl.encoder_layer.coordinate_encoder import CoordinateEncoder, CoordinateParameters

# Good for 2D projected x/y coordinates in meters
CRS_MERC = CRS.from_epsg(3857)

# Raw geographic coordinates
CRS_WGS84 = CRS.from_epsg(4326)

T_WGS84_TO_MERC = Transformer.from_crs(CRS_WGS84, CRS_MERC, always_xy=True)
T_MERC_TO_WGS84 = Transformer.from_crs(CRS_MERC, CRS_WGS84, always_xy=True)


class GeospatialEncoder(
    BaseEncoder[tuple[float, float, float] | tuple[float, float, float, float]]
):
    """Encode geospatial position and speed into coordinate-based SDRs."""

    def __init__(
        self,
        geo_params: GeospatialParameters,
        coord_params: CoordinateParameters | None = None,
    ):
        self._geo_params = copy.deepcopy(geo_params)
        self._coord_params = copy.deepcopy(
            coord_params if coord_params is not None else CoordinateParameters()
        )

        dims = 3 if self._geo_params.use_altitude else 2

        coord_params = CoordinateParameters(
            n=self._coord_params.n,
            w=self._coord_params.w,
            seed=self._coord_params.seed,
            max_radius=self._geo_params.max_radius,
            dims=dims,
            use_all_neighbors=self._coord_params.use_all_neighbors,
        )

        self._encoder = CoordinateEncoder(coord_params)
        super().__init__(self._encoder.size)

    @override
    def encode(
        self, input_value: tuple[float, float, float] | tuple[float, float, float, float]
    ) -> list[int]:
        """Encode ``(speed, lon, lat[, alt])`` into an SDR."""
        if len(input_value) == 3:
            speed, lon, lat = input_value
            alt = None
        elif len(input_value) == 4:
            speed, lon, lat, alt = input_value
        else:
            raise ValueError("Expected (speed, lon, lat) or (speed, lon, lat, alt).")

        speed = float(speed)
        lon = self._wrap_lon(float(lon))
        lat = self._clamp_lat(float(lat))
        alt = None if alt is None else float(alt)

        coord = self.coordinate_for_position(lon, lat, alt)
        radius = self.radius_for_speed(speed)

        return self._encoder.encode((coord, radius))

    @override
    def decode(
        self,
        encoded: list[int],
        candidates: Iterable[tuple[tuple[int, ...], int]] | None = None,
    ) -> tuple[tuple[float, float, Optional[float]] | None, float]:
        """Decode an SDR into an approximate ``(lon, lat, alt)`` tuple."""
        best_key, conf = self._encoder.decode(encoded, candidates=candidates)
        if best_key is None:
            return None, 0.0

        grid_coord, _radius = best_key
        pos = self.position_for_coordinate(grid_coord)
        return pos, conf

    def coordinate_for_position(
        self, lon: float, lat: float, alt: Optional[float]
    ) -> tuple[int, ...]:
        """Project geographic coordinates into integer grid coordinates."""
        lon = self._wrap_lon(float(lon))
        lat = self._clamp_lat(float(lat))

        x_m, y_m = T_WGS84_TO_MERC.transform(lon, lat)

        if self._geo_params.use_altitude:
            z_m = 0.0 if alt is None else float(alt)

            x_idx = int(round(x_m / float(self._geo_params.xy_scale)))
            y_idx = int(round(y_m / float(self._geo_params.xy_scale)))
            z_idx = int(round(z_m / float(self._geo_params.z_scale)))

            return (x_idx, y_idx, z_idx)

        x_idx = int(round(x_m / float(self._geo_params.xy_scale)))
        y_idx = int(round(y_m / float(self._geo_params.xy_scale)))
        return (x_idx, y_idx)

    def position_for_coordinate(
        self, coord: tuple[int, ...]
    ) -> tuple[float, float, Optional[float]]:
        """Map integer grid coordinates back to geographic position."""
        if self._geo_params.use_altitude and len(coord) == 3:
            x_m = coord[0] * float(self._geo_params.xy_scale)
            y_m = coord[1] * float(self._geo_params.xy_scale)
            z_m = coord[2] * float(self._geo_params.z_scale)

            lon, lat = T_MERC_TO_WGS84.transform(x_m, y_m)
            lon = self._wrap_lon(float(lon))
            lat = self._clamp_lat(float(lat))
            return lon, lat, float(z_m)

        if len(coord) != 2:
            raise ValueError(f"Expected 2D coordinate when altitude is disabled, got {len(coord)}")

        x_m = coord[0] * float(self._geo_params.xy_scale)
        y_m = coord[1] * float(self._geo_params.xy_scale)

        lon, lat = T_MERC_TO_WGS84.transform(x_m, y_m)
        lon = self._wrap_lon(float(lon))
        lat = self._clamp_lat(float(lat))
        return lon, lat, None

    def radius_for_speed(self, speed_mps: float) -> int:
        """Compute coordinate radius from speed and timestep settings."""
        overlap = 1.5
        coords_per_timestep = speed_mps * float(self._geo_params.timestep)
        r = int(round((coords_per_timestep / 2.0) * overlap))

        min_r = int(math.ceil((math.sqrt(self._coord_params.w) - 1.0) / 2.0))

        r = max(r, min_r)
        r = max(0, min(r, self._geo_params.max_radius))
        return r

    @staticmethod
    def _clamp_lat(lat: float) -> float:
        # Web Mercator is undefined at poles.
        return max(-85.05112878, min(85.05112878, lat))

    @staticmethod
    def _wrap_lon(lon: float) -> float:
        # wrap to [-180, 180)
        return (lon + 180.0) % 360.0 - 180.0


@dataclass
class GeospatialParameters(ParameterMarker):
    """Configuration parameters for :class:`GeospatialEncoder`."""

    # Backward-compatible aliases used by older trainer/tests.
    size: int | None = None
    scale: float | None = None

    # Horizontal meters per grid unit for lon/lat projected into Mercator.
    xy_scale: float = 5.0

    # Vertical meters per grid unit for altitude.
    z_scale: float = 0.5

    # Seconds between readings.
    timestep: float = 1.0

    # Clamp output radius.
    max_radius: int = 2

    # Whether altitude should be included as a third coordinate axis.
    use_altitude: bool = True

    encoder_class = GeospatialEncoder

    def __post_init__(self) -> None:
        if self.scale is not None:
            self.xy_scale = float(self.scale)


if __name__ == "__main__":
    coord_params = CoordinateParameters(n=400, w=25)

    geo_params = GeospatialParameters(
        xy_scale=5.0,
        z_scale=0.5,
        timestep=2.0,
        max_radius=40,
        use_altitude=True,
    )

    enc = GeospatialEncoder(geo_params=geo_params, coord_params=coord_params)

    a = enc.encode((5.0, -77.0365, 38.8977, 10.0))
    b = enc.encode((5.0, -77.0365, 38.8977, 10.5))

    print(enc.decode(a))
    print(enc.decode(b))
