"""Date encoder 2.0 for HTM, ported to Python from the C++ implementation.

This module provides a DateEncoder class that encodes various temporal features
(season, day of week, weekend, custom days, holiday, time of day) into a Sparse
Distributed Representation (SDR) for use in Hierarchical Temporal Memory (HTM) systems.

The encoder uses ScalarEncoder or RDSE instances for each enabled feature, concatenating
their outputs into a single SDR. The configuration is controlled via the
DateEncoderParameters dataclass.

"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, override

import numpy as np
import pandas as pd

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.log import get_logger


class DateEncoder(BaseEncoder[datetime | pd.Timestamp | time.struct_time | np.datetime64 | None]):
    """Python port of the HTM DateEncoder using scalar encoders with default parameters.

    Encodes up to 7 features using different encoders of a timestamp into one SDR:
    - year
    - season (day-of-year)
    - dayOfWeek
    - weekend
    - customDays
    - holiday
    - timeOfDay

    Important: Year and Season encoders are mutually exclusive (XOR).
    Enable year for absolute year tracking, OR season for cyclic day-of-year, but not both.

    Args:
        date_params: DateEncoderParameters instance specifying encoding options.
            If None, defaults to DateEncoderParameters().
        force_size: If True, forces the total output size to match date_params.size by dividing it equally among enabled sub-encoders. Defaults to False.
    """

    # !!enum!! type constants for indices
    SEASON = 0
    DAYOFWEEK = 1
    WEEKEND = 2
    CUSTOM = 3
    HOLIDAY = 4
    TIMEOFDAY = 5
    YEAR = 6

    def __init__(
        self,
        date_params: DateEncoderParameters | None = None,
        force_size: bool = False,
    ) -> None:

        self._logger = get_logger("DateEncoder")
        self._date_params: DateEncoderParameters = (
            copy.deepcopy(date_params) if date_params is not None else DateEncoderParameters()
        )
        self._customDays: set[int] = set()  # no repeating days
        """Set of integer day indices for custom days."""
        self._bucketMap: dict[int, int] = {}
        """Mapping from feature index to bucket position."""
        self._buckets: list[float] = []
        """List of bucket values for each feature."""
        self._force_size: bool = force_size  # default False
        """Whether to force the output size to match date_params.size even if it doesn't match sum of sub-encoder sizes."""
        self._size: int = self._date_params.size
        """Total number of bits DateEncoder."""
        self._rdse_used: bool = self._date_params.rdse_used
        """Flag indicating if RDSE is used."""
        self._all_valid_encoders: bool = False
        """Flag indicating if all sub-encoders are valid."""

        # Declare one encoder per feature
        self._year_encoder: RandomDistributedScalarEncoder | ScalarEncoder | None = None
        """Encoder for year."""
        self._season_encoder: RandomDistributedScalarEncoder | ScalarEncoder | None = None
        """Encoder for season (day of year)."""
        self._dayofweek_encoder: RandomDistributedScalarEncoder | ScalarEncoder | None = None
        """Encoder for day of week."""
        self._weekend_encoder: RandomDistributedScalarEncoder | ScalarEncoder | None = None
        """Encoder for weekend flag."""
        self._customdays_encoder: RandomDistributedScalarEncoder | ScalarEncoder | None = None
        """Encoder for custom day groups."""
        self._holiday_encoder: RandomDistributedScalarEncoder | ScalarEncoder | None = None
        """Encoder for holidays."""
        self._timeofday_encoder: RandomDistributedScalarEncoder | ScalarEncoder | None = None
        """Encoder for time of day."""

        # call initialize
        self._initialize(self._date_params)
        super().__init__(self._size)

    def _number_of_encoders(self, params: DateEncoderParameters) -> int:
        """Helper to count how many sub-encoders are enabled."""

        count = 0
        if params.year_active_bits > 0:
            count += 1
        if params.season_active_bits > 0:
            count += 1
        if params.day_of_week_active_bits > 0:
            count += 1
        if params.weekend_active_bits > 0:
            count += 1
        if params.custom_active_bits > 0:
            count += 1
        if params.holiday_active_bits > 0:
            count += 1
        if params.time_of_day_active_bits > 0:
            count += 1
        return count

    def _setup_feature_encoder(
        self,
        *,
        feature_key: int,
        size_value: int,
        active_bits: int,
        radius: float,
        resolution: float,
        sparsity: float,
        seed: int = 42,
    ) -> RandomDistributedScalarEncoder | ScalarEncoder | None:
        """Instantiate and register a sub-encoder, keeping _initialize readable.

        Must define either active_bits > 0 or sparsity > 0.0, and either radius > 0.0
        or resolution > 0.0.

        Args:
            feature_key: Integer key for the feature (e.g., SEASON).
            size_value: Size of the encoder (total bits).
            active_bits: Number of active bits for the encoder.
            radius: Radius for the encoder.
            resolution: Resolution for the encoder.
            sparsity: Sparsity for the encoder.
            seed: Random seed for the encoder. Defaults to 42.

        Returns:
            An instance of RandomDistributedScalarEncoder or ScalarEncoder.
        """
        encoder_params = {
            "size": size_value,
            "active_bits": active_bits,
            "radius": radius,
            "resolution": resolution,
            "sparsity": sparsity,
            "seed": seed,
        }

        if self._rdse_used:
            rdse_params = encoder_params.copy()
            if active_bits <= 0:
                return None
            params = RDSEParameters(**rdse_params)
            encoder = RandomDistributedScalarEncoder(params)
        else:
            scalar_params = encoder_params.copy()
            scalar_params.pop("seed")  # ScalarEncoder doesn't use seed
            if active_bits <= 0:
                return None
            params = ScalarEncoderParameters(**scalar_params)
            encoder = ScalarEncoder(params)

        self._bucketMap[feature_key] = len(self._buckets)
        self._buckets.append(0.0)

        return encoder

    def _initialize(self, date_params: DateEncoderParameters) -> None:
        """Configure encoders according to the supplied parameters.

        Args:
            date_params: DateEncoderParameters instance specifying encoding options.

        Raises:
            ValueError: If custom_days is specified but empty, or if no active bits are provided.
            RuntimeError: If no sub-encoders are successfully configured.
        """

        args = date_params
        size = 0
        number_of_encoders = self._number_of_encoders(args)
        self._logger.info(
            "DateEncoder: initializing with %d sub-encoders enabled.", number_of_encoders
        )
        self._bucketMap.clear()
        self._buckets.clear()

        # -------- Year --------
        if args.year_active_bits > 0:
            self._logger.info("DateEncoder: enabling year encoder.")
            self._year_encoder = self._setup_feature_encoder(
                feature_key=self.YEAR,
                size_value=(
                    args.year_size if not self._force_size else self._size // number_of_encoders
                ),
                active_bits=args.year_active_bits,
                radius=args.year_radius,
                resolution=args.year_resolution,
                sparsity=args.year_sparsity,
                seed=args.seed + self.YEAR,  # different seed per encoder for RDSE
            )
            if self._year_encoder is None:
                raise ValueError("DateEncoder: year encoder must be enabled.")
            size += self._year_encoder.size

        # -------- Season --------
        if args.season_active_bits > 0:
            self._logger.info("DateEncoder: enabling season encoder.")
            self._season_encoder = self._setup_feature_encoder(
                feature_key=self.SEASON,
                size_value=(
                    args.season_size if not self._force_size else self._size // number_of_encoders
                ),
                active_bits=args.season_active_bits,
                radius=args.season_radius,
                resolution=args.season_resolution,
                sparsity=args.season_sparsity,
                seed=args.seed + self.SEASON,
            )  # different seed per encoder for RDSE

            if self._season_encoder is None:
                raise ValueError("DateEncoder: season encoder must be enabled.")
            size += self._season_encoder.size

        # -------- Day of week --------
        if args.day_of_week_active_bits > 0:
            self._logger.info("DateEncoder: enabling day of week encoder.")
            self._dayofweek_encoder = self._setup_feature_encoder(
                feature_key=self.DAYOFWEEK,
                size_value=(
                    args.day_of_week_size
                    if not self._force_size
                    else self._size // number_of_encoders
                ),
                active_bits=args.day_of_week_active_bits,
                radius=args.day_of_week_radius,
                resolution=args.day_of_week_resolution,
                sparsity=args.day_of_week_sparsity,
                seed=args.seed + self.DAYOFWEEK,  # different seed per encoder for RDSE
            )
            if self._dayofweek_encoder is None:
                raise ValueError("DateEncoder: day of week encoder must be enabled.")
            size += self._dayofweek_encoder.size

        # -------- Weekend --------
        if args.weekend_active_bits > 0:
            self._logger.info("DateEncoder: enabling weekend encoder.")
            self._weekend_encoder = self._setup_feature_encoder(
                feature_key=self.WEEKEND,
                size_value=(
                    args.weekend_size if not self._force_size else self._size // number_of_encoders
                ),
                active_bits=args.weekend_active_bits,
                radius=args.weekend_radius,
                resolution=args.weekend_resolution,
                sparsity=args.weekend_sparsity,
                seed=args.seed + self.WEEKEND,  # different seed per encoder for RDSE
            )
            if self._weekend_encoder is None:
                raise ValueError("DateEncoder: weekend encoder must be enabled.")
            size += self._weekend_encoder.size

        # -------- Custom days --------
        if args.custom_active_bits > 0:
            self._logger.info("DateEncoder: enabling custom days encoder.")
            if not args.custom_days:
                raise ValueError(
                    "DateEncoder: custom_days must contain at least one pattern string."
                )

            # Map strings to Python tm_wday (0=Mon..6=Sun)
            daymap = {
                "mon": 0,
                "tue": 1,
                "wed": 2,
                "thu": 3,
                "fri": 4,
                "sat": 5,
                "sun": 6,
            }
            for spec in args.custom_days:
                s = spec.lower()
                parts = [x.strip() for x in s.split(",") if x.strip()]
                for day in parts:
                    if len(day) < 3:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    key = day[:3]
                    if key not in daymap:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    self._customDays.add(daymap[key])

            self._customdays_encoder = self._setup_feature_encoder(
                feature_key=self.CUSTOM,
                size_value=(
                    args.custom_size if not self._force_size else self._size // number_of_encoders
                ),
                active_bits=args.custom_active_bits,
                radius=args.custom_radius,
                resolution=args.custom_resolution,
                sparsity=args.custom_sparsity,
                seed=args.seed + self.CUSTOM,  # different seed per encoder for RDSE
            )
            if self._customdays_encoder is None:
                raise ValueError("DateEncoder: custom days encoder must be enabled.")
            size += self._customdays_encoder.size

        # -------- Holiday --------
        if args.holiday_active_bits > 0:
            self._logger.info("DateEncoder: enabling holiday encoder.")
            for day in args.holiday_dates:
                if len(day) not in (2, 3):
                    raise ValueError(
                        "DateEncoder: holiday_dates entries must be [mon,day] or [year,mon,day]."
                    )
            self._holiday_encoder = self._setup_feature_encoder(
                feature_key=self.HOLIDAY,
                size_value=(
                    args.holiday_size if not self._force_size else self._size // number_of_encoders
                ),
                active_bits=args.holiday_active_bits,
                radius=args.holiday_radius,
                resolution=args.holiday_resolution,
                sparsity=args.holiday_sparsity,
                seed=args.seed + self.HOLIDAY,  # different seed per encoder for RDSE
            )
            if self._holiday_encoder is None:
                raise ValueError("DateEncoder: holiday encoder must be enabled.")
            size += self._holiday_encoder.size

        # -------- Time of day --------
        if args.time_of_day_active_bits > 0:
            self._logger.info("DateEncoder: enabling time of day encoder.")
            self._timeofday_encoder = self._setup_feature_encoder(
                feature_key=self.TIMEOFDAY,
                size_value=(
                    args.time_of_day_size
                    if not self._force_size
                    else self._size // number_of_encoders
                ),
                active_bits=args.time_of_day_active_bits,
                radius=args.time_of_day_radius,
                resolution=args.time_of_day_resolution,
                sparsity=args.time_of_day_sparsity,
                seed=args.seed + self.TIMEOFDAY,  # different seed per encoder for RDSE
            )
            if self._timeofday_encoder is None:
                raise ValueError(
                    "DateEncoder: time of day encoder must be enabled when active bits are set."
                )
            size += self._timeofday_encoder.size

        self._size = size

        if self._size <= 0:
            raise RuntimeError("DateEncoder misconfigured: no sub-encoders enabled.")
        if self._size != self._date_params.size:
            self._logger.warning(
                "DateEncoder: total size %d does not match sum of sub-encoder sizes %d. Using computed size.",
                self._date_params.size,
                self._size,
            )

        self.size = self._size

    @override
    def encode(self, input_value: Any) -> list[int]:
        """Encode a timestamp-like value into output SDR.

        Supported input types:
        - None: Uses current local time
        - int/float: UNIX epoch seconds
        - datetime: Python datetime (naive treated as local)
        - struct_time: Used directly
        - np.datetime64: NumPy datetime64

        Args:
            input_value: Datetime, pd.Timestamp, struct_time, np.datetime64, or None for current time.

        Returns:
            List of active bit indices.

        Raises:
            ValueError: If input_value is not a valid timestamp-like type.
            TypeError: If input_value is of unsupported type.
            RuntimeError: If encoder is misconfigured with no sub-encoders.
        """
        if (
            type(input_value) is not datetime
            and type(input_value) is not pd.Timestamp
            and type(input_value) is not time.struct_time
            and type(input_value) is not np.datetime64
            and type(input_value) is not float
            and input_value is not None
        ):
            raise ValueError("You did not enter a valid date or time.")

        output_sdr: list[int] = []

        if input_value is None:
            t = time.localtime()
        elif isinstance(input_value, (int, float)):
            t = time.localtime(float(input_value))
        elif isinstance(input_value, datetime):
            ts = input_value.timestamp()
            t = time.localtime(ts)
        elif isinstance(input_value, time.struct_time):
            t = input_value
        elif isinstance(input_value, np.datetime64):
            ts = input_value.astype("datetime64[s]").astype("int")
            t = time.localtime(ts)
        else:
            raise TypeError(f"Unsupported type for DateEncoder.encode: {type(input_value)}")

        # Collect per-attribute SDRs to later concatenate
        sdrs: list[list[int]] = []

        # -- - Year ---
        if isinstance(self._year_encoder, (RandomDistributedScalarEncoder, ScalarEncoder)):
            year = float(t.tm_year)
            encoded_value = self._year_encoder.encode(year)
            bucket_idx = math.floor(year / self._year_encoder._radius)
            self._buckets[self._bucketMap[self.YEAR]] = float(bucket_idx)

            sdrs.append(encoded_value)

        # --- Season: day of year (0-based) ---
        if isinstance(self._season_encoder, (RandomDistributedScalarEncoder, ScalarEncoder)):
            day_of_year = float(t.tm_yday - 1)  # tm_yday is 1..366
            encoded_value = self._season_encoder.encode(day_of_year)
            bucket_idx = math.floor(day_of_year / self._season_encoder._radius)
            self._buckets[self._bucketMap[self.SEASON]] = float(bucket_idx)

            sdrs.append(encoded_value)

        # --- Day of week (Monday=0..Sunday=6, same as header comment) ---
        if isinstance(self._dayofweek_encoder, (RandomDistributedScalarEncoder, ScalarEncoder)):
            # C++: dayOfWeek = (tm_wday + 6) % 7, with tm_wday 0=Sun..6=Sat
            # Python tm_wday: 0=Mon..6=Sun
            # So emulate C++ tm_wday first:
            c_tm_wday = (t.tm_wday + 1) % 7  # now 0=Sun..6=Sat
            day_of_week = float((c_tm_wday + 6) % 7)
            encoded_value = self._dayofweek_encoder.encode(day_of_week)
            radius = max(self._dayofweek_encoder._radius, 1e-9)
            bucket_val = day_of_week - math.fmod(day_of_week, radius)
            self._buckets[self._bucketMap[self.DAYOFWEEK]] = bucket_val

            sdrs.append(encoded_value)

        else:
            # still compute c_tm_wday for weekend/custom use
            c_tm_wday = (t.tm_wday + 1) % 7

        # --- Weekend flag (Fri 18:00 .. Sun 23:59) ---
        if isinstance(self._weekend_encoder, (RandomDistributedScalarEncoder, ScalarEncoder)):
            # C++ logic uses C tm_wday (0=Sun..6=Sat)
            if c_tm_wday == 0 or c_tm_wday == 6 or (c_tm_wday == 5 and t.tm_hour > 18):
                val = 1.0
            else:
                val = 0.0
            encoded_value = self._weekend_encoder.encode(val)
            self._buckets[self._bucketMap[self.WEEKEND]] = val

            sdrs.append(encoded_value)

        # --- Custom days ---
        if isinstance(self._customdays_encoder, (RandomDistributedScalarEncoder, ScalarEncoder)):
            # customDays_ holds Python tm_wday (0=Mon..6=Sun)
            custom_val = 1.0 if t.tm_wday in self._customDays else 0.0
            encoded_value = self._customdays_encoder.encode(custom_val)
            self._buckets[self._bucketMap[self.CUSTOM]] = custom_val

            sdrs.append(encoded_value)

        # --- Holiday ramp ---
        if isinstance(self._holiday_encoder, (RandomDistributedScalarEncoder, ScalarEncoder)):
            val = self._holiday_value(t)
            encoded_value = self._holiday_encoder.encode(val)
            self._buckets[self._bucketMap[self.HOLIDAY]] = math.floor(val)

            sdrs.append(encoded_value)

        # --- Time of day ---
        if isinstance(self._timeofday_encoder, (RandomDistributedScalarEncoder, ScalarEncoder)):
            tod = t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0
            encoded_value = self._timeofday_encoder.encode(tod)
            radius = max(self._timeofday_encoder._radius, 1e-9)
            bucket_val = tod - math.fmod(tod, radius)
            self._buckets[self._bucketMap[self.TIMEOFDAY]] = bucket_val

            sdrs.append(encoded_value)

        if not sdrs:
            raise RuntimeError("DateEncoder misconfigured: no sub-encoders enabled.")

        for sdr in sdrs:
            output_sdr.extend(sdr)
        self._logger.info("Date encoded value: %s", input_value)
        #  TODO: could we use a union here and still maintian similar behavior?
        return output_sdr

    @override
    def decode(
        self, encoded: list[int], candidates: Iterable[float] | None = None
    ) -> dict[str, tuple[float | None]]:
        """
        This method checks if an encoder exists. Then, if it does, we run compute decode to and append
        the value to the decode_floats.

        :param self: Description
        :param encoded: Is the SDR that you want decoded.
        :type encoded: list[int]
        :param candidates: Iterable candidates, no function yet.
        :type candidates: Iterable[float] | None
        :return: Returns a Tuple of [value, confidence]....n times/the number of encoders that had been used.
        :rtype: dict[str, tuple[float | None]]
        """
        decoded_floats = {}
        if self._year_encoder is not None and isinstance(
            self._year_encoder, RandomDistributedScalarEncoder
        ):
            local_decode = self._compute_decode(self._year_encoder, encoded)
            decoded_floats["year"] = local_decode
        if self._season_encoder is not None and isinstance(
            self._season_encoder, RandomDistributedScalarEncoder
        ):
            local_decode = self._compute_decode(self._season_encoder, encoded)
            decoded_floats["season"] = local_decode
        if self._dayofweek_encoder is not None and isinstance(
            self._dayofweek_encoder, RandomDistributedScalarEncoder
        ):
            local_decode = self._compute_decode(self._dayofweek_encoder, encoded)
            decoded_floats["dayofweek"] = local_decode
        if self._weekend_encoder is not None and isinstance(
            self._weekend_encoder, RandomDistributedScalarEncoder
        ):
            local_decode = self._compute_decode(self._weekend_encoder, encoded)
            decoded_floats["weekend"] = local_decode
        if self._customdays_encoder is not None and isinstance(
            self._customdays_encoder, RandomDistributedScalarEncoder
        ):
            local_decode = self._compute_decode(self._customdays_encoder, encoded)
            decoded_floats["customdays"] = local_decode
        if self._holiday_encoder is not None and isinstance(
            self._holiday_encoder, RandomDistributedScalarEncoder
        ):
            local_decode = self._compute_decode(self._holiday_encoder, encoded)
            decoded_floats["holiday"] = local_decode
        if self._timeofday_encoder is not None and isinstance(
            self._timeofday_encoder, RandomDistributedScalarEncoder
        ):
            local_decode = self._compute_decode(self._timeofday_encoder, encoded)
            decoded_floats["timeofday"] = local_decode
        # for float in decoded_floats:
        self._logger.info("Date decoded value: %s", decoded_floats)
        return decoded_floats

    def _compute_decode(
        self, rdse: RandomDistributedScalarEncoder, encoded: list[int]
    ) -> tuple[float | None, float]:
        """
        This method takes in the encoder and sdr to be decoded. We slice from start of encoded
        to size of the encoder. We then remove those bits from the list and decode the slice.

        :param self: Description
        :param rdse: The encoder we are running decode from. Corresponds to 1 of 6 possible encoders.
        :type rdse: RandomDistributedScalarEncoder
        :param encoded: The sdr that has been input for decoding.
        :type encoded: list[int]
        :return: Return a tuple of [value, confidence]
        :rtype: float
        """
        size = rdse.size
        local_encode = encoded[:size]
        del encoded[:size]
        local_decode = rdse.decode(local_encode)
        return local_decode

    def _holiday_value(self, t: time.struct_time) -> float:
        """Return the holiday ramp value for the provided timestamp."""
        seconds_per_day = 86400.0
        input_ts = time.mktime(t)

        for h in self._date_params.holiday_dates:
            if len(h) == 3:
                year, mon, day = h
            else:
                year = t.tm_year
                mon, day = h
            h_ts = self.mktime(year, mon, day)

            if input_ts > h_ts:
                diff = input_ts - h_ts
                if diff < seconds_per_day:
                    return 1.0
                elif diff < 2.0 * seconds_per_day:
                    return 1.0 + (diff - seconds_per_day) / seconds_per_day
            else:
                diff = h_ts - input_ts
                if diff < seconds_per_day:
                    return 1.0 - diff / seconds_per_day

        return 0.0

    @staticmethod
    def mktime(year: int, mon: int, day: int, hr: int = 0, minute: int = 0, sec: int = 0) -> float:
        """Convenience to generate unix epoch seconds like the C++ static mktime."""
        dt = datetime(year, mon, day, hr, minute, sec)
        return time.mktime(dt.timetuple())


@dataclass
class DateEncoderParameters:
    """Configuration parameters for DateEncoder.

    Each field controls the encoding of a specific temporal feature.
    Set the corresponding width to a nonzero value to enable encoding for that feature.

    Important Constraints:
        - Year and Season encoders are mutually exclusive (XOR). Enable one or the other, not both.
          * Year: Encodes absolute year value (e.g., 2020, 2021)
          * Season: Encodes day-of-year (0-365, cyclic within a year)

    Atrtibutes:

        year_size: Size of the year encoder (total bits).
        year_active_bits: Number of active bits for absolute year (mutually exclusive with season_active_bits).
        year_sparsity: Sparsity for year encoding.
        year_radius: Radius for year encoding, in years.
        year_resolution: Resolution for year encoding.
        year_min: Minimum year value for encoding range.
        year_max: Maximum year value for encoding range.
        season_size: Size of the season encoder (total bits).
        season_active_bits: Number of active bits for season/day of year (mutually exclusive with year_active_bits).
        season_sparsity: Sparsity for season encoding.
        season_radius: Radius for season encoding, in days.
        season_resolution: Resolution for season encoding .
        day_of_week_size: Size of the day of week encoder (total bits).
        day_of_week_active_bits: Number of active bits for day of week.
        day_of_week_sparsity: Sparsity for day of week encoding .
        day_of_week_radius: Radius for day of week encoding.
        day_of_week_resolution: Resolution for day of week encoding.
        weekend_active_bits: Number of active bits for weekend flag.
        weekend_sparsity: Sparsity for weekend encoding .
        weekend_radius: Radius for weekend encoding .
        weekend_resolution: Resolution for weekend encoding .
        holiday_size: Size of the holiday encoder (total bits).
        holiday_active_bits: Number of active bits for holiday encoding.
        holiday_sparsity: Sparsity for holiday encoding .
        holiday_radius: Radius for holiday encoding .
        holiday_resolution: Resolution for holiday encoding .
        holiday_dates: List of holidays as [month, day] or [year, month, day].
        time_of_day_size: Size of the time of day encoder (total bits).
        time_of_day_active_bits: Number of active bits for time of day.
        time_of_day_sparsity: Sparsity for time of day encoding .
        time_of_day_radius: Radius for time of day encoding, in hours.
        time_of_day_resolution: Resolution for time of day encoding .
        custom_size: Size of the custom days encoder (total bits).
        custom_active_bits: Number of active bits for custom day groups.
        custom_sparsity: Sparsity for custom days encoding .
        custom_radius: Radius for custom days encoding .
        custom_resolution: Resolution for custom days encoding .
        custom_days: List of custom day group strings (e.g., ["mon,wed,fri"]).
        rdse_used: Enable RDSE usage for date encoder.

      /** NuPic C++ DateEncoder class
         * The DateEncoderParameters structure is used to pass configuration parameters to
         * the DateEncoder. These Six (6) members define the total number of bits in the output.
         *     Members:  season, dayOfWeek, weekend, holiday, timeOfDay, customDays
         *
         * Each member is a separate attribute of a date/time that can be activated
         * by providing a width parameter and sometimes a radius parameter.
         * Each is implemented separately using a ScalarEncoder and the results
         * are concatinated together.
         *
         * The width attribute determines the number of bits to be used for each member.
         * and 0 means don't use.  The width is like a weighting to indicate the relitive importance
         * of this member to the overall data value.
         *
         * The radius attribute indicates the size of the bucket; the quantization size.
         * All values in the same bucket generate the same pattern.
         *
         * To avoid problems with leap year, consider a year to have 366 days.
         * The timestamp will be converted to components such as time and dst based on
         * local timezone and location (see localtime()).
         *
         */
    """

    # total size
    size: int = 2051
    """Total number of bits in the output SDR. If force_size is False, this is ignored and computed from sub-encoder sizes."""

    # Year
    year_size: int = 293
    """Size of the year encoder (total bits)."""
    year_active_bits: int = 6
    """Set to greater than zero to enable year encoder. Number of active bits for year."""
    year_sparsity: float = 0.0
    """Sparsity for year encoding (not used)."""
    year_radius: float = 0.0
    """Radius for year encoding, in years."""
    year_resolution: float = 1.0
    """Resolution for year encoding (not used)."""

    # Season: day of year (0..366)
    # season size
    season_size: int = 293
    """Size of the season encoder (total bits)."""

    season_active_bits: int = 6
    """Set to greater than zero to enable season encoding. Number of active bits for season (day of year). how many bits to apply to season
       Member: season -  The portion of the year. Unit is day. Range is 0 to 366 (to avoid leap year issues)."""

    # season sparsity
    season_sparsity: float = 0.0
    """Sparsity for season encoding (not used)."""

    # season radius in days
    season_radius: float = 73.25
    """Radius for season encoding, in days (default ~4 seasons) days per season."""

    # season  resoulation
    season_resolution: float = 0.0
    """Resolution for season encoding (not used)."""

    # --------------------------------------------------------------------------

    # day of week size
    day_of_week_size: int = 293
    """Size of the day of week encoder (total bits)."""

    # day of week active bits
    day_of_week_active_bits: int = 6
    """Set to greater than zero to enable day of week encoding. Number of active bits for day of week, how many bits to apply to day of week."""

    # day of week sparsity
    day_of_week_sparsity: float = 0.0
    """Sparsity for day of week encoding (not used)."""

    # day of week radius
    day_of_week_radius: float = 41.85
    """Radius for day of week encoding, every day is a separate bucket."""

    # day of week resolution
    day_of_week_resolution: float = 0.0
    """Resolution for day of week encoding (not used)."""

    # --------------------------------------------------------------------------

    # Weekend flag (0/1, Fri 6pm through Sun midnight)
    weekend_size: int = 293
    """Size of the weekend encoder (total bits)."""

    # weekend active bits
    weekend_active_bits: int = 6
    """Set to greater than zero to enable weekend encoding. Number of active bits for weekend flag."""
    # weekend sparsity
    weekend_sparsity: float = 0.0
    """Sparsity for weekend encoding (not used)."""

    # weekend radius
    weekend_radius: float = 1.0
    """Radius for weekend encoding (not used)."""

    # weekend resolution
    weekend_resolution: float = 0.0
    """Resolution for weekend encoding (not used)."""

    # --------------------------------------------------------------------------

    # holiday active bits
    holiday_size: int = 293
    """Size of the holiday encoder (total bits)."""

    holiday_active_bits: int = 6
    """Set to greater than zero to enable holiday encoding. Number of active bits for holiday encoding."""

    # holiday sparsity
    holiday_sparsity: float = 0.0
    """Sparsity for holiday encoding (not used)."""

    # holiday radius
    holiday_radius: float = 1.0
    """Radius for holiday encoding (not used)."""

    # holiday resolution
    holiday_resolution: float = 0.0
    """Resolution for holiday encoding (not used)."""

    holiday_dates: list[list[int]] = field(default_factory=lambda: [[12, 25]])
    """List of holidays as [month, day] or [year, month, day]."""

    # --------------------------------------------------------------------------

    # Time of day: 0..24 hours
    # time of day size
    time_of_day_size: int = 293
    """Size of the time of day encoder (total bits)."""

    # time of day active bits
    time_of_day_active_bits: int = 6
    """Set to greater than zero to enable time of day encoding. Number of active bits for time of day."""

    # time of day sparsity
    time_of_day_sparsity: float = 0.0
    """Sparsity for time of day encoding (not used)."""

    # time of day radius
    time_of_day_radius: float = 12.20
    """Radius for time of day encoding, in hours."""

    # time of day resolution
    time_of_day_resolution: float = 0.0
    """Resolution for time of day encoding (not used)."""

    # --------------------------------------------------------------------------

    # custom days size
    custom_size: int = 293
    """Size of the custom days encoder (total bits)."""

    # custom days active bits
    custom_active_bits: int = 6
    """Set to greater than zero to enable custom days encoding. Number of active bits for custom day groups."""

    # custom days sparsity
    custom_sparsity: float = 0.0
    """Sparsity for custom days encoding (not used)."""

    # custom days radius
    custom_radius: float = 58.6
    """Radius for custom days encoding (not used)."""

    # custom days resolution
    custom_resolution: float = 0.0
    """Resolution for custom days encoding (not used)."""

    # Custom day groups (e.g. ["mon,wed,fri"])
    custom_days: list[str] = field(default_factory=lambda: ["mon,tue,wed,thu,fri"])
    """List of custom day group strings (e.g., ["mon,wed,fri"])."""

    # seed for encoders
    seed: int = 42
    """Random seed for encoder initialization."""

    rdse_used: bool = True
    """Enable RDSE usage for date encoder."""

    encoder_class = DateEncoder
    """Reference to the DateEncoder class for factory instantiation."""


# ---------------------------------------------------------------------------------------


if __name__ == "__main__":

    date_encoder = DateEncoder()
    print("DateEncoder size:", date_encoder.size)

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
        [1988, 5, 27, 11, 0],
    ]

    actual_encoding = []

    output_sdr: list[int] = []
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        actual_encoding.append(encoding)

    for output in actual_encoding:
        decode_tuple = date_encoder.decode(output)
        date_encoder._logger.info(decode_tuple)
