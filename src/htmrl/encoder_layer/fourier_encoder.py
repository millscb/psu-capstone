"""Fourier Transform-based encoder for time-series frequency analysis.

This module provides a FourierEncoder that transforms time-domain signals
into frequency-domain representations using Fast Fourier Transform (FFT).
It then encodes specified frequency ranges using RDSE encoders, enabling
HTM systems to learn from spectral features of time-series data.

Mathematical foundation:
- Continuous: G(f) = ∫ g(t) * exp(-2πift) dt
- Discrete: X_k = Σ x_n * exp(-2πikn/N)

References:
- https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
- https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, cast, override

import numpy as np
from scipy.fft import fft, fftfreq

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.log import logger


class FourierEncoder(BaseEncoder[np.ndarray], list[int]):
    """Encoder that uses Fourier Transform on time data. Build RDSE for each frequency component.

    Assume that time domain is in seconds.

    Args:
        parameters: Fourier encoder parameters. Defaults to FourierEncoderParameters().
    """

    def __init__(self, parameters: FourierEncoderParameters):

        if parameters is None:
            parameters = FourierEncoderParameters()

        # set the size of the base encoder
        super().__init__(parameters.size)

        self._params = copy.deepcopy(parameters)
        """Fourier encoder local copy of passed parameters."""
        self._sdrs: list[list[int]] = []
        """List to hold the SDRs for each frequency range."""
        # encoder params
        self._sensitivity = self._params.sensitivity_threshold
        """Magnitude threshold for considering a frequency component as a peak."""
        self._frequency_ranges = self._params.frequency_ranges
        """List of frequency ranges to find."""
        self._size = self._params.size  # default to 2048
        """Size of the encoder."""
        self._total_active_bits = self._params.total_active_bits  # default to 40
        """Total number of active bits in the encoder output SDR."""
        self._active_bits_in_ranges = self._params.active_bits_in_ranges
        """The number of active bits per frequency range in the encoder output SDR."""
        self._sparsity_in_ranges = self._params.sparsity_in_ranges
        """The sparsity per frequency range in the encoder output SDR."""
        self._resolutions_in_ranges = self._params.resolutions_in_ranges
        """The resolution per frequency range in the encoder output SDR."""
        self._seed = self._params.seed  # default to 32
        """Random seed for reproducibility."""
        self._bucket_sizes: list[int] = []
        """The frequency intervals."""

        # time domain params
        self._start_time = self._params.start_time  # default to 0.0
        """Start time of the time interval."""
        self._stop_time = self._params.stop_time  # default to 1.0
        """Stop time of the time interval."""
        self._period_size = self._stop_time - self._start_time
        """Total time period size in seconds."""
        self._total_samples = self._params.total_samples  # default to 256
        """Total number of samples in the period."""
        self._time_step = self._period_size / self._total_samples
        """Time step between samples."""
        self._sample_rate = float(1 / self._time_step)  # samples per second
        """Sample rate in Hz."""

        self._validate_params(self._params)
        self._calc_bucket_sizes()

    @property
    def sdrs(self) -> list[list[int]]:
        """Get the list of SDRs for each frequency range."""
        return self._sdrs

    def _normalize(self, input: np.ndarray) -> np.ndarray:
        """Normalize the FFT array to a unit vector."""

        # Calculate the normal vector norm (L2 norm)
        norm = np.linalg.norm(input)

        unit_vector = input / norm if norm != 0 else input

        return unit_vector

    def _trim(self, input_data: np.ndarray | list[float]) -> np.ndarray:
        """Trim input array into a power of 2 size"""

        # verify sizes of input data
        self._total_samples = len(input_data)
        self._time_step = self._period_size / self._total_samples
        self._sample_rate = float(1 / self._time_step)  # samples per second

        if isinstance(input_data, np.ndarray):
            input_data = input_data.astype(float, copy=False)

        else:
            input_data = np.asarray(input_data, dtype=float)

        # Always operate on a flattened 1D array so FFT math works correctly
        input_data = np.asarray(input_data, dtype=float).reshape(-1)

        total_samples = len(input_data)

        if total_samples & (total_samples - 1) != 0:

            target_size = 1 << (total_samples - 1).bit_length()
            padding = target_size - total_samples
            input_data = np.pad(input_data, (0, padding), mode="constant")

            logger.info(
                (
                    "FFT input adjusted:\n"
                    f"  • Original samples: {total_samples}\n"
                    f"  • Target samples:   {target_size}\n"
                    f"  • Zero padding:     {padding}\n"
                )
            )

        return copy.deepcopy(input_data)

    def _calc_bucket_sizes(self) -> None:
        """Calculate the bucket sizes for each frequency range."""

        for i in range(len(self._frequency_ranges)):

            freq_range = self._frequency_ranges[i]
            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            bucket_size = stop_freq - start_freq

            self._bucket_sizes.append(bucket_size)

    def _find_num_peaks(self, freq_data: np.ndarray) -> int:
        """Find the number of peaks in the given frequency data."""

        num_peaks = 0
        threshold = self._sensitivity

        for i in range(len(freq_data)):
            if freq_data[i] > threshold:
                num_peaks += 1
        return num_peaks

    @override
    def encode(self, input_value: Any) -> list[int]:
        """Transform the input signal via FFT and populate the provided SDR.

        Encodes a single frequency peak into a SDR as list[int].

        Args:
            input_value: np.ndarray | list[float]: The input time-domain signal to be encoded.

        Returns:
            list[int]: List of active bit indices in the encoded SDR.

        Raises:

            ValueError: If the input signal is not a 1D array or if the encoder parameters are invalid.

        """

        # validate parameters

        self._validate_params(self._params)

        if not isinstance(input_value, (np.ndarray, list)):
            raise ValueError("Input value must be a numpy array or a list of floats.")

        #  trim input data into a power of 2 size and reset internal params
        input_value = self._trim(cast(np.ndarray, input_value))

        # subtract mean to center the signal around zero
        input_value = input_value - np.mean(input_value)

        samples = self._total_samples
        time_step = self._time_step
        freq_ranges = self._frequency_ranges
        threshold = self._sensitivity

        dense_bits: list[int] = []

        #  list to hold values if we want to perform a mean later
        magnitudes = []

        #  perform FFT on input time data and normalize
        freq_data = cast(np.ndarray, fft(input_value))

        # Remove the DC bucket so normalization is not dominated by zero Hz energy
        if freq_data.size > 0:
            freq_data[0] = 0
        freq_data = self._normalize(freq_data)

        # Nyquist frequency limit and store freq buckets as indexes
        freq_data = freq_data[: samples // 2]

        all_freqs = np.real(np.abs(freq_data))
        all_peaks = self._find_num_peaks(all_freqs)
        logger.info(f"Total peaks found in signal: {all_peaks}")

        # LOOP THROUGH FREQUENCY RANGES !!
        print("Looping through frequency ranges:")
        for freq_range in freq_ranges:
            # reset total size
            size = self._size

            current_dense_bits: list[int] = []

            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            if stop_freq > (self._sample_rate / 2):
                logger.warning(
                    f"Frequency range end {stop_freq} exceeds Nyquist frequency {self._sample_rate / 2} Hz. Adjusting to Nyquist limit."
                )
                stop_freq = int(self._sample_rate // 2)

            freq_buckets = fftfreq(samples, time_step)[start_freq:stop_freq]

            # slice freq data to current frequency range  :: assuming freq ranges are in Hz
            freq_slice = freq_data[start_freq:stop_freq]
            freq_slice = np.real(np.abs(freq_slice))

            # find peak frequency in the current interval
            peak_index = np.argmax(np.abs(freq_slice))
            if peak_index == 0:
                logger.warning(f"!!-ZERO Peak Found in range--!! {freq_range}.")

            if freq_slice[peak_index] < threshold:
                logger.info(f"No significant peaks found in range {freq_range}.")

                continue

            else:
                logger.info(
                    f"FFT Peak Frequency in range {freq_range}: {float(freq_buckets[peak_index])} Hz with magnitude {freq_slice[peak_index]}"
                )

            # current resolution for this frequency range
            current_res = self._resolutions_in_ranges[self._frequency_ranges.index(freq_range)]
            current_interval_size = self._bucket_sizes[self._frequency_ranges.index(freq_range)]
            current_sparsity = self._sparsity_in_ranges[self._frequency_ranges.index(freq_range)]

            current_interval_size = min(current_interval_size, len(freq_slice))

            num_peaks = self._find_num_peaks(freq_slice)
            logger.info(f"Number of peaks found in range {freq_range}: {num_peaks}")

            size = (
                size // (num_peaks + 1) if num_peaks >= 1 else size
            )  # take the floor of the division to ensure size is an integer

            if size % 2 != 0:
                size += 1  # ensure size is even for FFT symmetry

            # build each encoder for frequency and magnitude :: thank you GC
            freq_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(
                    size=size,
                    sparsity=current_sparsity,  # sparisity requested by user
                    active_bits=0,
                    resolution=current_res,
                )
            )

            # encode the frequency range slice
            for f in range(current_interval_size):

                # find frequencies that contributed to the fft output by magnitude threshold
                if freq_slice[f] < threshold:

                    continue

                else:

                    freq_value = float(f + start_freq)

                    sdr_freq = freq_encoder.encode(freq_value)
                    magnitudes.append(freq_slice[f])

                    logger.info(
                        f"Encoding frequency bucket: {f + start_freq} with magnitude: {freq_slice[f]}"
                    )
                    current_dense_bits.extend(sdr_freq)

            # END of INNER for loop :: freq intervals

            #

            magnitude_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(
                    size=size,
                    sparsity=current_sparsity,
                    active_bits=0,
                    resolution=time_step,
                )
            )

            magnitude = float(np.mean(magnitudes))

            print(f"Mean magnitude: {magnitude}")

            sdr_magnitude = magnitude_encoder.encode(magnitude)

            current_dense_bits.extend(sdr_magnitude)

            if len(current_dense_bits) > self._size:
                logger.warning(
                    f"Encoded SDR size {len(current_dense_bits)} exceeds specified encoder size {self._size}. Consider increasing encoder size or adjusting sparsity/resolution parameters."
                )
                current_dense_bits = current_dense_bits[: self._size]

            elif len(current_dense_bits) < self._size:
                # pad with zeros if we have less bits than the encoder size
                current_dense_bits.extend([0] * (self._size - len(current_dense_bits)))
            dense_bits.extend(current_dense_bits)

        # END of OUTER for loop
        if len(dense_bits) > self.size:
            logger.warning(
                f"Total encoded SDR size {len(dense_bits)} exceeds specified encoder size {self.size}. Consider increasing encoder size or adjusting sparsity/resolution parameters."
            )

            number_of_sdrs = len(dense_bits) // self.size

            dense_list = [
                dense_bits[i * self.size : (i + 1) * self.size] for i in range(number_of_sdrs)
            ]

            self._sdrs = dense_list
        return dense_bits
        # END def encode

    @override
    def decode(
        self, encoded: list[int], candidates: Iterable[float] | None = None
    ) -> dict[str, list[tuple[tuple[int, int], float | None, float]] | tuple[float | None, float]]:
        """Decode a combined Fourier SDR into per-range frequency estimates and magnitude."""
        if len(encoded) != self.size:
            raise ValueError(
                f"Encoded input size ({len(encoded)}) does not match encoder size ({self.size})"
            )

        results: dict[
            str, list[tuple[tuple[int, int], float | None, float]] | tuple[float | None, float]
        ] = {"frequencies": []}

        for range_index, freq_range in enumerate(self._frequency_ranges):
            start_freq, stop_freq = freq_range
            current_res = self._resolutions_in_ranges[range_index]
            current_sparsity = self._sparsity_in_ranges[range_index]

            freq_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(
                    size=self._size,
                    sparsity=current_sparsity,
                    active_bits=0,
                    resolution=current_res,
                )
            )

            if candidates is not None:
                range_candidates = [
                    candidate for candidate in candidates if start_freq <= candidate < stop_freq
                ]
                if not range_candidates:
                    range_candidates = list(range(start_freq, stop_freq))
            else:
                range_candidates = list(range(start_freq, stop_freq))
            decoded_value, confidence = freq_encoder.decode(encoded, range_candidates)
            results["frequencies"].append((freq_range, decoded_value, confidence))

        magnitude_candidates: list[float] = []
        if self._time_step > 0:
            magnitude_candidates = list(
                np.arange(0.0, 1.0 + self._time_step, self._time_step, dtype=float)
            )

        if magnitude_candidates:
            magnitude_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(
                    size=self._size,
                    sparsity=current_sparsity,
                    active_bits=0,
                    resolution=current_res,
                )
            )
            results["magnitude"] = magnitude_encoder.decode(encoded, magnitude_candidates)

        return results

    def _validate_params(self, parameters: FourierEncoderParameters) -> None:
        """Check if the provided parameters are valid for the Fourier encoder.

        Args:
            parameters: The Fourier encoder parameters to check.

        Raises:
            ValueError: If parameters are invalid or inconsistent.
        """

        if parameters is None:
            raise ValueError("FourierEncoderParameters cannot be None.")

        params = copy.deepcopy(parameters)

        if params.size <= 0:
            raise ValueError("Encoder size must be greater than zero.")
        if params.total_samples <= 0:
            raise ValueError("total_samples must be greater than zero.")
        if params.stop_time <= params.start_time:
            raise ValueError("stop_time must be greater than start_time.")
        if params.seed < 0:
            raise ValueError("seed must be non-negative.")
        if params.sensitivity_threshold < 0:
            raise ValueError("sensitivity_threshold must be non-negative.")

        num_ranges = len(params.frequency_ranges)
        if num_ranges == 0:
            raise ValueError("At least one frequency range must be provided.")

        # Active bits and sparsity are mutually exclusive inputs, but normalized params
        # can legitimately contain both when they are equivalent representations.
        if params.active_bits_in_ranges and params.sparsity_in_ranges:
            same_length = len(params.active_bits_in_ranges) == len(params.sparsity_in_ranges)
            equivalent = same_length and all(
                int(round(sparsity * params.size)) == active_bits
                for active_bits, sparsity in zip(
                    params.active_bits_in_ranges,
                    params.sparsity_in_ranges,
                    strict=True,
                )
            )
            if not equivalent:
                raise ValueError(
                    "Cannot specify both active_bits_in_ranges and sparsity_in_ranges. Choose one."
                )

        # Normalize per-range resolutions: allow a single global value.
        if len(params.resolutions_in_ranges) == 0:
            raise ValueError("resolutions_in_ranges must contain at least one value.")
        if len(params.resolutions_in_ranges) == 1 and num_ranges > 1:
            params.resolutions_in_ranges = [params.resolutions_in_ranges[0]] * num_ranges
        elif len(params.resolutions_in_ranges) != num_ranges:
            raise ValueError(
                "resolutions_in_ranges must either have one global value or match frequency_ranges length."
            )

        if any(resolution <= 0 for resolution in params.resolutions_in_ranges):
            raise ValueError("All resolutions_in_ranges values must be greater than zero.")

        # Normalize sparsity/active_bits to per-range values.
        if params.sparsity_in_ranges:
            if len(params.sparsity_in_ranges) == 1 and num_ranges > 1:
                params.sparsity_in_ranges = [params.sparsity_in_ranges[0]] * num_ranges
            elif len(params.sparsity_in_ranges) != num_ranges:
                raise ValueError(
                    "sparsity_in_ranges must either have one global value or match frequency_ranges length."
                )

            if any(sparsity <= 0 or sparsity > 1 for sparsity in params.sparsity_in_ranges):
                raise ValueError("Each sparsity value must be in the interval (0, 1].")

            params.active_bits_in_ranges = [
                int(round(sparsity * params.size)) for sparsity in params.sparsity_in_ranges
            ]

            if any(active_bits <= 0 for active_bits in params.active_bits_in_ranges):
                raise ValueError(
                    "sparsity_in_ranges and size must produce active_bits_in_ranges > 0."
                )

        elif params.active_bits_in_ranges:
            if len(params.active_bits_in_ranges) == 1 and num_ranges > 1:
                params.active_bits_in_ranges = [params.active_bits_in_ranges[0]] * num_ranges
            elif len(params.active_bits_in_ranges) != num_ranges:
                raise ValueError(
                    "active_bits_in_ranges must either have one global value or match frequency_ranges length."
                )

            if any(active_bits <= 0 for active_bits in params.active_bits_in_ranges):
                raise ValueError("All active_bits_in_ranges values must be greater than zero.")
            if any(active_bits > params.size for active_bits in params.active_bits_in_ranges):
                raise ValueError("active_bits_in_ranges values cannot exceed encoder size.")

            params.sparsity_in_ranges = [
                active_bits / params.size for active_bits in params.active_bits_in_ranges
            ]

        else:
            if params.total_active_bits > 0:
                if params.total_active_bits > params.size:
                    raise ValueError("total_active_bits cannot exceed encoder size.")
                params.sparsity_in_ranges = [params.total_active_bits / params.size] * num_ranges
                params.active_bits_in_ranges = [params.total_active_bits] * num_ranges
            else:
                raise ValueError(
                    "Provide sparsity_in_ranges, active_bits_in_ranges, or total_active_bits > 0."
                )

        sample_rate = float(params.total_samples / (params.stop_time - params.start_time))
        nyquist = sample_rate / 2

        normalized_ranges = sorted(params.frequency_ranges, key=lambda value: value[0])
        previous_end: int | None = None

        for start_freq, stop_freq in normalized_ranges:
            if start_freq < 0 or stop_freq < 0:
                raise ValueError("Frequency ranges must be non-negative.")
            if start_freq >= stop_freq:
                raise ValueError("Frequency range start must be less than end.")
            if stop_freq > nyquist:
                raise ValueError(
                    f"Frequency range end {stop_freq} exceeds Nyquist frequency {nyquist} Hz."
                )

            if previous_end is not None and start_freq < previous_end:
                raise ValueError("Frequency ranges must not overlap.")
            previous_end = stop_freq

        predicted_sdr_size = params.size * num_ranges
        if predicted_sdr_size % params.size != 0:
            raise ValueError(
                "Predicted SDR shape is inconsistent with encoder size and frequency range count."
            )

        # Persist normalized values so encode/decode can rely on validated per-range lists.
        parameters.resolutions_in_ranges = params.resolutions_in_ranges
        parameters.sparsity_in_ranges = params.sparsity_in_ranges
        parameters.active_bits_in_ranges = params.active_bits_in_ranges


@dataclass
class FourierEncoderParameters:
    """Class to hold fourier encodder parameters

    parameters:
        samples : total number of samples or bucket size -> N
        size : size of the encoder output SDR
        start_time : start time of the time interval -> integral lower bound
        stop_time : stop time of the time interval -> integral upper bound
        interval_size : total number of buckets in the time interval -> N
        time_step_n : the curreent value of x at time n -> n

    """

    # reference to Encoder
    encoder_class = FourierEncoder
    """Reference to the FourierEncoder class."""

    size: int = 2048
    """The size the encoder"""

    total_active_bits: int = 0
    """Total number of active bits in the encoder output SDR."""

    frequency_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(0, 1024)])
    """List of frequency ranges to find."""

    active_bits_in_ranges: list[int] = field(default_factory=lambda: [])
    """The number of active bits per frequency range in the encoder output SDR."""

    sparsity_in_ranges: list[float] = field(default_factory=lambda: [0.02])
    """The sparsity per frequency range in the encoder output SDR."""

    resolutions_in_ranges: list[float] = field(default_factory=lambda: [1.0])
    """The resolution per frequency range in the encoder output SDR."""

    seed: int = 32
    """Random seed for reproducibility."""

    sensitivity_threshold: float = 0.1
    """Magnitude threshold for considering a frequency component as a peak."""

    #  time domain params
    start_time: float = 0.0
    """Start time of the time interval."""

    stop_time: float = 1.0
    """Stop time of the time interval."""

    total_samples: int = 2048
    """Total number of samples in the period."""
