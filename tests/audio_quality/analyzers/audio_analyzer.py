"""
Audio Analyzer Module

Provides audio quality measurement utilities for:
- THD (Total Harmonic Distortion)
- SNR (Signal-to-Noise Ratio)
- Frequency Response
- DC Offset Detection
- Aliasing Detection

Based on SonicEval methodology and standard audio measurement practices.
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class THDResult:
    """Total Harmonic Distortion measurement result."""
    thd_percent: float
    thd_db: float
    fundamental_freq: float
    fundamental_amplitude: float
    harmonics: List[Tuple[float, float]]  # (frequency, amplitude) pairs


@dataclass
class SNRResult:
    """Signal-to-Noise Ratio measurement result."""
    snr_db: float
    signal_power: float
    noise_power: float
    noise_floor_db: float


@dataclass
class FrequencyResponseResult:
    """Frequency response measurement result."""
    frequencies: np.ndarray
    magnitudes_db: np.ndarray
    phase_degrees: np.ndarray
    bandwidth_3db: Tuple[float, float]  # (low, high) cutoff frequencies


class AudioAnalyzer:
    """
    Audio quality measurement and analysis.

    Provides standardized measurements for audio plugin validation.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def measure_thd(
        self,
        signal: np.ndarray,
        fundamental_freq: Optional[float] = None,
        num_harmonics: int = 10,
        window: str = 'hann'
    ) -> THDResult:
        """
        Measure Total Harmonic Distortion.

        THD = sqrt(sum(V_harmonics^2)) / V_fundamental * 100%

        Args:
            signal: Audio signal array
            fundamental_freq: Expected fundamental (auto-detected if None)
            num_harmonics: Number of harmonics to measure
            window: Window function for FFT

        Returns:
            THDResult with measurements
        """
        # Apply window
        windowed = signal * signal.windows.get_window(window, len(signal))

        # Compute FFT
        fft_result = rfft(windowed)
        freqs = rfftfreq(len(windowed), 1 / self.sample_rate)
        magnitudes = np.abs(fft_result) / len(signal)

        # Find fundamental frequency
        if fundamental_freq is None:
            # Auto-detect from highest peak
            fund_idx = np.argmax(magnitudes[1:]) + 1
            fundamental_freq = freqs[fund_idx]
        else:
            fund_idx = np.argmin(np.abs(freqs - fundamental_freq))

        fundamental_amp = magnitudes[fund_idx]

        # Find harmonics
        harmonics = []
        harmonic_power = 0.0

        for n in range(2, num_harmonics + 2):
            harmonic_freq = fundamental_freq * n
            if harmonic_freq >= self.sample_rate / 2:
                break

            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonic_amp = magnitudes[harmonic_idx]
            harmonics.append((harmonic_freq, harmonic_amp))
            harmonic_power += harmonic_amp ** 2

        # Calculate THD
        thd_ratio = np.sqrt(harmonic_power) / fundamental_amp if fundamental_amp > 0 else 0
        thd_percent = thd_ratio * 100
        thd_db = 20 * np.log10(thd_ratio) if thd_ratio > 0 else -np.inf

        return THDResult(
            thd_percent=thd_percent,
            thd_db=thd_db,
            fundamental_freq=fundamental_freq,
            fundamental_amplitude=fundamental_amp,
            harmonics=harmonics
        )

    def measure_snr(
        self,
        signal_with_content: np.ndarray,
        noise_only: Optional[np.ndarray] = None,
        signal_freq: Optional[float] = None
    ) -> SNRResult:
        """
        Measure Signal-to-Noise Ratio.

        Args:
            signal_with_content: Signal containing both signal and noise
            noise_only: Pure noise sample (optional, for direct measurement)
            signal_freq: Frequency of signal component (for spectral method)

        Returns:
            SNRResult with measurements
        """
        if noise_only is not None:
            # Direct measurement from separate noise sample
            signal_power = np.mean(signal_with_content ** 2)
            noise_power = np.mean(noise_only ** 2)
        elif signal_freq is not None:
            # Spectral method - extract signal power at fundamental
            fft_result = rfft(signal_with_content)
            freqs = rfftfreq(len(signal_with_content), 1 / self.sample_rate)
            magnitudes = np.abs(fft_result)

            # Signal power at fundamental (and harmonics)
            fund_idx = np.argmin(np.abs(freqs - signal_freq))
            signal_power = (magnitudes[fund_idx] / len(signal_with_content)) ** 2

            # Noise power is everything else
            total_power = np.mean(signal_with_content ** 2)
            noise_power = total_power - signal_power
        else:
            # Estimate from signal statistics
            signal_power = np.mean(signal_with_content ** 2)
            # Use floor of magnitude spectrum as noise estimate
            fft_result = rfft(signal_with_content)
            magnitudes = np.abs(fft_result)
            noise_floor = np.percentile(magnitudes, 10)
            noise_power = (noise_floor / len(signal_with_content)) ** 2

        # Calculate SNR
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
            noise_floor_db = 10 * np.log10(noise_power)
        else:
            snr_db = np.inf
            noise_floor_db = -np.inf

        return SNRResult(
            snr_db=snr_db,
            signal_power=signal_power,
            noise_power=noise_power,
            noise_floor_db=noise_floor_db
        )

    def measure_frequency_response(
        self,
        impulse_response: np.ndarray,
        freq_range: Tuple[float, float] = (20, 20000)
    ) -> FrequencyResponseResult:
        """
        Measure frequency response from impulse response.

        Args:
            impulse_response: Impulse response of the system
            freq_range: Frequency range of interest

        Returns:
            FrequencyResponseResult with magnitude and phase
        """
        # Compute frequency response
        freqs, response = signal.freqz(
            impulse_response,
            worN=8192,
            fs=self.sample_rate
        )

        # Filter to frequency range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        response = response[mask]

        # Magnitude in dB
        magnitudes_db = 20 * np.log10(np.abs(response) + 1e-10)

        # Phase in degrees
        phase_degrees = np.angle(response, deg=True)

        # Find -3dB bandwidth
        max_mag = np.max(magnitudes_db)
        cutoff_level = max_mag - 3

        above_cutoff = magnitudes_db >= cutoff_level
        if np.any(above_cutoff):
            indices = np.where(above_cutoff)[0]
            low_freq = freqs[indices[0]]
            high_freq = freqs[indices[-1]]
        else:
            low_freq = high_freq = 0

        return FrequencyResponseResult(
            frequencies=freqs,
            magnitudes_db=magnitudes_db,
            phase_degrees=phase_degrees,
            bandwidth_3db=(low_freq, high_freq)
        )

    def measure_dc_offset(self, signal: np.ndarray) -> float:
        """
        Measure DC offset in signal.

        Returns:
            DC offset as fraction of full scale
        """
        return np.mean(signal)

    def detect_aliasing(
        self,
        signal: np.ndarray,
        expected_freq: float,
        threshold_db: float = -80
    ) -> Tuple[bool, List[Tuple[float, float]]]:
        """
        Detect aliasing artifacts in signal.

        Looks for unexpected frequency content that could indicate aliasing.

        Args:
            signal: Audio signal
            expected_freq: Expected fundamental frequency
            threshold_db: Detection threshold

        Returns:
            Tuple of (aliasing_detected, [(freq, amplitude_db), ...])
        """
        fft_result = rfft(signal)
        freqs = rfftfreq(len(signal), 1 / self.sample_rate)
        magnitudes = np.abs(fft_result) / len(signal)
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)

        # Expected frequencies (fundamental and harmonics)
        expected_freqs = set()
        for n in range(1, 50):
            f = expected_freq * n
            if f < self.sample_rate / 2:
                expected_freqs.add(f)

        # Find unexpected peaks
        peaks, properties = signal.find_peaks(magnitudes_db, height=threshold_db)
        unexpected = []

        for peak_idx in peaks:
            peak_freq = freqs[peak_idx]
            peak_amp = magnitudes_db[peak_idx]

            # Check if this is near an expected frequency
            is_expected = any(
                abs(peak_freq - ef) < 10  # 10 Hz tolerance
                for ef in expected_freqs
            )

            if not is_expected and peak_amp > threshold_db:
                unexpected.append((peak_freq, peak_amp))

        aliasing_detected = len(unexpected) > 0
        return aliasing_detected, unexpected

    def measure_rms(self, signal: np.ndarray) -> float:
        """Calculate RMS level of signal."""
        return np.sqrt(np.mean(signal ** 2))

    def measure_peak(self, signal: np.ndarray) -> float:
        """Calculate peak level of signal."""
        return np.max(np.abs(signal))

    def measure_crest_factor(self, signal: np.ndarray) -> float:
        """Calculate crest factor (peak/RMS ratio) in dB."""
        rms = self.measure_rms(signal)
        peak = self.measure_peak(signal)
        if rms > 0:
            return 20 * np.log10(peak / rms)
        return np.inf


class SpectrumAnalyzer:
    """Real-time spectrum analysis utilities."""

    def __init__(self, sample_rate: int = 48000, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.fft_size = fft_size

    def compute_spectrum(
        self,
        signal: np.ndarray,
        window: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude spectrum.

        Returns:
            Tuple of (frequencies, magnitudes_db)
        """
        # Pad or truncate to FFT size
        if len(signal) < self.fft_size:
            padded = np.zeros(self.fft_size)
            padded[:len(signal)] = signal
            signal = padded
        else:
            signal = signal[:self.fft_size]

        # Apply window
        windowed = signal * np.hanning(len(signal))

        # Compute FFT
        fft_result = rfft(windowed)
        freqs = rfftfreq(len(windowed), 1 / self.sample_rate)
        magnitudes = np.abs(fft_result) / len(signal)
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)

        return freqs, magnitudes_db

    def find_peaks(
        self,
        freqs: np.ndarray,
        magnitudes_db: np.ndarray,
        threshold_db: float = -60,
        min_distance_hz: float = 50
    ) -> List[Tuple[float, float]]:
        """
        Find spectral peaks.

        Returns:
            List of (frequency, magnitude_db) pairs
        """
        # Convert minimum distance to samples
        freq_resolution = freqs[1] - freqs[0]
        min_distance = int(min_distance_hz / freq_resolution)

        peaks, properties = signal.find_peaks(
            magnitudes_db,
            height=threshold_db,
            distance=max(1, min_distance)
        )

        return [(freqs[i], magnitudes_db[i]) for i in peaks]
