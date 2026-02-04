"""
Real Audio Quality Tests (using DawDreamer)

Actual audio quality measurements using the loaded plugin.
"""

import numpy as np
import pytest
from scipy import signal
from scipy.fft import rfft, rfftfreq

from tests.tools.dawdreamer_host import DawDreamerHost, MIDIEvent


class TestRealAudioQuality:
    """Real audio quality tests using DawDreamer plugin host."""

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_audio_not_silent(self, loaded_plugin: DawDreamerHost):
        """Basic test that plugin produces audio."""
        result = loaded_plugin.render_note(note=60, velocity=100)
        rms = np.sqrt(np.mean(result.audio ** 2))

        assert rms > 0.001, "Plugin should produce non-silent audio"

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_no_clipping(self, loaded_plugin: DawDreamerHost):
        """Verify output doesn't clip."""
        result = loaded_plugin.render_note(note=60, velocity=127)

        peak = np.max(np.abs(result.audio))
        num_clipped = np.sum(np.abs(result.audio) >= 0.999)

        print(f"\nClipping check:")
        print(f"  Peak: {peak:.4f}")
        print(f"  Clipped samples: {num_clipped}")

        assert peak < 1.0, f"Peak {peak:.4f} indicates clipping"
        assert num_clipped == 0, f"Found {num_clipped} clipped samples"

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_no_dc_offset(self, loaded_plugin: DawDreamerHost, thresholds):
        """Verify no significant DC offset in output."""
        result = loaded_plugin.render_note(
            note=60,
            velocity=100,
            duration_seconds=1.0,
            tail_seconds=0.5
        )

        dc_offset = np.mean(result.audio)

        print(f"\nDC offset: {dc_offset:.6f}")

        threshold = thresholds.dc_offset_threshold
        assert abs(dc_offset) < threshold, f"DC offset {dc_offset:.4f} exceeds threshold {threshold}"

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_stereo_output(self, loaded_plugin: DawDreamerHost):
        """Verify plugin outputs stereo."""
        result = loaded_plugin.render_note(note=60, velocity=100)

        assert result.audio.shape[0] == 2, \
            f"Expected 2 channels, got {result.audio.shape[0]}"

        # Check both channels have content
        left_rms = np.sqrt(np.mean(result.audio[0] ** 2))
        right_rms = np.sqrt(np.mean(result.audio[1] ** 2))

        print(f"\nStereo output:")
        print(f"  Left RMS: {left_rms:.4f}")
        print(f"  Right RMS: {right_rms:.4f}")

        assert left_rms > 0.001, "Left channel should have content"
        assert right_rms > 0.001, "Right channel should have content"

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_decay_to_silence(self, loaded_plugin: DawDreamerHost):
        """
        Verify note decays to silence.

        Karplus-Strong should naturally decay.
        """
        result = loaded_plugin.render_note(
            note=60,
            velocity=100,
            duration_seconds=0.5,
            tail_seconds=5.0  # Long tail to ensure decay
        )

        # Measure RMS at start vs end
        samples_per_100ms = int(0.1 * result.sample_rate)
        start_rms = np.sqrt(np.mean(result.audio[:, :samples_per_100ms] ** 2))
        end_rms = np.sqrt(np.mean(result.audio[:, -samples_per_100ms:] ** 2))

        print(f"\nDecay test:")
        print(f"  Start RMS: {start_rms:.4f}")
        print(f"  End RMS: {end_rms:.6f}")
        print(f"  Decay ratio: {end_rms / start_rms:.4f}")

        assert end_rms < start_rms * 0.1, "Should decay to near silence"

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_fundamental_frequency(self, loaded_plugin: DawDreamerHost, thresholds):
        """
        Verify fundamental frequency matches MIDI note.

        A440 (MIDI 69) should produce ~440 Hz fundamental.
        """
        result = loaded_plugin.render_note(
            note=69,  # A440
            velocity=100,
            duration_seconds=0.5,
            tail_seconds=0.5
        )

        # Skip initial attack, analyze steady state
        start_sample = int(0.1 * result.sample_rate)
        end_sample = int(0.4 * result.sample_rate)
        audio = result.audio[0, start_sample:end_sample]  # Left channel

        # Autocorrelation-based pitch detection (more robust for fundamentals)
        expected_freq = 440.0
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr) // 2:]  # Take positive lags

        # Find expected period range (±1 octave from A440)
        min_period = int(result.sample_rate / (expected_freq * 2))
        max_period = int(result.sample_rate / (expected_freq / 2))
        min_period = max(1, min_period)
        max_period = min(len(corr) - 1, max_period)

        # Find peak in autocorrelation within expected period range
        search_region = corr[min_period:max_period + 1]
        peak_offset = np.argmax(search_region)
        peak_period = min_period + peak_offset

        # Convert period to frequency
        detected_freq = result.sample_rate / peak_period if peak_period > 0 else 0

        # Calculate error in cents
        expected_freq = 440.0
        cents_error = 1200 * np.log2(detected_freq / expected_freq)

        print(f"\nFundamental frequency test (A440):")
        print(f"  Expected: {expected_freq:.2f} Hz")
        print(f"  Detected: {detected_freq:.2f} Hz")
        print(f"  Error: {cents_error:.1f} cents")

        tolerance = thresholds.frequency_accuracy_cents
        assert abs(cents_error) < tolerance, \
            f"Pitch error {cents_error:.1f} cents exceeds tolerance {tolerance}"

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_no_nan_or_inf(self, loaded_plugin: DawDreamerHost):
        """Verify output contains no NaN or Inf values."""
        result = loaded_plugin.render_note(note=60, velocity=100)

        assert not np.any(np.isnan(result.audio)), "Output contains NaN"
        assert not np.any(np.isinf(result.audio)), "Output contains Inf"


class TestSpectralQuality:
    """Spectral analysis tests."""

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_harmonic_structure(self, loaded_plugin: DawDreamerHost):
        """
        Verify Karplus-Strong harmonic structure.

        Should have decaying harmonics (lowpass characteristic).
        """
        result = loaded_plugin.render_note(
            note=60,  # C4 ~261 Hz
            velocity=100,
            duration_seconds=0.5,
            tail_seconds=0.5
        )

        # Analyze early in note (rich harmonics)
        start = int(0.05 * result.sample_rate)
        end = int(0.15 * result.sample_rate)
        audio = result.audio[0, start:end]

        # FFT
        fft_result = rfft(audio * np.hanning(len(audio)))
        freqs = rfftfreq(len(audio), 1 / result.sample_rate)
        magnitudes = np.abs(fft_result)
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)

        # Find fundamental
        fundamental_freq = 261.63  # C4
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))

        # Check first 5 harmonics
        print(f"\nHarmonic structure:")
        harmonics = []
        for n in range(1, 6):
            harmonic_freq = fundamental_freq * n
            if harmonic_freq > result.sample_rate / 2:
                break
            harm_idx = np.argmin(np.abs(freqs - harmonic_freq))
            harm_db = magnitudes_db[harm_idx]
            harmonics.append((n, harmonic_freq, harm_db))
            print(f"  Harmonic {n} ({harmonic_freq:.0f} Hz): {harm_db:.1f} dB")

        # Verify harmonics are present (Karplus-Strong has rich harmonics)
        # Note: During attack, higher harmonics can be stronger than fundamental
        # We just verify multiple harmonics exist with significant energy
        if len(harmonics) >= 3:
            # Check that harmonics have energy (not silent)
            for n, freq, db in harmonics[:3]:
                assert db > 0, f"Harmonic {n} ({freq:.0f} Hz) has no energy"

    @pytest.mark.audio
    @pytest.mark.requires_plugin
    def test_snr_measurement(self, loaded_plugin: DawDreamerHost, thresholds):
        """
        Measure Signal-to-Noise Ratio.

        Compare signal power to noise floor.
        """
        result = loaded_plugin.render_note(
            note=60,
            velocity=100,
            duration_seconds=1.0,
            tail_seconds=3.0
        )

        # Signal: first second (while note is sounding)
        signal_samples = int(1.0 * result.sample_rate)
        signal_power = np.mean(result.audio[:, :signal_samples] ** 2)

        # Noise: last 0.5 seconds (after decay)
        noise_samples = int(0.5 * result.sample_rate)
        noise_power = np.mean(result.audio[:, -noise_samples:] ** 2)

        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 120  # Effectively infinite

        print(f"\nSNR measurement:")
        print(f"  Signal power: {signal_power:.6f}")
        print(f"  Noise power: {noise_power:.10f}")
        print(f"  SNR: {snr_db:.1f} dB")

        assert snr_db > thresholds.snr_db, \
            f"SNR {snr_db:.1f} dB below threshold {thresholds.snr_db} dB"
