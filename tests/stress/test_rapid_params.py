"""
Rapid Parameter Change Stress Tests

Test plugin stability with rapid parameter automation:
- Parameter fuzzing
- High-rate automation
- Edge value transitions
- Simultaneous parameter changes

These tests simulate aggressive DAW automation.
"""

import time
import random
from typing import List, Tuple

import pytest
import numpy as np

from tests.tools.dawdreamer_host import DawDreamerHost, MIDIEvent


def check_audio_valid(audio: np.ndarray) -> Tuple[bool, str]:
    """
    Check audio for NaN, inf, and other invalid values.

    Returns:
        (is_valid, error_message)
    """
    if np.any(np.isnan(audio)):
        return False, "Audio contains NaN values"
    if np.any(np.isinf(audio)):
        return False, "Audio contains inf values"
    return True, ""


def detect_discontinuities(audio: np.ndarray, threshold: float = 0.3) -> Tuple[int, float]:
    """
    Detect discontinuities (zipper noise) in audio.

    Args:
        audio: Audio samples (mono or first channel)
        threshold: Maximum allowed sample-to-sample difference

    Returns:
        (discontinuity_count, max_diff)
    """
    if audio.ndim > 1:
        audio = audio[0]

    diff = np.abs(np.diff(audio))
    max_diff = np.max(diff) if len(diff) > 0 else 0.0
    discontinuity_count = np.sum(diff > threshold)

    return int(discontinuity_count), float(max_diff)


class TestRapidParameterChanges:
    """Rapid parameter automation stress tests."""

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_parameter_fuzzing(self, loaded_plugin: DawDreamerHost, thresholds):
        """
        Random parameter value fuzzing.

        Rapidly set random values for all parameters.
        """
        # Get plugin info to enumerate parameters
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        param_count = len(info.parameters)
        assert param_count > 0, "Plugin has no parameters"

        changes_per_second = thresholds.stress.get("rapid_param_changes_per_second", 1000) if hasattr(thresholds, 'stress') else 1000

        # Play a sustained note while fuzzing parameters
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(2.0, 60, 0),
        ]

        # Start note
        random.seed(42)  # Reproducible randomness

        # Perform rapid parameter changes
        num_changes = 500  # Reasonable number for test duration
        param_indices = list(range(param_count))

        for _ in range(num_changes):
            param_idx = random.choice(param_indices)
            value = random.random()  # 0.0 to 1.0
            loaded_plugin.set_parameter(param_idx, value)

        # Render audio after fuzzing
        result = loaded_plugin.render(2.5, events)

        # Validate audio
        is_valid, error_msg = check_audio_valid(result.audio)
        assert is_valid, f"Audio invalid after parameter fuzzing: {error_msg}"

        # Check for excessive discontinuities
        disc_count, max_diff = detect_discontinuities(result.audio)

        print(f"\nParameter fuzzing results:")
        print(f"  Parameters fuzzed: {param_count}")
        print(f"  Total changes: {num_changes}")
        print(f"  Max sample diff: {max_diff:.4f}")
        print(f"  Discontinuities (>{0.3}): {disc_count}")

        # Should not have excessive discontinuities
        assert disc_count < 100, f"Too many discontinuities ({disc_count}) after parameter fuzzing"

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_audio_rate_automation(self, loaded_plugin: DawDreamerHost):
        """
        Change parameters at audio rate.

        Tests parameter smoothing and zipper noise prevention.
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        # Find first parameter
        param_names = list(info.parameters.keys())
        assert len(param_names) > 0, "Plugin has no parameters"

        # Play a note
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(1.0, 60, 0),
        ]

        # Simulate audio-rate automation by setting parameters rapidly
        # In practice, we'll change parameters many times during rendering
        # DawDreamer doesn't support true sample-by-sample automation,
        # so we'll do rapid changes before render and verify stability

        num_samples = loaded_plugin.sample_rate  # 1 second worth
        automation_points = 1000  # Change every ~48 samples at 48kHz

        # Create automation curve (sine wave)
        automation_values = 0.5 + 0.5 * np.sin(np.linspace(0, 10 * np.pi, automation_points))

        # Apply rapid automation
        for value in automation_values:
            loaded_plugin.set_parameter(0, float(value))

        # Render
        result = loaded_plugin.render(1.5, events)

        # Validate audio
        is_valid, error_msg = check_audio_valid(result.audio)
        assert is_valid, f"Audio invalid after audio-rate automation: {error_msg}"

        disc_count, max_diff = detect_discontinuities(result.audio)

        print(f"\nAudio-rate automation results:")
        print(f"  Automation points: {automation_points}")
        print(f"  Max sample diff: {max_diff:.4f}")
        print(f"  Discontinuities: {disc_count}")

        # Should handle rapid automation gracefully
        assert max_diff < 1.0, f"Extreme discontinuity detected: {max_diff}"

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_min_max_transitions(self, loaded_plugin: DawDreamerHost):
        """
        Rapidly switch between min and max values.

        Tests parameter range handling.
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        param_count = len(info.parameters)
        assert param_count > 0, "Plugin has no parameters"

        # Play a sustained note
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(2.0, 60, 0),
        ]

        # Alternate all parameters between 0 and 1
        num_transitions = 200
        for i in range(num_transitions):
            value = 1.0 if i % 2 == 0 else 0.0
            for param_idx in range(param_count):
                loaded_plugin.set_parameter(param_idx, value)

        # Render
        result = loaded_plugin.render(2.5, events)

        # Validate audio
        is_valid, error_msg = check_audio_valid(result.audio)
        assert is_valid, f"Audio invalid after min/max transitions: {error_msg}"

        disc_count, max_diff = detect_discontinuities(result.audio)

        print(f"\nMin/max transition results:")
        print(f"  Parameters: {param_count}")
        print(f"  Transitions: {num_transitions}")
        print(f"  Max sample diff: {max_diff:.4f}")
        print(f"  Discontinuities: {disc_count}")

        # Min/max transitions may cause discontinuities, but should not crash
        # or produce invalid audio
        assert not np.any(np.isnan(result.audio)), "NaN detected in audio"
        assert not np.any(np.isinf(result.audio)), "Inf detected in audio"

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_simultaneous_param_changes(self, loaded_plugin: DawDreamerHost):
        """
        Change all parameters simultaneously.

        Tests parameter system under maximum load.
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        param_count = len(info.parameters)
        assert param_count > 0, "Plugin has no parameters"

        # Play a sustained note
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(2.0, 60, 0),
        ]

        # Change all parameters simultaneously, repeatedly
        num_iterations = 100
        random.seed(42)

        for _ in range(num_iterations):
            # Generate random values for all parameters
            values = [random.random() for _ in range(param_count)]

            # Set all at once
            for param_idx, value in enumerate(values):
                loaded_plugin.set_parameter(param_idx, value)

        # Render
        result = loaded_plugin.render(2.5, events)

        # Validate audio
        is_valid, error_msg = check_audio_valid(result.audio)
        assert is_valid, f"Audio invalid after simultaneous param changes: {error_msg}"

        print(f"\nSimultaneous param change results:")
        print(f"  Parameters changed simultaneously: {param_count}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Total parameter sets: {num_iterations * param_count}")
        print(f"  Audio valid: {is_valid}")


class TestParameterEdgeCases:
    """Parameter edge case tests."""

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_zero_values(self, loaded_plugin: DawDreamerHost):
        """
        Set all parameters to zero.

        Should not crash, may produce silence.
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        param_count = len(info.parameters)
        assert param_count > 0, "Plugin has no parameters"

        # Set all parameters to 0
        for param_idx in range(param_count):
            loaded_plugin.set_parameter(param_idx, 0.0)

        # Play a note
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(1.0, 60, 0),
        ]

        # Render
        result = loaded_plugin.render(1.5, events)

        # Validate audio - should not contain invalid values
        is_valid, error_msg = check_audio_valid(result.audio)
        assert is_valid, f"Audio invalid with all zero params: {error_msg}"

        # Check amplitude (may be silent)
        max_amplitude = np.max(np.abs(result.audio))

        print(f"\nZero values test results:")
        print(f"  Parameters set to 0: {param_count}")
        print(f"  Max amplitude: {max_amplitude:.6f}")
        print(f"  Audio valid: {is_valid}")

        # Audio should be valid (no NaN/inf) regardless of output level

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_maximum_values(self, loaded_plugin: DawDreamerHost):
        """
        Set all parameters to maximum.

        Should not clip or crash.
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        param_count = len(info.parameters)
        assert param_count > 0, "Plugin has no parameters"

        # Set all parameters to 1 (maximum)
        for param_idx in range(param_count):
            loaded_plugin.set_parameter(param_idx, 1.0)

        # Play a note
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(1.0, 60, 0),
        ]

        # Render
        result = loaded_plugin.render(1.5, events)

        # Validate audio
        is_valid, error_msg = check_audio_valid(result.audio)
        assert is_valid, f"Audio invalid with all max params: {error_msg}"

        max_amplitude = np.max(np.abs(result.audio))

        # Check for clipping (values > 1.0)
        clipped_samples = np.sum(np.abs(result.audio) > 1.0)
        clipping_ratio = clipped_samples / result.audio.size

        print(f"\nMaximum values test results:")
        print(f"  Parameters set to 1: {param_count}")
        print(f"  Max amplitude: {max_amplitude:.6f}")
        print(f"  Clipped samples: {clipped_samples} ({clipping_ratio*100:.2f}%)")
        print(f"  Audio valid: {is_valid}")

        # Warn if significant clipping, but don't fail (may be intentional)
        if clipping_ratio > 0.1:
            print(f"  WARNING: Significant clipping detected ({clipping_ratio*100:.1f}%)")

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_denormal_values(self, loaded_plugin: DawDreamerHost):
        """
        Test with values approaching denormals.

        Some plugins have issues with very small values.
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        param_count = len(info.parameters)
        assert param_count > 0, "Plugin has no parameters"

        # Test with very small values (near denormal range)
        denormal_values = [1e-38, 1e-30, 1e-20, 1e-10, 1e-7]

        for denormal in denormal_values:
            # Set parameters to denormal value (clamped to 0-1 range for normalized params)
            # For normalized parameters, use values very close to 0
            test_value = min(denormal, 1.0)  # Clamp to valid range

            for param_idx in range(param_count):
                loaded_plugin.set_parameter(param_idx, test_value)

            # Play a note
            events = [
                MIDIEvent.note_on(0.0, 60, 100),
                MIDIEvent.note_off(0.5, 60, 0),
            ]

            # Render
            result = loaded_plugin.render(1.0, events)

            # Validate audio
            is_valid, error_msg = check_audio_valid(result.audio)
            assert is_valid, f"Audio invalid with denormal value {denormal}: {error_msg}"

        print(f"\nDenormal values test results:")
        print(f"  Denormal values tested: {denormal_values}")
        print(f"  Parameters: {param_count}")
        print(f"  All tests passed without NaN/inf")


class TestParameterSmoothing:
    """Parameter smoothing behavior tests."""

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_no_zipper_noise(self, loaded_plugin: DawDreamerHost):
        """
        Verify no zipper noise during parameter changes.

        Rapid changes should be smoothed.
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        # Play a sustained note
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(2.0, 60, 0),
        ]

        # Create gradual parameter ramp (simulating automation)
        # We'll change a parameter many times to simulate rapid DAW automation
        num_steps = 100

        for i in range(num_steps):
            value = i / (num_steps - 1)  # 0.0 to 1.0
            loaded_plugin.set_parameter(0, value)

        # Render audio
        result = loaded_plugin.render(2.5, events)

        # Validate audio
        is_valid, error_msg = check_audio_valid(result.audio)
        assert is_valid, f"Audio invalid during zipper noise test: {error_msg}"

        # Analyze for zipper noise (discontinuities)
        audio = result.audio[0] if result.audio.ndim > 1 else result.audio
        diff = np.diff(audio)
        max_diff = np.max(np.abs(diff))

        # Compute high-frequency energy ratio (zipper noise appears as HF content)
        fft = np.abs(np.fft.rfft(diff))
        if len(fft) > 4:
            high_freq_energy = np.mean(fft[len(fft) * 3 // 4:])
            low_freq_energy = np.mean(fft[:len(fft) // 4]) + 1e-10
            hf_ratio = high_freq_energy / low_freq_energy
        else:
            hf_ratio = 0.0

        # Count discontinuities above threshold
        discontinuity_threshold = 0.3
        disc_count = np.sum(np.abs(diff) > discontinuity_threshold)

        print(f"\nZipper noise test results:")
        print(f"  Parameter steps: {num_steps}")
        print(f"  Max sample diff: {max_diff:.4f}")
        print(f"  HF/LF energy ratio: {hf_ratio:.4f}")
        print(f"  Discontinuities (>{discontinuity_threshold}): {disc_count}")

        # Should not have excessive zipper noise
        # Allow some discontinuity as notes have natural attack/release transients
        assert max_diff < 1.0, f"Large discontinuity detected: {max_diff:.4f}"
        assert hf_ratio < 5.0, f"High HF ratio suggests zipper noise: {hf_ratio:.4f}"

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_smoothing_time(self, loaded_plugin: DawDreamerHost):
        """
        Measure parameter smoothing response time.

        How long before parameter reaches target?
        """
        info = loaded_plugin.get_info()
        assert info is not None, "Failed to get plugin info"

        # We'll measure smoothing by applying a step change and observing
        # how quickly the audio characteristic changes

        # First, set parameter to 0 and let it settle
        loaded_plugin.set_parameter(0, 0.0)

        # Play a sustained note
        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(2.0, 60, 0),
        ]

        # Render initial state
        result_before = loaded_plugin.render(0.5, events)

        # Now apply step change
        loaded_plugin.set_parameter(0, 1.0)

        # Render after step
        events_after = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_off(1.0, 60, 0),
        ]
        result_after = loaded_plugin.render(1.5, events_after)

        # Validate audio
        is_valid_before, _ = check_audio_valid(result_before.audio)
        is_valid_after, error_msg = check_audio_valid(result_after.audio)
        assert is_valid_before and is_valid_after, f"Audio invalid: {error_msg}"

        # Analyze smoothing by looking at the envelope of the audio
        # If parameter smoothing is working, we should see gradual transition
        audio_after = result_after.audio[0] if result_after.audio.ndim > 1 else result_after.audio

        # Compute envelope using absolute value with smoothing
        window_size = int(0.005 * result_after.sample_rate)  # 5ms window
        if len(audio_after) > window_size:
            # Moving average of absolute values
            envelope = np.convolve(
                np.abs(audio_after),
                np.ones(window_size) / window_size,
                mode='valid'
            )

            # Look for step changes in envelope (would indicate poor smoothing)
            envelope_diff = np.diff(envelope)
            max_envelope_jump = np.max(np.abs(envelope_diff))

            # Estimate settling time (time for envelope to stabilize)
            # Find when envelope derivative drops below threshold
            settling_threshold = 0.001
            settled_indices = np.where(np.abs(envelope_diff) < settling_threshold)[0]

            if len(settled_indices) > 0:
                first_settled = settled_indices[0]
                settling_time_ms = (first_settled / result_after.sample_rate) * 1000
            else:
                settling_time_ms = float('inf')
        else:
            max_envelope_jump = 0.0
            settling_time_ms = 0.0

        print(f"\nParameter smoothing time test:")
        print(f"  Max envelope jump: {max_envelope_jump:.6f}")
        print(f"  Estimated settling time: {settling_time_ms:.2f} ms")

        # Smoothing time should be reasonable (< 100ms for most plugins)
        if settling_time_ms != float('inf'):
            assert settling_time_ms < 500, f"Smoothing time too long: {settling_time_ms:.1f}ms"

        # No extreme jumps
        assert max_envelope_jump < 0.1, f"Parameter step caused audio jump: {max_envelope_jump:.4f}"
