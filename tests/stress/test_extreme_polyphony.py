"""
Extreme Polyphony Stress Tests

Push the plugin to its limits with maximum voice counts:
- 64 simultaneous voices
- 128 simultaneous voices
- Rapid voice triggering
- Voice count ramping

These tests verify stability under extreme load.
"""

import time
import subprocess
from typing import List

import pytest
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TestExtremePolyphony:
    """Extreme polyphony stress tests."""

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.requires_plugin
    def test_64_voice_stress(self, loaded_plugin, thresholds):
        """
        Stress test with 64 simultaneous voices.

        Verifies:
        - No crash
        - Audio continues
        - CPU stays within bounds
        """
        num_voices = 64

        result = loaded_plugin.render_polyphony_test(
            num_voices=num_voices,
            base_note=24,
            velocity=80,
            stagger_ms=5.0,
            hold_seconds=2.0
        )

        # Should not crash and should produce audio
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.0001, f"64 voices should produce audio, RMS={rms}"

        # No NaN or Inf values
        assert not np.any(np.isnan(result.audio)), "Audio contains NaN values"
        assert not np.any(np.isinf(result.audio)), "Audio contains Inf values"

        # Check realtime capability
        assert result.realtime_ratio < 1.0, \
            f"64 voices should be realtime capable: {result.realtime_ratio:.2f}x realtime"

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.requires_plugin
    def test_128_voice_stress(self, loaded_plugin, thresholds):
        """
        Stress test with 128 simultaneous voices (max MIDI).

        This is the maximum possible polyphony with standard MIDI.
        """
        num_voices = 128

        result = loaded_plugin.render_polyphony_test(
            num_voices=num_voices,
            base_note=12,  # Start very low to spread across range
            velocity=64,
            stagger_ms=3.0,
            hold_seconds=2.0
        )

        # Should not crash
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.0, f"128 voices should produce some audio, RMS={rms}"

        # No NaN or Inf values
        assert not np.any(np.isnan(result.audio)), "Audio contains NaN values"
        assert not np.any(np.isinf(result.audio)), "Audio contains Inf values"

        # May not be realtime at 128 voices, but should complete
        assert result.realtime_ratio < 5.0, \
            f"128 voices too slow: {result.realtime_ratio:.2f}x realtime"

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.requires_plugin
    def test_polyphony_ramp(self, loaded_plugin):
        """
        Gradually increase polyphony until failure.

        Find the breaking point and verify graceful degradation.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        voice_counts = [8, 16, 32, 64, 96, 128]
        results = []

        for num_voices in voice_counts:
            result = loaded_plugin.render_polyphony_test(
                num_voices=num_voices,
                base_note=36,
                velocity=80,
                stagger_ms=5.0,
                hold_seconds=1.0
            )

            rms = np.sqrt(np.mean(result.audio ** 2))
            has_nan = np.any(np.isnan(result.audio))

            results.append({
                'voices': num_voices,
                'rms': rms,
                'realtime_ratio': result.realtime_ratio,
                'has_nan': has_nan
            })

            # Stop if we detect problems
            if has_nan:
                break

        # All tested levels should be stable
        for r in results:
            assert not r['has_nan'], f"NaN detected at {r['voices']} voices"
            assert r['rms'] > 0, f"No audio at {r['voices']} voices"

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.requires_plugin
    def test_rapid_voice_cycling(self, loaded_plugin):
        """
        Rapidly trigger and release voices.

        Tests voice allocation/deallocation under stress.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        events = []
        num_cycles = 50
        notes_per_cycle = 16

        for cycle in range(num_cycles):
            cycle_start = cycle * 0.1  # 100ms per cycle

            # Trigger notes
            for i in range(notes_per_cycle):
                note = 36 + i
                events.append(MIDIEvent.note_on(cycle_start, note, 100))

            # Release all notes
            for i in range(notes_per_cycle):
                note = 36 + i
                events.append(MIDIEvent.note_off(cycle_start + 0.08, note))

        result = loaded_plugin.render(6.0, events)

        # Should not crash
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Rapid voice cycling should produce audio"

        # No audio artifacts
        assert not np.any(np.isnan(result.audio)), "Audio contains NaN values"
        max_amplitude = np.max(np.abs(result.audio))
        assert max_amplitude < 3.0, f"Audio clipped: max={max_amplitude}"


class TestPolyphonyEdgeCases:
    """Edge cases in polyphony handling."""

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_all_same_note(self, loaded_plugin):
        """
        Trigger same note many times simultaneously.

        Some synths handle this specially.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        # Trigger middle C 32 times rapidly
        events = []
        for i in range(32):
            events.append(MIDIEvent.note_on(i * 0.01, 60, 100))

        # Release all
        events.append(MIDIEvent.note_off(1.5, 60))

        result = loaded_plugin.render(3.0, events)

        # Should not crash and should produce audio
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Same note polyphony should produce audio"

        # No NaN values
        assert not np.any(np.isnan(result.audio)), "Audio contains NaN values"

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_chromatic_cluster(self, loaded_plugin):
        """
        Trigger all 12 chromatic notes simultaneously.

        Dense harmonic content.
        """
        # C major chromatic cluster around middle C
        notes = list(range(60, 72))  # 12 notes

        result = loaded_plugin.render_chord(
            notes=notes,
            velocity=80,
            duration_seconds=1.0,
            tail_seconds=1.0
        )

        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.01, "Chromatic cluster should produce audio"

        # Check for reasonable max amplitude
        # With 12 simultaneous voices, peaks can exceed 2.0 without clipping issues
        max_amplitude = np.max(np.abs(result.audio))
        assert max_amplitude < 5.0, f"Chromatic cluster excessively loud: max={max_amplitude}"

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_full_range_chord(self, loaded_plugin):
        """
        Trigger notes across entire MIDI range simultaneously.

        From note 0 to 127.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        # Notes spread across full range (every 8 semitones)
        notes = list(range(12, 120, 8))  # ~14 notes spread across range

        events = []
        for note in notes:
            events.append(MIDIEvent.note_on(0.0, note, 80))
        for note in notes:
            events.append(MIDIEvent.note_off(1.0, note))

        result = loaded_plugin.render(3.0, events)

        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Full range chord should produce audio"

        # No NaN values
        assert not np.any(np.isnan(result.audio)), "Audio contains NaN values"


class TestPolyphonyRecovery:
    """Recovery after polyphony stress."""

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.requires_plugin
    def test_recovery_after_overload(self, loaded_plugin):
        """
        Test plugin recovers after polyphony overload.

        After releasing all voices, should return to normal.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        # First, overload with many voices
        overload_result = loaded_plugin.render_polyphony_test(
            num_voices=64,
            base_note=36,
            velocity=100,
            stagger_ms=5.0,
            hold_seconds=1.0
        )

        # Then play a single note to verify recovery
        recovery_result = loaded_plugin.render_note(
            note=60,
            velocity=100,
            duration_seconds=1.0,
            tail_seconds=1.0
        )

        # Single note should still work correctly
        rms = np.sqrt(np.mean(recovery_result.audio ** 2))
        assert rms > 0.01, "Plugin should recover and play single note after overload"

        # CPU should return to normal
        assert recovery_result.realtime_ratio < 0.5, \
            f"CPU should be low after recovery: {recovery_result.realtime_ratio:.2f}x realtime"

    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_voice_stealing_stability(self, loaded_plugin):
        """
        Verify voice stealing doesn't cause instability.

        Continuously exceed polyphony limit.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        events = []
        max_plugin_voices = 16  # Plugin's configured limit

        # Continuously trigger more notes than voices available
        for cycle in range(10):
            cycle_start = cycle * 0.3

            # Trigger 20 notes (more than 16 voice limit)
            for i in range(20):
                note = 36 + (i + cycle) % 48  # Vary notes each cycle
                events.append(MIDIEvent.note_on(cycle_start + i * 0.01, note, 100))

        # Release all at the end
        for note in range(36, 84):
            events.append(MIDIEvent.note_off(4.0, note))

        result = loaded_plugin.render(5.0, events)

        # Should handle voice stealing without crash
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Voice stealing should not cause silence"

        # No audio glitches
        assert not np.any(np.isnan(result.audio)), "Voice stealing caused NaN"
        max_amplitude = np.max(np.abs(result.audio))
        assert max_amplitude < 3.0, f"Voice stealing caused clipping: max={max_amplitude}"
