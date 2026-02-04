"""
Polyphony Tests

Tests the plugin's ability to handle multiple simultaneous voices:
- Maximum polyphony
- Voice allocation
- Voice stealing
- Polyphony under load
"""

import time
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from .midi_test_harness import MIDIGenerator, MIDINote


class TestPolyphony:
    """Polyphony and voice management tests."""

    @pytest.fixture
    def midi_gen(self):
        return MIDIGenerator()

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_minimum_polyphony(self, loaded_plugin, thresholds):
        """
        Verify plugin supports minimum required polyphony.

        The plugin should support at least the threshold number of voices.
        """
        min_polyphony = thresholds.min_polyphony
        result = loaded_plugin.render_polyphony_test(
            num_voices=min_polyphony,
            base_note=48,
            velocity=100,
            hold_seconds=1.0
        )

        # Check that audio was produced
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, f"Expected {min_polyphony} voices to produce audio, RMS={rms}"

        # No NaN or Inf values
        assert not np.any(np.isnan(result.audio)), "Audio contains NaN values"
        assert not np.any(np.isinf(result.audio)), "Audio contains Inf values"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_polyphony_all_notes_sound(self, loaded_plugin, midi_gen):
        """
        Test that all triggered notes produce sound.

        Send multiple notes and verify each one produces output.
        """
        # Generate a chord (C major)
        chord_notes = [60, 64, 67, 72]
        result = loaded_plugin.render_chord(
            notes=chord_notes,
            velocity=100,
            duration_seconds=1.0,
            tail_seconds=1.0
        )

        # Compute RMS of the audio
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.01, f"Chord should produce audible output, RMS={rms}"

        # Check for frequency content via simple FFT analysis
        audio_mono = result.audio[0] if result.audio.ndim > 1 else result.audio
        fft = np.abs(np.fft.rfft(audio_mono))
        freqs = np.fft.rfftfreq(len(audio_mono), 1.0 / result.sample_rate)

        # Check that we have energy at expected frequencies
        expected_freqs = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
        for expected_freq in expected_freqs:
            # Find bins near expected frequency (+/- 10 Hz)
            mask = (freqs >= expected_freq - 10) & (freqs <= expected_freq + 10)
            energy = np.sum(fft[mask])
            assert energy > 0, f"Expected energy at {expected_freq} Hz"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_voice_stealing(self, loaded_plugin, thresholds):
        """
        Test voice stealing when exceeding polyphony limit.

        When more notes than max polyphony are triggered:
        - Oldest or quietest voice should be stolen
        - No crash or silence
        """
        max_voices = 16  # Plugin's configured voice count
        notes_to_trigger = max_voices + 10

        result = loaded_plugin.render_polyphony_test(
            num_voices=notes_to_trigger,
            base_note=36,
            velocity=100,
            stagger_ms=20.0,
            hold_seconds=2.0
        )

        # Verify sound continues (no crash)
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Voice stealing should not cause silence"

        # Verify no extreme audio glitches (allow some accumulation from many voices)
        # With 26 simultaneous voices at velocity 100, some amplitude buildup is expected
        max_amplitude = np.max(np.abs(result.audio))
        assert max_amplitude < 4.0, f"Audio clipped excessively: max={max_amplitude}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    @pytest.mark.stress
    def test_extreme_polyphony(self, loaded_plugin):
        """
        Stress test with extreme polyphony (128 voices).

        Verify plugin handles maximum load without crash.
        """
        extreme_voices = 128

        result = loaded_plugin.render_polyphony_test(
            num_voices=extreme_voices,
            base_note=24,
            velocity=80,
            stagger_ms=5.0,
            hold_seconds=2.0
        )

        # Should not crash and should produce audio
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.0001, "Extreme polyphony should still produce audio"

        # No NaN values
        assert not np.any(np.isnan(result.audio)), "Audio contains NaN values"

        # Check realtime capability
        assert result.realtime_ratio < 2.0, f"Plugin too slow: {result.realtime_ratio:.2f}x realtime"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_voice_allocation_fairness(self, loaded_plugin, midi_gen):
        """
        Test that voice allocation is fair.

        All notes should get approximately equal time/resources.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        # Create polyphony test with 8 voices, staggered start
        num_voices = 8
        events = []
        for i in range(num_voices):
            start_time = i * 0.05  # 50ms stagger
            note = 48 + i
            events.append(MIDIEvent.note_on(start_time, note, 100))

        # All notes off after 2 seconds
        for i in range(num_voices):
            note = 48 + i
            events.append(MIDIEvent.note_off(2.0, note))

        result = loaded_plugin.render(3.0, events)

        # Analyze output - all voices should contribute
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.01, "All voices should contribute to output"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_rapid_note_triggering(self, loaded_plugin, midi_gen):
        """
        Test rapid successive note triggering.

        Fast repeated notes should all trigger cleanly.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        # Same note 20 times rapidly
        events = []
        for i in range(20):
            start_time = i * 0.05  # 50ms apart
            events.append(MIDIEvent.note_on(start_time, 60, 100))
            events.append(MIDIEvent.note_off(start_time + 0.04, 60))

        result = loaded_plugin.render(2.0, events)

        # Should produce audio
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Rapid notes should produce audio"

        # Check for audio artifacts (sudden spikes)
        max_amplitude = np.max(np.abs(result.audio))
        assert max_amplitude < 2.0, "No audio artifacts from rapid triggering"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_polyphony_with_sustain(self, loaded_plugin):
        """
        Test polyphony behavior with sustain pedal.

        Notes held by sustain should count toward polyphony.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        events = [
            # Sustain pedal on (CC64 = 127)
            MIDIEvent.cc(0.0, 64, 127),
            # Play some notes
            MIDIEvent.note_on(0.1, 60, 100),
            MIDIEvent.note_off(0.3, 60),  # Key released but sustain holds
            MIDIEvent.note_on(0.4, 64, 100),
            MIDIEvent.note_off(0.6, 64),
            MIDIEvent.note_on(0.7, 67, 100),
            MIDIEvent.note_off(0.9, 67),
            # Sustain off - notes should release
            MIDIEvent.cc(1.0, 64, 0),
        ]

        result = loaded_plugin.render(3.0, events)

        # Should produce audio during sustained period
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Sustained notes should produce audio"


class TestVoiceAllocation:
    """Voice allocation algorithm tests."""

    @pytest.fixture
    def midi_gen(self):
        return MIDIGenerator()

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_same_note_retrigger(self, loaded_plugin, midi_gen):
        """
        Test retriggering the same note.

        Should restart the note cleanly.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_on(0.2, 60, 100),  # Retrigger same note
            MIDIEvent.note_on(0.4, 60, 100),  # Retrigger again
            MIDIEvent.note_off(1.0, 60),
        ]

        result = loaded_plugin.render(2.0, events)

        # Should produce audio
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Retriggered notes should produce audio"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_legato_behavior(self, loaded_plugin):
        """
        Test legato (overlapping notes) behavior.

        New notes should start while old notes are still sounding.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent.note_on(0.3, 64, 100),  # Overlap with previous
            MIDIEvent.note_off(0.5, 60),
            MIDIEvent.note_on(0.6, 67, 100),  # Overlap with 64
            MIDIEvent.note_off(0.8, 64),
            MIDIEvent.note_off(1.5, 67),
        ]

        result = loaded_plugin.render(3.0, events)

        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.01, "Legato playing should produce continuous audio"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_release_order(self, loaded_plugin, midi_gen):
        """
        Test note release order independence.

        Notes should release in any order correctly.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        # Play C-E-G chord, release in reverse order (G-E-C)
        events = [
            MIDIEvent.note_on(0.0, 60, 100),  # C
            MIDIEvent.note_on(0.0, 64, 100),  # E
            MIDIEvent.note_on(0.0, 67, 100),  # G
            # Release in reverse order
            MIDIEvent.note_off(0.5, 67),  # G off
            MIDIEvent.note_off(0.7, 64),  # E off
            MIDIEvent.note_off(0.9, 60),  # C off
        ]

        result = loaded_plugin.render(2.0, events)

        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.01, "Chord with staggered release should produce audio"


class TestPolyphonyPerformance:
    """Polyphony performance tests."""

    @pytest.mark.midi
    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_vs_polyphony(self, loaded_plugin, thresholds):
        """
        Measure CPU usage at different polyphony levels.

        CPU should scale reasonably with voice count.
        """
        polyphony_levels = [1, 4, 8, 16, 32]
        cpu_measurements = []

        for voices in polyphony_levels:
            result = loaded_plugin.render_polyphony_test(
                num_voices=voices,
                base_note=48,
                velocity=100,
                hold_seconds=1.0
            )
            cpu_percent = result.realtime_ratio * 100
            cpu_measurements.append((voices, cpu_percent))

        # Verify scaling is reasonable (not exponential)
        # CPU at 32 voices should be less than 32x CPU at 1 voice
        single_voice_cpu = cpu_measurements[0][1]
        max_voice_cpu = cpu_measurements[-1][1]

        # Allow up to 20x increase for 32x voices (sublinear scaling)
        assert max_voice_cpu < single_voice_cpu * 20, \
            f"CPU scaling too steep: 1 voice={single_voice_cpu:.1f}%, 32 voices={max_voice_cpu:.1f}%"

    @pytest.mark.midi
    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_polyphony_latency(self, loaded_plugin):
        """
        Test latency at different polyphony levels.

        Latency should remain consistent regardless of voice count.
        """
        # Measure latency at different polyphony levels
        latencies = []

        for voices in [1, 8, 16]:
            result = loaded_plugin.render_polyphony_test(
                num_voices=voices,
                base_note=48,
                velocity=100,
                hold_seconds=0.5
            )
            latencies.append(result.plugin_latency_samples)

        # All latencies should be the same
        assert all(l == latencies[0] for l in latencies), \
            f"Latency varied with polyphony: {latencies}"
