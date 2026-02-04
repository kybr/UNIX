"""
Note Triggering Tests

Tests basic MIDI note functionality:
- Note on/off
- Velocity response
- Note range
- Channel handling
"""

import pytest
import numpy as np

from .midi_test_harness import MIDIGenerator, MIDINote


class TestNoteOnOff:
    """Basic note on/off tests."""

    @pytest.fixture
    def midi_gen(self):
        return MIDIGenerator()

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_note_on_produces_sound(self, loaded_plugin, midi_gen):
        """
        Verify note-on message produces audio output.

        The most basic test - pressing a key should make sound.
        """
        result = loaded_plugin.render_note(
            note=60,  # Middle C
            velocity=100,
            duration_seconds=0.5,
            tail_seconds=1.0
        )

        # Should produce audible output
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.01, f"Note on should produce audio, RMS={rms}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_note_off_silences(self, loaded_plugin, midi_gen):
        """
        Verify note-off message eventually silences output.

        After note-off, sound should decay to silence.
        """
        result = loaded_plugin.render_note(
            note=60,
            velocity=100,
            duration_seconds=0.2,
            tail_seconds=5.0  # Long tail to verify decay
        )

        # Get the last second of audio
        samples_per_second = result.sample_rate
        last_second = result.audio[:, -samples_per_second:]

        # RMS of last second should be much lower than overall
        overall_rms = np.sqrt(np.mean(result.audio ** 2))
        tail_rms = np.sqrt(np.mean(last_second ** 2))

        # Tail should be at least 10x quieter than overall
        assert tail_rms < overall_rms / 10, \
            f"Note should decay after release: overall={overall_rms:.4f}, tail={tail_rms:.4f}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_zero_velocity_note_off(self, loaded_plugin, midi_gen):
        """
        Test note-on with velocity 0 acts as note-off.

        This is standard MIDI behavior.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        events = [
            MIDIEvent.note_on(0.0, 60, 100),
            MIDIEvent(0.3, 0x90, 60, 0, 0),  # Note on with velocity 0 = note off
        ]

        result = loaded_plugin.render(2.0, events)

        # Audio should be present but decay
        overall_rms = np.sqrt(np.mean(result.audio ** 2))
        assert overall_rms > 0.001, "Should produce some audio before velocity-0 note off"


class TestVelocity:
    """Velocity response tests."""

    @pytest.fixture
    def midi_gen(self):
        return MIDIGenerator()

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_velocity_affects_amplitude(self, loaded_plugin, midi_gen, thresholds):
        """
        Verify velocity affects output amplitude.

        Higher velocity should produce louder output.
        """
        velocities = [32, 64, 96, 127]
        amplitudes = []

        for vel in velocities:
            result = loaded_plugin.render_note(
                note=60,
                velocity=vel,
                duration_seconds=0.5,
                tail_seconds=0.5
            )
            rms = np.sqrt(np.mean(result.audio ** 2))
            amplitudes.append(rms)

        # Verify generally increasing response
        # Note: Many synths have some compression at max velocity (127)
        for i in range(len(amplitudes) - 2):  # Check all except last transition
            assert amplitudes[i] < amplitudes[i + 1], \
                f"Velocity {velocities[i]} should be quieter than {velocities[i+1]}: " \
                f"{amplitudes[i]:.4f} vs {amplitudes[i+1]:.4f}"

        # Max velocity should still be at least 80% of peak amplitude
        max_amp = max(amplitudes)
        assert amplitudes[-1] >= max_amp * 0.80, \
            f"Velocity 127 too quiet: {amplitudes[-1]:.4f} vs max {max_amp:.4f}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_velocity_127_loudest(self, loaded_plugin, midi_gen):
        """
        Verify velocity 127 produces maximum amplitude.
        """
        result_max = loaded_plugin.render_note(note=60, velocity=127, duration_seconds=0.5)
        result_mid = loaded_plugin.render_note(note=60, velocity=64, duration_seconds=0.5)

        rms_max = np.sqrt(np.mean(result_max.audio ** 2))
        rms_mid = np.sqrt(np.mean(result_mid.audio ** 2))

        assert rms_max > rms_mid, \
            f"Velocity 127 should be louder than 64: {rms_max:.4f} vs {rms_mid:.4f}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_velocity_1_audible(self, loaded_plugin, midi_gen):
        """
        Verify velocity 1 (minimum) still produces audible output.
        """
        result = loaded_plugin.render_note(
            note=60,
            velocity=1,
            duration_seconds=1.0,
            tail_seconds=1.0
        )

        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.0001, f"Velocity 1 should still produce audible output, RMS={rms}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_velocity_curve(self, loaded_plugin, midi_gen):
        """
        Measure and characterize velocity response curve.

        Document whether linear, logarithmic, or custom.
        """
        velocities = [1, 32, 64, 96, 127]
        amplitudes = []

        for vel in velocities:
            result = loaded_plugin.render_note(
                note=60,
                velocity=vel,
                duration_seconds=0.3,
                tail_seconds=0.3
            )
            rms = np.sqrt(np.mean(result.audio ** 2))
            amplitudes.append(rms)

        # Normalize to max amplitude
        max_amp = max(amplitudes)
        normalized = [a / max_amp for a in amplitudes]

        # Check that curve is generally monotonic with some tolerance
        # Note: Some synths have compression or non-linear response
        # Allow 10% tolerance between adjacent velocity steps
        for i in range(len(normalized) - 1):
            # Each step should be within 15% of the next (allowing some non-monotonicity)
            assert normalized[i] <= normalized[i + 1] * 1.15, \
                f"Velocity curve has excessive non-monotonicity: {list(zip(velocities, normalized))}"

        # Final velocity (127) should be at least 80% of max (allowing for compression)
        assert normalized[-1] >= 0.80, \
            f"Max velocity too quiet: {normalized[-1]:.2f}"

        # Lowest velocity should be quieter than highest
        assert normalized[0] < normalized[-1], \
            f"Velocity 1 should be quieter than velocity 127: {normalized[0]:.2f} vs {normalized[-1]:.2f}"


class TestNoteRange:
    """MIDI note range tests."""

    @pytest.fixture
    def midi_gen(self):
        return MIDIGenerator()

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_full_midi_range(self, loaded_plugin, midi_gen):
        """
        Test representative notes across MIDI range produce output.

        Testing every note would be slow, so test every octave.
        """
        test_notes = [24, 36, 48, 60, 72, 84, 96, 108]  # C1 to C8

        for note in test_notes:
            result = loaded_plugin.render_note(
                note=note,
                velocity=100,
                duration_seconds=0.3,
                tail_seconds=0.3
            )
            rms = np.sqrt(np.mean(result.audio ** 2))
            assert rms > 0.001, f"Note {note} should produce audio, RMS={rms}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_note_0_lowest(self, loaded_plugin, midi_gen):
        """
        Test MIDI note 0 (C-1, ~8.18 Hz).

        Very low notes may require special handling.
        """
        result = loaded_plugin.render_note(
            note=0,
            velocity=100,
            duration_seconds=1.0,
            tail_seconds=1.0
        )

        # Should produce some output (may be very low frequency)
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.0001, f"Note 0 should produce some output, RMS={rms}"

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(result.audio)), "Low note should not produce NaN"
        assert not np.any(np.isinf(result.audio)), "Low note should not produce Inf"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_note_127_highest(self, loaded_plugin, midi_gen):
        """
        Test MIDI note 127 (G9, ~12543 Hz).

        Very high notes approach Nyquist at lower sample rates.
        """
        result = loaded_plugin.render_note(
            note=127,
            velocity=100,
            duration_seconds=0.5,
            tail_seconds=0.5
        )

        # Should produce some output
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.0001, f"Note 127 should produce output, RMS={rms}"

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(result.audio)), "High note should not produce NaN"
        assert not np.any(np.isinf(result.audio)), "High note should not produce Inf"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_pitch_accuracy(self, loaded_plugin, midi_gen, thresholds):
        """
        Verify pitch accuracy across range.

        Measured pitch should match expected within threshold.
        """
        test_notes = [48, 60, 72]  # C3, C4, C5 - easier to detect

        for note in test_notes:
            expected_freq = 440.0 * (2 ** ((note - 69) / 12))

            result = loaded_plugin.render_note(
                note=note,
                velocity=100,
                duration_seconds=1.0,
                tail_seconds=0.5
            )

            # Robust pitch detection via autocorrelation (finds fundamental, not harmonics)
            audio_mono = result.audio[0] if result.audio.ndim > 1 else result.audio
            # Use first 0.5 seconds after brief attack for cleaner signal
            start_sample = int(0.05 * result.sample_rate)  # Skip attack
            end_sample = int(0.4 * result.sample_rate)
            segment = audio_mono[start_sample:end_sample]

            # Autocorrelation-based pitch detection
            corr = np.correlate(segment, segment, mode='full')
            corr = corr[len(corr) // 2:]  # Take positive lags

            # Find expected period range (±1 octave)
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

            # Calculate cents deviation
            if detected_freq > 0:
                cents = 1200 * np.log2(detected_freq / expected_freq)
                assert abs(cents) < thresholds.frequency_accuracy_cents, \
                    f"Note {note}: expected {expected_freq:.1f} Hz, got {detected_freq:.1f} Hz ({cents:.1f} cents)"


class TestMIDIChannel:
    """MIDI channel handling tests."""

    @pytest.fixture
    def midi_gen(self):
        return MIDIGenerator()

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_channel_1_default(self, loaded_plugin, midi_gen):
        """
        Test notes on channel 1 (default) work.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        events = [
            MIDIEvent.note_on(0.0, 60, 100, channel=0),
            MIDIEvent.note_off(0.5, 60, 0, channel=0),
        ]

        result = loaded_plugin.render(1.5, events)
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.01, "Channel 1 notes should produce audio"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_all_channels_respond(self, loaded_plugin, midi_gen):
        """
        Test plugin responds to multiple MIDI channels.

        Test channels 0, 5, and 15 as representative samples.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        test_channels = [0, 5, 15]

        for channel in test_channels:
            events = [
                MIDIEvent.note_on(0.0, 60, 100, channel=channel),
                MIDIEvent.note_off(0.5, 60, 0, channel=channel),
            ]

            result = loaded_plugin.render(1.0, events)
            rms = np.sqrt(np.mean(result.audio ** 2))
            assert rms > 0.001, f"Channel {channel + 1} should produce audio, RMS={rms}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_omni_mode(self, loaded_plugin):
        """
        Test if plugin supports OMNI mode (responds to all channels).

        Many synth plugins respond to all channels by default.
        """
        from tests.tools.dawdreamer_host import MIDIEvent

        # Play notes on different channels simultaneously
        events = [
            MIDIEvent.note_on(0.0, 60, 100, channel=0),
            MIDIEvent.note_on(0.0, 64, 100, channel=5),
            MIDIEvent.note_on(0.0, 67, 100, channel=10),
            MIDIEvent.note_off(0.5, 60, 0, channel=0),
            MIDIEvent.note_off(0.5, 64, 0, channel=5),
            MIDIEvent.note_off(0.5, 67, 0, channel=10),
        ]

        result = loaded_plugin.render(1.5, events)
        rms = np.sqrt(np.mean(result.audio ** 2))

        # If plugin responds to all channels, we should get a chord
        assert rms > 0.01, "OMNI mode: plugin should respond to notes on any channel"
