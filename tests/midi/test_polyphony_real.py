"""
Real Polyphony Tests (using DawDreamer)

Actual tests that load the plugin and verify polyphony behavior.
"""

import numpy as np
import pytest

from tests.tools.dawdreamer_host import DawDreamerHost, MIDIEvent


class TestRealPolyphony:
    """Real polyphony tests using DawDreamer plugin host."""

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_single_note_produces_audio(self, loaded_plugin: DawDreamerHost):
        """
        Verify a single note produces audio output.

        The most basic test - play a note, hear sound.
        """
        result = loaded_plugin.render_note(
            note=60,  # Middle C
            velocity=100,
            duration_seconds=0.5,
            tail_seconds=1.0
        )

        # Should have stereo audio
        assert result.audio.shape[0] == 2, "Expected stereo output"

        # Should have non-zero output
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, f"Expected audible output, got RMS={rms}"

        print(f"\nSingle note test:")
        print(f"  RMS: {rms:.4f}")
        print(f"  Peak: {np.max(np.abs(result.audio)):.4f}")
        print(f"  CPU: {result.realtime_ratio * 100:.1f}% of realtime")

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_chord_produces_richer_sound(self, loaded_plugin: DawDreamerHost):
        """
        Verify a chord produces more energy than a single note.

        Multiple notes should sum to louder output.
        """
        # Single note
        single = loaded_plugin.render_note(note=60, velocity=100, duration_seconds=0.5)
        single_rms = np.sqrt(np.mean(single.audio ** 2))

        # Chord (C major)
        chord = loaded_plugin.render_chord(
            notes=[60, 64, 67],  # C, E, G
            velocity=100,
            duration_seconds=0.5
        )
        chord_rms = np.sqrt(np.mean(chord.audio ** 2))

        print(f"\nChord vs single note:")
        print(f"  Single RMS: {single_rms:.4f}")
        print(f"  Chord RMS: {chord_rms:.4f}")
        print(f"  Ratio: {chord_rms / single_rms:.2f}x")

        # Chord should be louder (roughly 1.5-3x for 3 notes)
        assert chord_rms > single_rms, "Chord should be louder than single note"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_polyphony_8_voices(self, loaded_plugin: DawDreamerHost):
        """
        Test 8 simultaneous voices.

        All voices should sound without issues.
        """
        result = loaded_plugin.render_polyphony_test(
            num_voices=8,
            base_note=48,
            velocity=100,
            stagger_ms=10,
            hold_seconds=1.0
        )

        rms = np.sqrt(np.mean(result.audio ** 2))

        print(f"\n8-voice polyphony:")
        print(f"  RMS: {rms:.4f}")
        print(f"  CPU: {result.realtime_ratio * 100:.1f}% of realtime")

        assert rms > 0.01, "8 voices should produce significant output"
        assert result.realtime_ratio < 0.5, "Should process faster than realtime"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_polyphony_16_voices(self, loaded_plugin: DawDreamerHost):
        """Test 16 simultaneous voices."""
        result = loaded_plugin.render_polyphony_test(
            num_voices=16,
            base_note=36,
            velocity=100,
            stagger_ms=5,
            hold_seconds=1.0
        )

        rms = np.sqrt(np.mean(result.audio ** 2))

        print(f"\n16-voice polyphony:")
        print(f"  RMS: {rms:.4f}")
        print(f"  CPU: {result.realtime_ratio * 100:.1f}% of realtime")

        assert rms > 0.01, "16 voices should produce output"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_polyphony_32_voices(self, loaded_plugin: DawDreamerHost):
        """Test 32 simultaneous voices."""
        result = loaded_plugin.render_polyphony_test(
            num_voices=32,
            base_note=36,
            velocity=80,
            stagger_ms=5,
            hold_seconds=1.0
        )

        rms = np.sqrt(np.mean(result.audio ** 2))

        print(f"\n32-voice polyphony:")
        print(f"  RMS: {rms:.4f}")
        print(f"  CPU: {result.realtime_ratio * 100:.1f}% of realtime")

        assert rms > 0.005, "32 voices should produce output"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    @pytest.mark.stress
    def test_polyphony_64_voices(self, loaded_plugin: DawDreamerHost):
        """Stress test with 64 voices."""
        result = loaded_plugin.render_polyphony_test(
            num_voices=64,
            base_note=24,
            velocity=64,
            stagger_ms=2,
            hold_seconds=1.0
        )

        rms = np.sqrt(np.mean(result.audio ** 2))

        print(f"\n64-voice polyphony (stress):")
        print(f"  RMS: {rms:.4f}")
        print(f"  CPU: {result.realtime_ratio * 100:.1f}% of realtime")

        # Should still produce output and not exceed realtime
        assert rms > 0.001, "64 voices should produce some output"
        assert result.realtime_ratio < 1.0, "Should still be realtime capable"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_velocity_affects_amplitude(self, loaded_plugin: DawDreamerHost):
        """
        Verify velocity affects output amplitude.

        Higher velocity = louder.
        """
        results = []

        for velocity in [20, 64, 100, 127]:
            result = loaded_plugin.render_note(
                note=60,
                velocity=velocity,
                duration_seconds=0.3,
                tail_seconds=0.5
            )
            rms = np.sqrt(np.mean(result.audio ** 2))
            results.append((velocity, rms))

        print(f"\nVelocity response:")
        for vel, rms in results:
            print(f"  Velocity {vel:3d}: RMS = {rms:.4f}")

        # Verify general trend: max velocity should be louder than min
        # Note: Karplus-Strong with random noise can have some variation
        assert results[-1][1] > results[0][1] * 2, \
            f"Max velocity should be significantly louder than min: {results[-1][1]:.4f} vs {results[0][1]:.4f}"

    @pytest.mark.midi
    @pytest.mark.requires_plugin
    def test_different_pitches(self, loaded_plugin: DawDreamerHost):
        """
        Test different pitches produce output.

        Verify low, mid, and high notes all work.
        """
        test_notes = [
            (36, "C2 - Low"),
            (60, "C4 - Middle"),
            (84, "C6 - High"),
        ]

        print(f"\nPitch test:")
        for note, name in test_notes:
            result = loaded_plugin.render_note(
                note=note,
                velocity=100,
                duration_seconds=0.3,
                tail_seconds=0.5
            )
            rms = np.sqrt(np.mean(result.audio ** 2))
            print(f"  {name}: RMS = {rms:.4f}")

            assert rms > 0.001, f"{name} should produce output"


class TestPolyphonyPerformance:
    """Performance tests for polyphony."""

    @pytest.mark.midi
    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_scaling(self, loaded_plugin: DawDreamerHost):
        """
        Measure how CPU scales with polyphony.

        CPU should scale roughly linearly with voice count.
        """
        results = loaded_plugin.measure_cpu_per_voice(
            max_voices=32,
            note=60,
            duration=1.0
        )

        print(f"\nCPU scaling with polyphony:")
        print(f"  {'Voices':>6} {'CPU %':>8} {'Per Voice':>10}")
        print(f"  {'-'*6} {'-'*8} {'-'*10}")

        for num_voices, cpu_percent in results:
            per_voice = cpu_percent / num_voices
            print(f"  {num_voices:>6} {cpu_percent:>7.1f}% {per_voice:>9.2f}%")

        # Verify CPU stays reasonable
        max_cpu = max(cpu for _, cpu in results)
        assert max_cpu < 50, f"CPU usage {max_cpu:.1f}% too high at max polyphony"
