"""
Real Performance Tests (using DawDreamer)

Actual CPU and latency measurements using the loaded plugin.
"""

import time
import numpy as np
import pytest

from tests.tools.dawdreamer_host import DawDreamerHost, MIDIEvent


class TestRealPerformance:
    """Real performance measurements using DawDreamer plugin host."""

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_faster_than_realtime(self, loaded_plugin: DawDreamerHost):
        """
        Verify plugin processes faster than realtime.

        Essential for any audio plugin.
        """
        result = loaded_plugin.render_note(
            note=60,
            velocity=100,
            duration_seconds=1.0,
            tail_seconds=1.0
        )

        print(f"\nRealtime performance:")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  CPU time: {result.cpu_time_seconds:.3f}s")
        print(f"  Realtime ratio: {result.realtime_ratio:.4f}")
        print(f"  CPU usage: {result.realtime_ratio * 100:.1f}%")

        assert result.realtime_ratio < 1.0, \
            f"Plugin too slow: {result.realtime_ratio:.2f}x realtime"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_single_voice(self, loaded_plugin: DawDreamerHost, thresholds):
        """
        Measure CPU usage for a single voice.

        Should be very low for Karplus-Strong.
        """
        result = loaded_plugin.render_note(
            note=60,
            velocity=100,
            duration_seconds=2.0,
            tail_seconds=0.0
        )

        cpu_percent = result.realtime_ratio * 100

        print(f"\nSingle voice CPU: {cpu_percent:.2f}%")

        assert cpu_percent < thresholds.cpu_single_voice_percent, \
            f"Single voice CPU {cpu_percent:.2f}% exceeds threshold"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_16_voices(self, loaded_plugin: DawDreamerHost, thresholds):
        """Measure CPU usage with 16 simultaneous voices."""
        result = loaded_plugin.render_polyphony_test(
            num_voices=16,
            base_note=36,
            velocity=100,
            hold_seconds=2.0
        )

        cpu_percent = result.realtime_ratio * 100

        print(f"\n16-voice CPU: {cpu_percent:.2f}%")

        threshold = thresholds.cpu_16_voices_percent
        assert cpu_percent < threshold, \
            f"16-voice CPU {cpu_percent:.2f}% exceeds threshold {threshold}%"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_scaling_linearity(self, loaded_plugin: DawDreamerHost):
        """
        Test that CPU scales roughly linearly with voices.

        Doubling voices should roughly double CPU.
        """
        results = loaded_plugin.measure_cpu_per_voice(
            max_voices=32,
            duration=1.0
        )

        print(f"\nCPU scaling:")
        for voices, cpu in results:
            per_voice = cpu / voices
            print(f"  {voices:2d} voices: {cpu:5.2f}% ({per_voice:.3f}% per voice)")

        # Check scaling is reasonable
        # CPU per voice shouldn't increase dramatically
        per_voice_values = [cpu / voices for voices, cpu in results]

        if len(per_voice_values) >= 2:
            ratio = per_voice_values[-1] / per_voice_values[0]
            assert ratio < 3.0, \
                f"CPU per voice increased {ratio:.1f}x (expected <3x)"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_latency_reported(self, loaded_plugin: DawDreamerHost):
        """Check plugin reports latency."""
        result = loaded_plugin.render_note(note=60, velocity=100)

        latency_samples = result.plugin_latency_samples
        latency_ms = latency_samples / loaded_plugin.sample_rate * 1000

        print(f"\nPlugin latency:")
        print(f"  Samples: {latency_samples}")
        print(f"  Time: {latency_ms:.2f} ms")

        # Karplus-Strong shouldn't have significant latency
        assert latency_ms < 50, f"Latency {latency_ms:.1f}ms too high"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_consistent_performance(self, loaded_plugin: DawDreamerHost):
        """
        Verify consistent CPU usage across multiple renders.

        Performance should be stable.
        """
        cpu_samples = []

        for _ in range(10):
            result = loaded_plugin.render_note(
                note=60,
                velocity=100,
                duration_seconds=0.5,
                tail_seconds=0.5
            )
            cpu_samples.append(result.realtime_ratio * 100)

        mean_cpu = np.mean(cpu_samples)
        std_cpu = np.std(cpu_samples)
        cv = std_cpu / mean_cpu if mean_cpu > 0 else 0

        print(f"\nPerformance consistency (10 renders):")
        print(f"  Mean CPU: {mean_cpu:.2f}%")
        print(f"  Std dev: {std_cpu:.3f}%")
        print(f"  CV: {cv:.3f}")

        # Coefficient of variation should be reasonably low (< 30%)
        # Note: System background processes can cause some variation
        assert cv < 0.3, f"Performance too variable (CV={cv:.2f})"


class TestStressPerformance:
    """Stress test performance."""

    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_max_polyphony_performance(self, loaded_plugin: DawDreamerHost):
        """
        Measure performance at maximum practical polyphony.

        64 voices should still be realtime capable.
        """
        result = loaded_plugin.render_polyphony_test(
            num_voices=64,
            base_note=24,
            velocity=80,
            hold_seconds=2.0
        )

        cpu_percent = result.realtime_ratio * 100

        print(f"\n64-voice stress test:")
        print(f"  CPU: {cpu_percent:.1f}%")
        print(f"  Realtime capable: {result.realtime_ratio < 1.0}")

        assert result.realtime_ratio < 1.0, \
            "64 voices should still be realtime capable"

    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_sustained_load(self, loaded_plugin: DawDreamerHost):
        """
        Test performance under sustained load.

        Render many notes sequentially and check for degradation.
        """
        cpu_over_time = []

        for i in range(20):
            result = loaded_plugin.render_note(
                note=48 + (i % 12),
                velocity=100,
                duration_seconds=0.2,
                tail_seconds=0.3
            )
            cpu_over_time.append(result.realtime_ratio * 100)

        print(f"\nSustained load (20 renders):")
        print(f"  First: {cpu_over_time[0]:.2f}%")
        print(f"  Last: {cpu_over_time[-1]:.2f}%")
        print(f"  Max: {max(cpu_over_time):.2f}%")

        # Should not degrade significantly over time
        # Note: Some variation is expected due to system scheduling and cache effects
        degradation = cpu_over_time[-1] / cpu_over_time[0] if cpu_over_time[0] > 0 else 1
        assert degradation < 3.0, \
            f"Performance degraded by {degradation:.1f}x over time"
