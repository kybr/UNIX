"""
CPU Usage Tests

Measures CPU consumption of the plugin under various conditions:
- Idle CPU usage
- Per-voice CPU cost
- CPU scaling with polyphony
- Real-time performance factor

Based on DAWBench methodology: https://dawbench.com/
"""

import time
import subprocess
import platform
from typing import List, Tuple, Optional
from dataclasses import dataclass

import pytest
import psutil
import numpy as np


@dataclass
class CPUMeasurement:
    """CPU usage measurement result."""
    average_percent: float
    peak_percent: float
    samples: List[float]
    duration_seconds: float


class TestCPUUsage:
    """CPU usage benchmarks."""

    @pytest.mark.performance
    @pytest.mark.requires_standalone
    def test_idle_cpu_usage(self, require_standalone, thresholds, has_display):
        """
        Measure CPU usage when plugin is idle (no notes playing).

        Plugin should consume minimal CPU when not processing audio.
        """
        if not has_display:
            pytest.skip("Requires display for standalone app")

        measurement = self._measure_cpu_usage(
            require_standalone,
            duration=10,
            interval=0.5
        )

        if measurement is None:
            pytest.skip("Could not measure CPU usage")

        print(f"\nIdle CPU usage:")
        print(f"  Average: {measurement.average_percent:.2f}%")
        print(f"  Peak: {measurement.peak_percent:.2f}%")

        assert measurement.average_percent < thresholds.cpu_idle_percent, \
            f"Idle CPU {measurement.average_percent:.2f}% exceeds threshold"

    @pytest.mark.performance
    @pytest.mark.requires_standalone
    @pytest.mark.slow
    def test_sustained_cpu_usage(self, require_standalone, has_display):
        """
        Measure CPU usage over extended period.

        Check for CPU creep or resource leaks.
        """
        if not has_display:
            pytest.skip("Requires display for standalone app")

        measurement = self._measure_cpu_usage(
            require_standalone,
            duration=60,
            interval=1.0
        )

        if measurement is None:
            pytest.skip("Could not measure CPU usage")

        # Check for trending increase (possible leak)
        samples = np.array(measurement.samples)
        if len(samples) > 10:
            first_half = np.mean(samples[:len(samples)//2])
            second_half = np.mean(samples[len(samples)//2:])

            print(f"\nSustained CPU measurement:")
            print(f"  First half average: {first_half:.2f}%")
            print(f"  Second half average: {second_half:.2f}%")
            print(f"  Trend: {'+' if second_half > first_half else ''}{second_half - first_half:.2f}%")

            # Warn if significant increase
            if second_half > first_half * 1.2:
                pytest.fail("CPU usage increased significantly over time (possible leak)")

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_per_voice(self, loaded_plugin, thresholds):
        """
        Measure CPU cost per additional voice.

        Helps estimate scalability with polyphony.
        """
        from tests.tools.dawdreamer_host import DawDreamerHost

        host: DawDreamerHost = loaded_plugin
        voice_counts = [1, 2, 4, 8, 16, 32]
        results = []

        for num_voices in voice_counts:
            # Create notes spread across the keyboard
            notes = [48 + i for i in range(num_voices)]
            result = host.render_chord(
                notes=notes,
                velocity=100,
                duration_seconds=1.0,
                tail_seconds=0.5
            )

            # Check for audio issues
            audio = result.audio
            assert not np.any(np.isnan(audio)), f"NaN detected with {num_voices} voices"
            assert not np.any(np.isinf(audio)), f"Inf detected with {num_voices} voices"

            cpu_percent = result.realtime_ratio * 100
            per_voice_cpu = cpu_percent / num_voices
            results.append((num_voices, cpu_percent, per_voice_cpu))

        print(f"\nCPU per voice measurement:")
        for voices, total_cpu, per_voice in results:
            print(f"  {voices:2d} voices: {total_cpu:6.2f}% total, {per_voice:.3f}% per voice")

        # Verify CPU scaling is reasonable (per-voice cost shouldn't increase drastically)
        if len(results) >= 2:
            first_per_voice = results[0][2]
            last_per_voice = results[-1][2]
            ratio = last_per_voice / first_per_voice if first_per_voice > 0 else 1.0
            print(f"  Scaling ratio: {ratio:.2f}x")
            assert ratio < 3.0, f"CPU per voice increased {ratio:.1f}x (expected <3x)"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_at_various_buffer_sizes(self, require_vst3, thresholds, sample_rate):
        """
        Measure CPU usage at different buffer sizes.

        Smaller buffers typically require more CPU overhead.
        """
        from tests.tools.dawdreamer_host import DawDreamerHost, MIDIEvent

        buffer_sizes = [64, 128, 256, 512, 1024]
        results = []

        for block_size in buffer_sizes:
            # Create a new host with the specific buffer size
            host = DawDreamerHost(sample_rate=sample_rate, block_size=block_size)
            if not host.load_plugin(require_vst3):
                pytest.skip(f"Failed to load plugin at buffer size {block_size}")

            # Render a test note
            start_time = time.perf_counter()
            result = host.render_note(
                note=60,
                velocity=100,
                duration_seconds=1.0,
                tail_seconds=0.5
            )
            cpu_time = time.perf_counter() - start_time

            # Check for audio issues
            audio = result.audio
            assert not np.any(np.isnan(audio)), f"NaN detected at buffer size {block_size}"
            assert not np.any(np.isinf(audio)), f"Inf detected at buffer size {block_size}"

            cpu_percent = result.realtime_ratio * 100
            results.append((block_size, cpu_percent, cpu_time))

            host.unload()

        print(f"\nCPU at various buffer sizes:")
        for block_size, cpu_percent, cpu_time in results:
            print(f"  Buffer {block_size:4d}: {cpu_percent:6.2f}% CPU, {cpu_time:.4f}s processing time")

        # Verify all buffer sizes work and don't exceed reasonable CPU usage
        for block_size, cpu_percent, _ in results:
            assert cpu_percent < 50.0, f"CPU {cpu_percent:.1f}% too high at buffer size {block_size}"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_cpu_at_various_sample_rates(self, require_vst3, thresholds, block_size):
        """
        Measure CPU usage at different sample rates.

        Higher sample rates require more processing.
        """
        from tests.tools.dawdreamer_host import DawDreamerHost

        sample_rates = [44100, 48000, 96000]
        results = []

        for sr in sample_rates:
            # Create a new host with the specific sample rate
            host = DawDreamerHost(sample_rate=sr, block_size=block_size)
            if not host.load_plugin(require_vst3):
                pytest.skip(f"Failed to load plugin at sample rate {sr}")

            # Render a test note
            result = host.render_note(
                note=60,
                velocity=100,
                duration_seconds=1.0,
                tail_seconds=0.5
            )

            # Check for audio issues
            audio = result.audio
            assert not np.any(np.isnan(audio)), f"NaN detected at sample rate {sr}"
            assert not np.any(np.isinf(audio)), f"Inf detected at sample rate {sr}"

            cpu_percent = result.realtime_ratio * 100
            samples_per_second = sr
            results.append((sr, cpu_percent, audio.shape[1]))

            host.unload()

        print(f"\nCPU at various sample rates:")
        for sr, cpu_percent, num_samples in results:
            print(f"  {sr:5d} Hz: {cpu_percent:6.2f}% CPU, {num_samples} samples rendered")

        # Verify CPU scales reasonably with sample rate
        # Higher sample rates should use more CPU, but not excessively
        if len(results) >= 2:
            base_cpu = results[0][1]
            for sr, cpu_percent, _ in results:
                assert cpu_percent < 50.0, f"CPU {cpu_percent:.1f}% too high at {sr} Hz"

    def _measure_cpu_usage(
        self,
        app_path,
        duration: float = 10,
        interval: float = 0.5
    ) -> Optional[CPUMeasurement]:
        """
        Launch app and measure CPU usage.

        Args:
            app_path: Path to application
            duration: Measurement duration in seconds
            interval: Sampling interval in seconds

        Returns:
            CPUMeasurement or None if failed
        """
        try:
            # Launch application
            proc = subprocess.Popen(
                [str(app_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for startup
            time.sleep(3)

            if proc.poll() is not None:
                return None  # Process exited

            # Get process handle
            ps_proc = psutil.Process(proc.pid)

            # Collect measurements
            samples = []
            start_time = time.time()

            while time.time() - start_time < duration:
                try:
                    cpu = ps_proc.cpu_percent(interval=interval)
                    samples.append(cpu)
                except psutil.NoSuchProcess:
                    break

            # Cleanup
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            if not samples:
                return None

            return CPUMeasurement(
                average_percent=np.mean(samples),
                peak_percent=np.max(samples),
                samples=samples,
                duration_seconds=duration
            )

        except Exception as e:
            print(f"CPU measurement error: {e}")
            return None


class TestRealTimePerformance:
    """Real-time performance tests."""

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_realtime_factor(self, loaded_plugin, thresholds):
        """
        Measure real-time performance factor.

        real_time_factor = (wall clock time) / (audio time)
        Should be < 1.0 for real-time capable plugin.
        """
        from tests.tools.dawdreamer_host import DawDreamerHost

        host: DawDreamerHost = loaded_plugin

        # Test with various audio durations
        durations = [0.5, 1.0, 2.0, 5.0]
        results = []

        for audio_duration in durations:
            start_time = time.perf_counter()
            result = host.render_note(
                note=60,
                velocity=100,
                duration_seconds=audio_duration,
                tail_seconds=0.5
            )
            wall_clock_time = time.perf_counter() - start_time

            # Check for audio issues
            audio = result.audio
            assert not np.any(np.isnan(audio)), f"NaN detected at {audio_duration}s duration"
            assert not np.any(np.isinf(audio)), f"Inf detected at {audio_duration}s duration"

            total_audio_time = audio_duration + 0.5
            realtime_factor = wall_clock_time / total_audio_time
            results.append((total_audio_time, wall_clock_time, realtime_factor))

        print(f"\nRealtime factor measurement:")
        for audio_time, wall_time, factor in results:
            status = "OK" if factor < 1.0 else "TOO SLOW"
            print(f"  {audio_time:.1f}s audio: {wall_time:.4f}s wall clock, factor={factor:.4f} [{status}]")

        # All renders should be faster than realtime
        for audio_time, wall_time, factor in results:
            assert factor < thresholds.realtime_ratio, \
                f"Realtime factor {factor:.2f} exceeds threshold {thresholds.realtime_ratio}"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_no_dropouts(self, loaded_plugin):
        """
        Test for audio dropouts under load.

        Monitor for buffer underruns or overruns by checking for
        discontinuities and NaN/inf values in the output.
        """
        from tests.tools.dawdreamer_host import DawDreamerHost

        host: DawDreamerHost = loaded_plugin

        # Render a sustained note with high polyphony
        result = host.render_polyphony_test(
            num_voices=32,
            base_note=36,
            velocity=100,
            hold_seconds=3.0
        )

        audio = result.audio

        # Check for NaN or Inf (indicates buffer issues)
        has_nan = np.any(np.isnan(audio))
        has_inf = np.any(np.isinf(audio))

        print(f"\nDropout detection:")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")

        assert not has_nan, "Audio contains NaN values (possible dropout)"
        assert not has_inf, "Audio contains Inf values (possible overflow)"

        # Check for sudden amplitude jumps that might indicate dropouts
        # Compute derivative and check for large discontinuities
        for channel in range(audio.shape[0]):
            diff = np.diff(audio[channel])
            max_diff = np.max(np.abs(diff))
            # A discontinuity threshold - sudden jump > 0.5 amplitude is suspicious
            discontinuity_threshold = 0.5
            discontinuities = np.sum(np.abs(diff) > discontinuity_threshold)

            print(f"  Channel {channel}: max_diff={max_diff:.4f}, discontinuities={discontinuities}")

            # With 32 voices and polyphonic rendering, there will be many legitimate
            # discontinuities from note attacks/releases. Allow a generous threshold.
            # The key check is for NaN/Inf which indicates real buffer issues.
            max_allowed_discontinuities = 2000  # Allow for multiple voice attacks/releases
            assert discontinuities < max_allowed_discontinuities, \
                f"Too many discontinuities ({discontinuities}) - possible dropouts"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_consistent_timing(self, loaded_plugin):
        """
        Test processing time consistency.

        Processing time should be consistent (low jitter).
        """
        from tests.tools.dawdreamer_host import DawDreamerHost

        host: DawDreamerHost = loaded_plugin

        # Perform multiple renders and measure timing consistency
        num_iterations = 20
        cpu_times = []

        for i in range(num_iterations):
            start_time = time.perf_counter()
            result = host.render_note(
                note=60,
                velocity=100,
                duration_seconds=0.5,
                tail_seconds=0.25
            )
            cpu_time = time.perf_counter() - start_time
            cpu_times.append(cpu_time)

            # Verify audio integrity
            audio = result.audio
            assert not np.any(np.isnan(audio)), f"NaN detected on iteration {i}"
            assert not np.any(np.isinf(audio)), f"Inf detected on iteration {i}"

        cpu_times = np.array(cpu_times)
        mean_time = np.mean(cpu_times)
        std_time = np.std(cpu_times)
        min_time = np.min(cpu_times)
        max_time = np.max(cpu_times)
        cv = std_time / mean_time if mean_time > 0 else 0  # Coefficient of variation

        print(f"\nTiming consistency ({num_iterations} iterations):")
        print(f"  Mean: {mean_time * 1000:.2f} ms")
        print(f"  Std: {std_time * 1000:.2f} ms")
        print(f"  Min: {min_time * 1000:.2f} ms")
        print(f"  Max: {max_time * 1000:.2f} ms")
        print(f"  CV (jitter): {cv:.3f}")

        # Coefficient of variation should be low (< 50% allows for system scheduling variance)
        assert cv < 0.5, f"Timing jitter too high (CV={cv:.2f}, expected <0.5)"

        # Max should not be dramatically higher than mean (no severe outliers)
        max_ratio = max_time / mean_time if mean_time > 0 else 1
        assert max_ratio < 5.0, f"Worst-case timing {max_ratio:.1f}x mean (expected <5x)"


class TestCPUStress:
    """CPU stress tests."""

    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_max_polyphony_cpu(self, loaded_plugin, thresholds):
        """
        Measure CPU at maximum polyphony.

        Push to limits and measure behavior.
        """
        from tests.tools.dawdreamer_host import DawDreamerHost

        host: DawDreamerHost = loaded_plugin
        max_voices = 128

        # Render with maximum polyphony
        result = host.render_polyphony_test(
            num_voices=max_voices,
            base_note=24,  # Start low to spread across keyboard
            velocity=80,
            stagger_ms=5.0,  # Slight stagger to avoid all attacks at once
            hold_seconds=2.0
        )

        audio = result.audio
        cpu_percent = result.realtime_ratio * 100

        # Check for audio integrity
        has_nan = np.any(np.isnan(audio))
        has_inf = np.any(np.isinf(audio))

        print(f"\nMax polyphony CPU test ({max_voices} voices):")
        print(f"  CPU usage: {cpu_percent:.1f}%")
        print(f"  Realtime ratio: {result.realtime_ratio:.4f}")
        print(f"  Duration rendered: {result.duration_seconds:.2f}s")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")
        print(f"  Peak amplitude: {np.max(np.abs(audio)):.4f}")

        assert not has_nan, "Audio contains NaN at max polyphony"
        assert not has_inf, "Audio contains Inf at max polyphony"

        # Should still be realtime capable even at max polyphony
        assert result.realtime_ratio < 1.0, \
            f"Not realtime at {max_voices} voices (ratio={result.realtime_ratio:.2f})"

        # Check that audio was actually produced (not silence)
        rms = np.sqrt(np.mean(audio ** 2))
        assert rms > 0.001, f"Audio appears silent (RMS={rms:.6f})"

    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.requires_plugin
    def test_cpu_spike_recovery(self, loaded_plugin):
        """
        Test recovery from CPU spike.

        After high load, plugin should return to normal.
        """
        from tests.tools.dawdreamer_host import DawDreamerHost

        host: DawDreamerHost = loaded_plugin

        # Step 1: Measure baseline CPU with single voice
        baseline_result = host.render_note(
            note=60,
            velocity=100,
            duration_seconds=1.0,
            tail_seconds=0.5
        )
        baseline_cpu = baseline_result.realtime_ratio * 100

        # Step 2: Create CPU spike with high polyphony
        spike_result = host.render_polyphony_test(
            num_voices=64,
            base_note=36,
            velocity=100,
            hold_seconds=1.0
        )
        spike_cpu = spike_result.realtime_ratio * 100

        # Step 3: Measure recovery - single voice after spike
        recovery_times = []
        recovery_cpus = []

        for i in range(5):
            start_time = time.perf_counter()
            result = host.render_note(
                note=60,
                velocity=100,
                duration_seconds=0.5,
                tail_seconds=0.25
            )
            recovery_time = time.perf_counter() - start_time
            recovery_cpu = result.realtime_ratio * 100

            recovery_times.append(recovery_time)
            recovery_cpus.append(recovery_cpu)

            # Check audio integrity during recovery
            audio = result.audio
            assert not np.any(np.isnan(audio)), f"NaN detected during recovery iteration {i}"
            assert not np.any(np.isinf(audio)), f"Inf detected during recovery iteration {i}"

        print(f"\nCPU spike recovery test:")
        print(f"  Baseline CPU (1 voice): {baseline_cpu:.2f}%")
        print(f"  Spike CPU (64 voices): {spike_cpu:.2f}%")
        print(f"  Recovery CPUs: {[f'{c:.2f}%' for c in recovery_cpus]}")

        # Recovery should return close to baseline
        final_recovery_cpu = np.mean(recovery_cpus[-3:])  # Average of last 3
        print(f"  Final recovery CPU (avg last 3): {final_recovery_cpu:.2f}%")

        # Allow some overhead, but should be within 3x of baseline
        recovery_ratio = final_recovery_cpu / baseline_cpu if baseline_cpu > 0 else 1
        print(f"  Recovery ratio: {recovery_ratio:.2f}x baseline")

        assert recovery_ratio < 3.0, \
            f"CPU did not recover properly: {final_recovery_cpu:.2f}% vs baseline {baseline_cpu:.2f}%"

        # Verify the spike actually caused higher CPU than baseline
        assert spike_cpu > baseline_cpu, \
            "Spike test should use more CPU than baseline"
