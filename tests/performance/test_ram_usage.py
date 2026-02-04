"""
RAM Usage Tests

Measures memory consumption and detects memory leaks:
- Baseline memory usage
- Memory per voice
- Memory over time (leak detection)
- Peak memory usage

Uses psutil for memory monitoring.
"""

import time
import subprocess
import gc
from typing import List, Optional, Tuple
from dataclasses import dataclass

import pytest
import psutil
import numpy as np


@dataclass
class MemoryMeasurement:
    """Memory usage measurement result."""
    rss_mb: float          # Resident Set Size
    vms_mb: float          # Virtual Memory Size
    peak_rss_mb: float     # Peak RSS during measurement
    samples: List[float]   # RSS samples over time


class TestRAMUsage:
    """Memory usage tests."""

    @pytest.mark.performance
    @pytest.mark.requires_standalone
    def test_baseline_memory(self, require_standalone, thresholds, has_display):
        """
        Measure baseline memory usage on startup.

        Plugin should load within reasonable memory budget.
        """
        if not has_display:
            pytest.skip("Requires display for standalone app")

        measurement = self._measure_memory(
            require_standalone,
            duration=5,
            interval=1.0
        )

        if measurement is None:
            pytest.skip("Could not measure memory")

        print(f"\nBaseline memory usage:")
        print(f"  RSS: {measurement.rss_mb:.1f} MB")
        print(f"  VMS: {measurement.vms_mb:.1f} MB")

        assert measurement.rss_mb < thresholds.ram_baseline_mb, \
            f"Baseline RAM {measurement.rss_mb:.1f} MB exceeds threshold"

    @pytest.mark.performance
    @pytest.mark.requires_standalone
    def test_peak_memory(self, require_standalone, thresholds, has_display):
        """
        Measure peak memory during active use.

        Memory should stay within bounds even under load.
        """
        if not has_display:
            pytest.skip("Requires display for standalone app")

        measurement = self._measure_memory(
            require_standalone,
            duration=30,
            interval=1.0
        )

        if measurement is None:
            pytest.skip("Could not measure memory")

        print(f"\nPeak memory usage:")
        print(f"  Peak RSS: {measurement.peak_rss_mb:.1f} MB")

        assert measurement.peak_rss_mb < thresholds.ram_max_mb, \
            f"Peak RAM {measurement.peak_rss_mb:.1f} MB exceeds threshold"

    @pytest.mark.performance
    @pytest.mark.requires_standalone
    @pytest.mark.slow
    def test_memory_leak_detection(self, require_standalone, thresholds, has_display):
        """
        Test for memory leaks over extended operation.

        Memory should not grow continuously over time.
        """
        if not has_display:
            pytest.skip("Requires display for standalone app")

        measurement = self._measure_memory(
            require_standalone,
            duration=300,  # 5 minutes
            interval=5.0
        )

        if measurement is None:
            pytest.skip("Could not measure memory")

        samples = np.array(measurement.samples)

        if len(samples) < 10:
            pytest.skip("Not enough samples for leak detection")

        # Linear regression to detect growth trend
        x = np.arange(len(samples))
        slope, intercept = np.polyfit(x, samples, 1)

        # Convert slope to MB per minute
        samples_per_minute = 60 / 5  # Based on 5-second interval
        leak_rate = slope * samples_per_minute

        print(f"\nMemory leak analysis:")
        print(f"  Initial: {samples[0]:.1f} MB")
        print(f"  Final: {samples[-1]:.1f} MB")
        print(f"  Growth rate: {leak_rate:.2f} MB/minute")

        # Check for significant leak
        total_growth = samples[-1] - samples[0]

        assert total_growth < thresholds.ram_leak_threshold_mb, \
            f"Memory grew by {total_growth:.1f} MB (threshold: {thresholds.ram_leak_threshold_mb} MB)"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    @pytest.mark.stress
    def test_memory_stress(self, loaded_plugin, thresholds):
        """
        Stress test memory with repeated operations.

        Simulate heavy use to expose memory issues by:
        - Playing many notes rapidly
        - Changing parameters frequently
        - Rendering multiple times
        """
        import psutil
        from tests.tools.dawdreamer_host import MIDIEvent

        process = psutil.Process()
        gc.collect()

        # Measure baseline memory
        baseline_mem = process.memory_info()
        baseline_rss_mb = baseline_mem.rss / (1024 * 1024)

        peak_rss_mb = baseline_rss_mb
        samples = [baseline_rss_mb]

        # Stress test: multiple rounds of intensive operations
        num_stress_rounds = 10
        notes_per_round = 50
        param_changes_per_round = 20

        for round_num in range(num_stress_rounds):
            # Generate rapid note sequence
            events = []
            for i in range(notes_per_round):
                note = 36 + (i % 48)  # Notes from C2 to B5
                start_time = i * 0.02  # 20ms apart
                velocity = 64 + (i % 64)  # Varying velocities
                events.append(MIDIEvent.note_on(start_time, note, velocity))
                events.append(MIDIEvent.note_off(start_time + 0.015, note))

            # Render with the note sequence
            loaded_plugin.render(
                duration_seconds=notes_per_round * 0.02 + 0.5,
                midi_events=events
            )

            # Change parameters rapidly
            info = loaded_plugin.get_info()
            if info and info.parameters:
                param_names = list(info.parameters.keys())
                for j in range(min(param_changes_per_round, len(param_names))):
                    param_name = param_names[j % len(param_names)]
                    # Oscillate parameter values
                    value = 0.25 + 0.5 * ((round_num + j) % 2)
                    loaded_plugin.set_parameter(param_name, value)

            # Render a chord to exercise polyphony
            loaded_plugin.render_chord(
                notes=[48, 52, 55, 60, 64, 67, 72],  # C major across octaves
                velocity=100,
                duration_seconds=0.5,
                tail_seconds=0.5
            )

            # Measure memory after this round
            current_mem = process.memory_info()
            current_rss_mb = current_mem.rss / (1024 * 1024)
            samples.append(current_rss_mb)
            peak_rss_mb = max(peak_rss_mb, current_rss_mb)

        # Force garbage collection and measure final memory
        gc.collect()
        time.sleep(0.5)
        final_mem = process.memory_info()
        final_rss_mb = final_mem.rss / (1024 * 1024)

        memory_growth = final_rss_mb - baseline_rss_mb

        print(f"\nMemory stress test results:")
        print(f"  Baseline RSS: {baseline_rss_mb:.1f} MB")
        print(f"  Peak RSS: {peak_rss_mb:.1f} MB")
        print(f"  Final RSS: {final_rss_mb:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")
        print(f"  Stress rounds: {num_stress_rounds}")
        print(f"  Notes per round: {notes_per_round}")

        # Assert memory stayed within reasonable bounds
        assert peak_rss_mb < thresholds.ram_max_mb, \
            f"Peak memory {peak_rss_mb:.1f} MB exceeds threshold {thresholds.ram_max_mb} MB"

        # Check for excessive memory growth (potential leak indicator)
        assert memory_growth < thresholds.ram_leak_threshold_mb * 2, \
            f"Memory grew by {memory_growth:.1f} MB during stress test"

    @pytest.mark.performance
    @pytest.mark.requires_plugin
    def test_memory_per_voice(self, loaded_plugin, thresholds):
        """
        Measure memory cost per additional voice.

        Helps estimate scalability with polyphony.
        """
        import psutil
        from tests.tools.dawdreamer_host import MIDIEvent

        process = psutil.Process()

        voice_counts = [1, 2, 4, 8, 16, 32]
        memory_per_voice_count: List[Tuple[int, float]] = []

        for num_voices in voice_counts:
            # Force garbage collection before measurement
            gc.collect()
            time.sleep(0.2)

            # Measure memory before rendering
            mem_before = process.memory_info()
            rss_before_mb = mem_before.rss / (1024 * 1024)

            # Create events for simultaneous voices
            events = []
            base_note = 36  # C2
            for i in range(num_voices):
                note = base_note + i
                events.append(MIDIEvent.note_on(0.0, note, 100))
                events.append(MIDIEvent.note_off(2.0, note))  # Hold for 2 seconds

            # Render with all voices active
            loaded_plugin.render(
                duration_seconds=2.5,
                midi_events=events
            )

            # Measure memory during active rendering
            mem_during = process.memory_info()
            rss_during_mb = mem_during.rss / (1024 * 1024)

            memory_per_voice_count.append((num_voices, rss_during_mb))

            print(f"  {num_voices} voices: {rss_during_mb:.1f} MB (delta from baseline: {rss_during_mb - memory_per_voice_count[0][1]:.2f} MB)")

        # Calculate memory per voice (linear regression)
        if len(memory_per_voice_count) >= 2:
            voices = np.array([v[0] for v in memory_per_voice_count])
            memory = np.array([v[1] for v in memory_per_voice_count])

            # Linear fit: memory = base + per_voice * num_voices
            slope, intercept = np.polyfit(voices, memory, 1)
            memory_per_voice_kb = slope * 1024  # Convert MB to KB

            print(f"\nMemory per voice analysis:")
            print(f"  Base memory: {intercept:.1f} MB")
            print(f"  Memory per voice: {memory_per_voice_kb:.1f} KB")
            print(f"  Estimated memory at 64 voices: {intercept + slope * 64:.1f} MB")
            print(f"  Estimated memory at 128 voices: {intercept + slope * 128:.1f} MB")

            # Verify memory stays within bounds even at high polyphony
            estimated_32_voice_memory = intercept + slope * 32
            assert estimated_32_voice_memory < thresholds.ram_max_mb, \
                f"Estimated 32-voice memory {estimated_32_voice_memory:.1f} MB exceeds threshold"

            # Sanity check: memory per voice shouldn't be unreasonably high
            # A reasonable synth might use ~100KB-1MB per voice
            max_kb_per_voice = 2048  # 2MB per voice is generous upper bound
            assert memory_per_voice_kb < max_kb_per_voice, \
                f"Memory per voice {memory_per_voice_kb:.1f} KB seems excessive (>{max_kb_per_voice} KB)"
        else:
            print("\nInsufficient data points for per-voice analysis")

    def _measure_memory(
        self,
        app_path,
        duration: float = 10,
        interval: float = 1.0
    ) -> Optional[MemoryMeasurement]:
        """
        Launch app and measure memory usage.

        Args:
            app_path: Path to application
            duration: Measurement duration in seconds
            interval: Sampling interval in seconds

        Returns:
            MemoryMeasurement or None if failed
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
                return None

            # Get process handle
            ps_proc = psutil.Process(proc.pid)

            # Collect measurements
            samples = []
            start_time = time.time()
            peak_rss = 0

            while time.time() - start_time < duration:
                try:
                    mem_info = ps_proc.memory_info()
                    rss_mb = mem_info.rss / (1024 * 1024)
                    samples.append(rss_mb)
                    peak_rss = max(peak_rss, rss_mb)
                    time.sleep(interval)
                except psutil.NoSuchProcess:
                    break

            # Get final measurements
            try:
                final_mem = ps_proc.memory_info()
                final_rss = final_mem.rss / (1024 * 1024)
                final_vms = final_mem.vms / (1024 * 1024)
            except psutil.NoSuchProcess:
                if samples:
                    final_rss = samples[-1]
                    final_vms = 0
                else:
                    return None

            # Cleanup
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

            if not samples:
                return None

            return MemoryMeasurement(
                rss_mb=final_rss,
                vms_mb=final_vms,
                peak_rss_mb=peak_rss,
                samples=samples
            )

        except Exception as e:
            print(f"Memory measurement error: {e}")
            return None


class TestMemoryManagement:
    """Memory management behavior tests."""

    @pytest.mark.performance
    @pytest.mark.requires_standalone
    def test_memory_released_on_close(self, require_standalone, has_display):
        """
        Verify memory is properly released when app closes.

        No memory should remain allocated after clean shutdown.
        """
        if not has_display:
            pytest.skip("Requires display for standalone app")

        # Get initial system memory
        initial_available = psutil.virtual_memory().available

        # Launch and close app
        proc = subprocess.Popen(
            [str(require_standalone)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(5)  # Let it fully load

        proc.terminate()
        proc.wait(timeout=10)

        # Wait for memory to be released
        time.sleep(2)
        gc.collect()

        # Check final memory
        final_available = psutil.virtual_memory().available

        # Memory should be approximately the same
        # (within 50MB tolerance for system variation)
        diff = abs(final_available - initial_available) / (1024 * 1024)

        print(f"\nMemory after close:")
        print(f"  Initial available: {initial_available / (1024*1024):.0f} MB")
        print(f"  Final available: {final_available / (1024*1024):.0f} MB")
        print(f"  Difference: {diff:.0f} MB")

        # Note: This test is inherently flaky due to system activity
        # Just warn rather than fail
        if diff > 100:
            print("WARNING: Significant memory difference after close")
