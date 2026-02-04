"""
Long Session Stress Tests

Test plugin stability over extended operation:
- Memory leak detection
- CPU stability
- Audio quality consistency
- Resource cleanup

These tests simulate extended sessions by rendering many short segments
rather than running in real-time for hours.
"""

import time
import subprocess
import gc
import os
from typing import List, Optional
from dataclasses import dataclass

import pytest
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class SessionMetrics:
    """Metrics collected during long session."""
    duration_hours: float
    cpu_samples: List[float]
    ram_samples: List[float]
    errors_detected: int
    audio_glitches: int


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class TestLongSession:
    """Extended operation stability tests using DawDreamer."""

    @pytest.mark.stress
    @pytest.mark.slow
    def test_1_hour_session(self, loaded_plugin, thresholds):
        """
        Simulate 1 hour of plugin use via DawDreamer.

        Renders 100 x 10 second segments (simulating ~16 minutes of audio)
        while monitoring memory and CPU usage.

        Monitor for:
        - Memory leaks
        - CPU creep
        - Audio issues
        - Crashes
        """
        if not HAS_PSUTIL:
            pytest.skip("psutil required for monitoring")

        from tests.tools.dawdreamer_host import MIDIEvent

        num_segments = 100
        segment_duration = 10.0  # seconds per segment

        cpu_samples = []
        ram_samples = []
        errors = 0

        # Initial memory reading
        gc.collect()
        initial_ram = get_process_memory_mb()
        ram_samples.append(initial_ram)

        print(f"\nSimulating 1-hour session with {num_segments} x {segment_duration}s renders")
        print(f"Initial RAM: {initial_ram:.1f} MB")

        for i in range(num_segments):
            try:
                # Create varied MIDI events for each segment
                events = []
                note_base = 48 + (i % 24)  # Vary the base note
                num_notes = 4 + (i % 5)  # 4-8 notes per segment

                for j in range(num_notes):
                    note_time = j * (segment_duration / num_notes)
                    note = note_base + (j * 4) % 12
                    events.append(MIDIEvent.note_on(note_time, note, 80 + (j * 10) % 48))
                    events.append(MIDIEvent.note_off(note_time + 0.5, note))

                # Render segment
                result = loaded_plugin.render(segment_duration, events)

                # Record CPU (realtime ratio as percentage)
                cpu_samples.append(result.realtime_ratio * 100)

                # Record memory every 10 segments
                if i % 10 == 0:
                    gc.collect()
                    current_ram = get_process_memory_mb()
                    ram_samples.append(current_ram)
                    print(f"  Segment {i}/{num_segments}: CPU={result.realtime_ratio*100:.1f}%, RAM={current_ram:.1f} MB")

                # Check for audio issues (silence or clipping)
                audio = result.audio
                if np.max(np.abs(audio)) < 0.001:
                    errors += 1
                    print(f"  WARNING: Silent output at segment {i}")
                if np.max(np.abs(audio)) > 1.0:
                    errors += 1
                    print(f"  WARNING: Clipping at segment {i}")

            except Exception as e:
                errors += 1
                print(f"  ERROR at segment {i}: {e}")

        # Final memory reading
        gc.collect()
        final_ram = get_process_memory_mb()
        ram_samples.append(final_ram)

        # Analysis
        cpu_arr = np.array(cpu_samples)
        ram_arr = np.array(ram_samples)
        ram_growth = final_ram - initial_ram

        print(f"\nSession Summary:")
        print(f"  Simulated duration: {num_segments * segment_duration / 60:.1f} minutes")
        print(f"  CPU - Avg: {np.mean(cpu_arr):.1f}%, Peak: {np.max(cpu_arr):.1f}%")
        print(f"  RAM - Start: {initial_ram:.1f} MB, End: {final_ram:.1f} MB")
        print(f"  RAM Growth: {ram_growth:.1f} MB")
        print(f"  Errors: {errors}")

        # Assertions
        leak_threshold = getattr(thresholds, 'ram_leak_threshold_mb', 10.0)
        assert ram_growth < leak_threshold, \
            f"Memory grew by {ram_growth:.1f} MB (threshold: {leak_threshold} MB)"
        assert errors == 0, f"Detected {errors} errors during session"

    @pytest.mark.stress
    @pytest.mark.slow
    def test_overnight_session(self, loaded_plugin, thresholds):
        """
        Simulate overnight (8 hour) session via DawDreamer.

        Renders 500 x 10 second segments while monitoring resources.
        Catches slow memory leaks and rare crashes.
        """
        if not HAS_PSUTIL:
            pytest.skip("psutil required for monitoring")

        from tests.tools.dawdreamer_host import MIDIEvent

        num_segments = 500  # More segments for longer simulation
        segment_duration = 10.0

        cpu_samples = []
        ram_samples = []
        errors = 0

        gc.collect()
        initial_ram = get_process_memory_mb()
        ram_samples.append(initial_ram)

        print(f"\nSimulating overnight session with {num_segments} x {segment_duration}s renders")
        print(f"Initial RAM: {initial_ram:.1f} MB")

        for i in range(num_segments):
            try:
                # Create varied MIDI events
                events = []
                note_base = 36 + (i % 36)
                num_notes = 2 + (i % 8)

                for j in range(num_notes):
                    note_time = j * (segment_duration / num_notes)
                    note = note_base + (j * 3) % 24
                    velocity = 60 + (i * j) % 68
                    events.append(MIDIEvent.note_on(note_time, note, velocity))
                    events.append(MIDIEvent.note_off(note_time + 1.0, note))

                result = loaded_plugin.render(segment_duration, events)
                cpu_samples.append(result.realtime_ratio * 100)

                # Sample memory every 50 segments
                if i % 50 == 0:
                    gc.collect()
                    current_ram = get_process_memory_mb()
                    ram_samples.append(current_ram)
                    print(f"  Segment {i}/{num_segments}: CPU={result.realtime_ratio*100:.1f}%, RAM={current_ram:.1f} MB")

                # Check for audio issues
                audio = result.audio
                if np.max(np.abs(audio)) < 0.001:
                    errors += 1
                if np.max(np.abs(audio)) > 1.0:
                    errors += 1

            except Exception as e:
                errors += 1
                print(f"  ERROR at segment {i}: {e}")

        gc.collect()
        final_ram = get_process_memory_mb()
        ram_samples.append(final_ram)

        cpu_arr = np.array(cpu_samples)
        ram_arr = np.array(ram_samples)
        ram_growth = final_ram - initial_ram

        print(f"\nOvernight Session Summary:")
        print(f"  Simulated duration: {num_segments * segment_duration / 60:.1f} minutes")
        print(f"  CPU - Avg: {np.mean(cpu_arr):.1f}%, Peak: {np.max(cpu_arr):.1f}%, Std: {np.std(cpu_arr):.1f}%")
        print(f"  RAM - Start: {initial_ram:.1f} MB, End: {final_ram:.1f} MB")
        print(f"  RAM Growth: {ram_growth:.1f} MB")
        print(f"  Errors: {errors}")

        # More lenient threshold for longer session
        leak_threshold = getattr(thresholds, 'ram_leak_threshold_mb', 10.0) * 2
        assert ram_growth < leak_threshold, \
            f"Memory grew by {ram_growth:.1f} MB (threshold: {leak_threshold} MB)"
        assert errors == 0, f"Detected {errors} errors during session"

        # Check CPU stability
        if len(cpu_arr) > 10:
            cpu_std = np.std(cpu_arr)
            assert cpu_std < 30, f"CPU usage unstable (std dev: {cpu_std:.1f}%)"

    @pytest.mark.stress
    @pytest.mark.slow
    def test_session_with_activity(self, loaded_plugin, thresholds):
        """
        Long session with simulated user activity via DawDreamer.

        Periodically trigger notes, change parameters, and test
        plugin response under varied conditions.
        """
        if not HAS_PSUTIL:
            pytest.skip("psutil required for monitoring")

        from tests.tools.dawdreamer_host import MIDIEvent

        num_segments = 50
        segment_duration = 5.0

        cpu_samples = []
        ram_samples = []
        errors = 0

        gc.collect()
        initial_ram = get_process_memory_mb()

        print(f"\nSimulating session with activity: {num_segments} segments")
        print(f"Initial RAM: {initial_ram:.1f} MB")

        # Get parameter info
        info = loaded_plugin.get_info()
        param_names = list(info.parameters.keys()) if info else []

        for i in range(num_segments):
            try:
                # Change parameters periodically (simulating user tweaking knobs)
                if param_names and i % 5 == 0:
                    param_idx = i % len(param_names)
                    param_name = param_names[param_idx]
                    new_value = (i * 0.1) % 1.0  # Vary between 0 and 1
                    loaded_plugin.set_parameter(param_name, new_value)
                    print(f"  Set {param_name} = {new_value:.2f}")

                # Create varied MIDI patterns
                events = []

                if i % 3 == 0:
                    # Single notes
                    for j in range(8):
                        t = j * 0.5
                        note = 48 + (j * 5) % 36
                        events.append(MIDIEvent.note_on(t, note, 90))
                        events.append(MIDIEvent.note_off(t + 0.3, note))
                elif i % 3 == 1:
                    # Chords
                    for chord_idx in range(4):
                        t = chord_idx * 1.0
                        for note_offset in [0, 4, 7]:  # Major triad
                            note = 48 + (chord_idx * 5) + note_offset
                            events.append(MIDIEvent.note_on(t, note, 80))
                            events.append(MIDIEvent.note_off(t + 0.8, note))
                else:
                    # Fast arpeggios
                    for j in range(16):
                        t = j * 0.2
                        note = 36 + (j * 7) % 48
                        events.append(MIDIEvent.note_on(t, note, 100))
                        events.append(MIDIEvent.note_off(t + 0.15, note))

                result = loaded_plugin.render(segment_duration, events)
                cpu_samples.append(result.realtime_ratio * 100)

                # Sample memory every 10 segments
                if i % 10 == 0:
                    gc.collect()
                    current_ram = get_process_memory_mb()
                    ram_samples.append(current_ram)
                    print(f"  Segment {i}: CPU={result.realtime_ratio*100:.1f}%, RAM={current_ram:.1f} MB")

                # Check audio output
                audio = result.audio
                if np.max(np.abs(audio)) < 0.001:
                    errors += 1
                    print(f"  WARNING: Silent output at segment {i}")

            except Exception as e:
                errors += 1
                print(f"  ERROR at segment {i}: {e}")

        gc.collect()
        final_ram = get_process_memory_mb()
        ram_growth = final_ram - initial_ram

        print(f"\nActivity Session Summary:")
        print(f"  CPU - Avg: {np.mean(cpu_samples):.1f}%, Peak: {np.max(cpu_samples):.1f}%")
        print(f"  RAM Growth: {ram_growth:.1f} MB")
        print(f"  Errors: {errors}")

        leak_threshold = getattr(thresholds, 'ram_leak_threshold_mb', 10.0)
        assert ram_growth < leak_threshold, \
            f"Memory grew by {ram_growth:.1f} MB (threshold: {leak_threshold} MB)"
        assert errors == 0, f"Detected {errors} errors during activity session"

    def _run_session(
        self,
        app_path,
        duration_seconds: float,
        sample_interval: float = 60
    ) -> Optional[SessionMetrics]:
        """
        Run a monitoring session.

        Args:
            app_path: Path to application
            duration_seconds: Total duration
            sample_interval: Time between measurements

        Returns:
            SessionMetrics or None if failed
        """
        try:
            # Launch application
            proc = subprocess.Popen(
                [str(app_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            time.sleep(5)  # Wait for startup

            if proc.poll() is not None:
                return None

            ps_proc = psutil.Process(proc.pid)

            # Collect metrics
            cpu_samples = []
            ram_samples = []
            start_time = time.time()
            errors = 0

            while time.time() - start_time < duration_seconds:
                try:
                    cpu = ps_proc.cpu_percent(interval=1)
                    mem = ps_proc.memory_info().rss / (1024 * 1024)

                    cpu_samples.append(cpu)
                    ram_samples.append(mem)

                    elapsed = time.time() - start_time
                    print(f"  [{elapsed/3600:.2f}h] CPU: {cpu:.1f}%, RAM: {mem:.1f} MB")

                except psutil.NoSuchProcess:
                    errors += 1
                    print("  ERROR: Process died unexpectedly")
                    break

                time.sleep(sample_interval - 1)  # Account for cpu_percent time

            # Cleanup
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

            return SessionMetrics(
                duration_hours=duration_seconds / 3600,
                cpu_samples=cpu_samples,
                ram_samples=ram_samples,
                errors_detected=errors,
                audio_glitches=0  # Would need audio monitoring
            )

        except Exception as e:
            print(f"Session error: {e}")
            return None

    def _analyze_session(self, metrics: SessionMetrics, thresholds) -> None:
        """Analyze session metrics and assert quality."""
        cpu_arr = np.array(metrics.cpu_samples)
        ram_arr = np.array(metrics.ram_samples)

        print(f"\nSession Summary ({metrics.duration_hours:.1f} hours):")
        print(f"  CPU - Avg: {np.mean(cpu_arr):.1f}%, Peak: {np.max(cpu_arr):.1f}%")
        print(f"  RAM - Start: {ram_arr[0]:.1f} MB, End: {ram_arr[-1]:.1f} MB")
        print(f"  RAM Growth: {ram_arr[-1] - ram_arr[0]:.1f} MB")
        print(f"  Errors: {metrics.errors_detected}")

        # Check for memory leak
        ram_growth = ram_arr[-1] - ram_arr[0]
        leak_threshold = thresholds.performance.get("ram_leak_threshold_mb", 10)

        assert ram_growth < leak_threshold, \
            f"Memory grew by {ram_growth:.1f} MB (threshold: {leak_threshold} MB)"

        # Check for errors
        assert metrics.errors_detected == 0, \
            f"Detected {metrics.errors_detected} errors during session"

        # Check CPU stability
        cpu_std = np.std(cpu_arr)
        if len(cpu_arr) > 10:
            assert cpu_std < 20, f"CPU usage unstable (std dev: {cpu_std:.1f}%)"


class TestSessionRecovery:
    """Test recovery after long sessions using DawDreamer."""

    @pytest.mark.stress
    def test_clean_shutdown_after_session(self, plugin_host, require_vst3, thresholds):
        """
        Verify clean shutdown after extended use via DawDreamer.

        Plugin should release cleanly without hanging or leaving resources.
        """
        if not HAS_PSUTIL:
            pytest.skip("psutil required for monitoring")

        from tests.tools.dawdreamer_host import MIDIEvent

        # Load plugin
        if not plugin_host.load_plugin(require_vst3):
            pytest.fail(f"Failed to load plugin: {require_vst3}")

        gc.collect()
        initial_ram = get_process_memory_mb()

        print(f"\nTesting clean shutdown after session")
        print(f"Initial RAM: {initial_ram:.1f} MB")

        # Run a session with multiple renders
        num_renders = 20
        for i in range(num_renders):
            events = [
                MIDIEvent.note_on(0.0, 60 + i % 12, 100),
                MIDIEvent.note_off(1.0, 60 + i % 12),
            ]
            result = plugin_host.render(2.0, events)

        ram_during = get_process_memory_mb()
        print(f"RAM during session: {ram_during:.1f} MB")

        # Unload plugin (simulating shutdown)
        start_unload = time.perf_counter()
        plugin_host.unload()
        unload_time = time.perf_counter() - start_unload

        print(f"Unload time: {unload_time*1000:.1f} ms")

        # Force garbage collection
        gc.collect()
        time.sleep(0.5)
        gc.collect()

        final_ram = get_process_memory_mb()
        print(f"RAM after unload: {final_ram:.1f} MB")

        # Verify unload was quick (should be under 1 second)
        assert unload_time < 1.0, f"Unload took too long: {unload_time:.2f}s"

        # Verify memory was mostly released
        ram_recovered = ram_during - final_ram
        print(f"RAM recovered: {ram_recovered:.1f} MB")

        # Should recover most of the memory used by the plugin
        # Allow some tolerance for Python/DawDreamer overhead
        if ram_during - initial_ram > 5:  # Only check if plugin used significant memory
            recovery_ratio = ram_recovered / (ram_during - initial_ram)
            print(f"Recovery ratio: {recovery_ratio*100:.1f}%")
            # Should recover at least 50% of plugin memory
            assert recovery_ratio > 0.5 or final_ram - initial_ram < 10, \
                f"Poor memory recovery: only {recovery_ratio*100:.1f}%"

    @pytest.mark.stress
    def test_memory_freed_after_session(self, dawdreamer_available, require_vst3, thresholds):
        """
        Verify memory is released after closing plugin.

        Tests that loading, using, and unloading the plugin
        multiple times does not accumulate memory.
        """
        if not dawdreamer_available:
            pytest.skip("DawDreamer not installed")

        if not HAS_PSUTIL:
            pytest.skip("psutil required")

        from tests.tools.dawdreamer_host import DawDreamerHost, MIDIEvent

        gc.collect()
        baseline_ram = get_process_memory_mb()

        print(f"\nTesting memory release after multiple sessions")
        print(f"Baseline RAM: {baseline_ram:.1f} MB")

        num_cycles = 5
        ram_after_cycles = []

        for cycle in range(num_cycles):
            # Create new host and load plugin
            host = DawDreamerHost(sample_rate=48000)
            if not host.load_plugin(require_vst3):
                pytest.fail(f"Failed to load plugin: {require_vst3}")

            # Do some work
            for i in range(10):
                events = [
                    MIDIEvent.note_on(0.0, 48 + i * 4, 100),
                    MIDIEvent.note_off(0.5, 48 + i * 4),
                ]
                host.render(1.0, events)

            # Unload and cleanup
            host.unload()
            del host

            # Force garbage collection
            gc.collect()
            time.sleep(0.2)
            gc.collect()

            current_ram = get_process_memory_mb()
            ram_after_cycles.append(current_ram)
            print(f"  Cycle {cycle + 1}: RAM = {current_ram:.1f} MB")

        # Check for memory accumulation across cycles
        final_ram = ram_after_cycles[-1]
        ram_growth = final_ram - baseline_ram

        print(f"\nMemory Summary:")
        print(f"  Baseline: {baseline_ram:.1f} MB")
        print(f"  After {num_cycles} cycles: {final_ram:.1f} MB")
        print(f"  Total growth: {ram_growth:.1f} MB")

        # Check for memory trend (is it growing steadily?)
        if len(ram_after_cycles) >= 3:
            # Linear regression to detect trend
            x = np.arange(len(ram_after_cycles))
            y = np.array(ram_after_cycles)
            slope = np.polyfit(x, y, 1)[0]
            print(f"  Memory trend: {slope:.2f} MB/cycle")

            # Warn if memory is steadily increasing
            if slope > 2.0:
                print(f"  WARNING: Memory increasing at {slope:.2f} MB/cycle")

        # Allow reasonable tolerance for Python/GC overhead
        leak_threshold = getattr(thresholds, 'ram_leak_threshold_mb', 10.0) * 2
        assert ram_growth < leak_threshold, \
            f"Memory leak of {ram_growth:.1f} MB after {num_cycles} session cycles (threshold: {leak_threshold} MB)"
