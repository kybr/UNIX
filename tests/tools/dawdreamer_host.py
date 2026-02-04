"""
DawDreamer Plugin Host

Real plugin hosting for testing VST3/AU plugins:
- Load and instantiate plugins
- Send MIDI events with precise timing
- Render audio offline
- Parameter automation
- CPU/memory profiling

DawDreamer: https://github.com/DBraun/DawDreamer
"""

import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

try:
    import dawdreamer as daw
    HAS_DAWDREAMER = True
except ImportError:
    HAS_DAWDREAMER = False


@dataclass
class MIDIEvent:
    """MIDI event with timing."""
    time_seconds: float
    status: int  # Note on = 0x90, Note off = 0x80, etc.
    data1: int   # Note number or CC number
    data2: int   # Velocity or CC value
    channel: int = 0

    @classmethod
    def note_on(cls, time: float, note: int, velocity: int = 100, channel: int = 0):
        return cls(time, 0x90, note, velocity, channel)

    @classmethod
    def note_off(cls, time: float, note: int, velocity: int = 0, channel: int = 0):
        return cls(time, 0x80, note, velocity, channel)

    @classmethod
    def cc(cls, time: float, controller: int, value: int, channel: int = 0):
        return cls(time, 0xB0, controller, value, channel)

    @classmethod
    def pitch_bend(cls, time: float, value: int, channel: int = 0):
        """Pitch bend value 0-16383, center = 8192."""
        return cls(time, 0xE0, value & 0x7F, (value >> 7) & 0x7F, channel)


@dataclass
class RenderResult:
    """Result of audio rendering."""
    audio: np.ndarray          # Shape: (channels, samples)
    sample_rate: int
    duration_seconds: float
    cpu_time_seconds: float    # Processing time
    realtime_ratio: float      # < 1.0 means faster than realtime
    plugin_latency_samples: int


@dataclass
class PluginInfo:
    """Information about loaded plugin."""
    name: str
    vendor: str
    version: str
    num_inputs: int
    num_outputs: int
    parameters: Dict[str, Tuple[int, float, float, float]]  # name -> (index, min, max, default)
    latency_samples: int
    is_synth: bool


class DawDreamerHost:
    """
    Plugin host using DawDreamer for testing.

    Provides high-level interface for:
    - Loading VST3/AU plugins
    - Rendering audio with MIDI input
    - Parameter automation
    - Performance measurement
    """

    def __init__(self, sample_rate: int = 48000, block_size: int = 512):
        if not HAS_DAWDREAMER:
            raise ImportError("DawDreamer not installed. Run: pip install dawdreamer")

        self.sample_rate = sample_rate
        self.block_size = block_size
        self.engine = daw.RenderEngine(sample_rate, block_size)
        self.plugin: Optional[daw.PluginProcessor] = None
        self.plugin_path: Optional[Path] = None

    def load_plugin(self, plugin_path: Path) -> bool:
        """
        Load a VST3 or AU plugin.

        Args:
            plugin_path: Path to .vst3 or .component

        Returns:
            True if loaded successfully
        """
        try:
            path_str = str(plugin_path)
            self.plugin = self.engine.make_plugin_processor("synth", path_str)
            self.plugin_path = plugin_path

            # Set as graph endpoint
            self.engine.load_graph([
                (self.plugin, [])
            ])

            return True

        except Exception as e:
            print(f"Failed to load plugin: {e}")
            self.plugin = None
            return False

    def get_info(self) -> Optional[PluginInfo]:
        """Get information about the loaded plugin."""
        if self.plugin is None:
            return None

        # Get parameters - try different DawDreamer API versions
        params = {}
        try:
            # Try newer API first
            param_count = self.plugin.get_plugin_parameter_size()
        except AttributeError:
            try:
                # Try alternative method name
                param_count = self.plugin.get_num_parameters()
            except AttributeError:
                # Fall back to probing - try parameters 0-100
                param_count = 0
                for i in range(100):
                    try:
                        self.plugin.get_parameter_name(i)
                        param_count = i + 1
                    except (IndexError, RuntimeError):
                        break

        for i in range(param_count):
            try:
                name = self.plugin.get_parameter_name(i)
                # DawDreamer uses normalized 0-1 values
                params[name] = (i, 0.0, 1.0, self.plugin.get_parameter(i))
            except (IndexError, RuntimeError):
                break

        return PluginInfo(
            name=self.plugin.get_name(),
            vendor="",  # Not directly available
            version="",
            num_inputs=0,  # Synth - no audio input
            num_outputs=2,  # Assume stereo
            parameters=params,
            latency_samples=self.plugin.get_latency_samples(),
            is_synth=True
        )

    def set_parameter(self, name_or_index, value: float) -> bool:
        """
        Set a parameter value.

        Args:
            name_or_index: Parameter name or index
            value: Normalized value 0.0-1.0

        Returns:
            True if successful
        """
        if self.plugin is None:
            return False

        try:
            if isinstance(name_or_index, str):
                self.plugin.set_parameter_by_name(name_or_index, value)
            else:
                self.plugin.set_parameter(name_or_index, value)
            return True
        except Exception as e:
            print(f"Failed to set parameter: {e}")
            return False

    def get_parameter(self, name_or_index) -> float:
        """Get a parameter value."""
        if self.plugin is None:
            return 0.0

        try:
            if isinstance(name_or_index, str):
                # Find index by name - probe parameters
                for i in range(100):  # Reasonable max
                    try:
                        if self.plugin.get_parameter_name(i) == name_or_index:
                            return self.plugin.get_parameter(i)
                    except (IndexError, RuntimeError):
                        break
                return 0.0
            else:
                return self.plugin.get_parameter(name_or_index)
        except Exception:
            return 0.0

    def render(
        self,
        duration_seconds: float,
        midi_events: Optional[List[MIDIEvent]] = None,
        bpm: float = 120.0
    ) -> RenderResult:
        """
        Render audio with MIDI input.

        Args:
            duration_seconds: Duration to render
            midi_events: List of MIDI events with timing
            bpm: Tempo in BPM

        Returns:
            RenderResult with audio and metrics
        """
        if self.plugin is None:
            raise RuntimeError("No plugin loaded")

        # Clear any previous MIDI
        self.plugin.clear_midi()

        # Process MIDI events - pair note-ons with note-offs
        if midi_events:
            # Collect note-on events by note number
            active_notes: Dict[int, MIDIEvent] = {}

            for event in midi_events:
                if event.status == 0x90 and event.data2 > 0:  # Note on with velocity > 0
                    active_notes[event.data1] = event
                elif event.status == 0x80 or (event.status == 0x90 and event.data2 == 0):  # Note off
                    note_num = event.data1
                    if note_num in active_notes:
                        note_on = active_notes[note_num]
                        note_duration = event.time_seconds - note_on.time_seconds
                        if note_duration <= 0:
                            note_duration = 0.001  # Minimum duration

                        # Add the complete note to DawDreamer
                        self.plugin.add_midi_note(
                            note_num,              # note
                            note_on.data2,         # velocity
                            note_on.time_seconds,  # start time in seconds
                            note_duration          # duration in seconds
                        )
                        del active_notes[note_num]
                elif event.status == 0xE0:  # Pitch bend
                    # DawDreamer uses add_pitch_bend(time, value) where value is 0-16383
                    # data1 = LSB, data2 = MSB
                    pitch_value = event.data1 | (event.data2 << 7)
                    try:
                        self.plugin.add_pitch_bend(event.time_seconds, pitch_value)
                    except AttributeError:
                        # DawDreamer may not support pitch bend directly
                        # Try using automation on pitchwheel parameter if available
                        pass

            # Handle any remaining active notes (no note-off found)
            for note_num, note_on in active_notes.items():
                remaining_duration = duration_seconds - note_on.time_seconds
                if remaining_duration > 0:
                    self.plugin.add_midi_note(
                        note_num,
                        note_on.data2,
                        note_on.time_seconds,
                        remaining_duration
                    )

        # Render
        start_time = time.perf_counter()
        self.engine.set_bpm(bpm)
        self.engine.render(duration_seconds)
        cpu_time = time.perf_counter() - start_time

        # Get audio
        audio = self.plugin.get_audio()

        return RenderResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration_seconds,
            cpu_time_seconds=cpu_time,
            realtime_ratio=cpu_time / duration_seconds,
            plugin_latency_samples=self.plugin.get_latency_samples()
        )

    def render_note(
        self,
        note: int,
        velocity: int = 100,
        duration_seconds: float = 1.0,
        tail_seconds: float = 2.0
    ) -> RenderResult:
        """
        Render a single note.

        Args:
            note: MIDI note number
            velocity: Note velocity
            duration_seconds: Note duration
            tail_seconds: Additional time for decay

        Returns:
            RenderResult with audio
        """
        total_duration = duration_seconds + tail_seconds

        events = [
            MIDIEvent.note_on(0.0, note, velocity),
            MIDIEvent.note_off(duration_seconds, note),
        ]

        return self.render(total_duration, events)

    def render_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_seconds: float = 1.0,
        tail_seconds: float = 2.0
    ) -> RenderResult:
        """
        Render a chord (multiple simultaneous notes).

        Args:
            notes: List of MIDI note numbers
            velocity: Note velocity
            duration_seconds: Chord duration
            tail_seconds: Additional time for decay

        Returns:
            RenderResult with audio
        """
        total_duration = duration_seconds + tail_seconds

        events = []
        for note in notes:
            events.append(MIDIEvent.note_on(0.0, note, velocity))
        for note in notes:
            events.append(MIDIEvent.note_off(duration_seconds, note))

        return self.render(total_duration, events)

    def render_polyphony_test(
        self,
        num_voices: int,
        base_note: int = 48,
        velocity: int = 100,
        stagger_ms: float = 10.0,
        hold_seconds: float = 2.0
    ) -> RenderResult:
        """
        Render a polyphony stress test.

        Args:
            num_voices: Number of simultaneous voices
            base_note: Starting MIDI note
            velocity: Note velocity
            stagger_ms: Time between note onsets
            hold_seconds: How long to hold all notes

        Returns:
            RenderResult with audio
        """
        events = []

        # Note ons with stagger
        for i in range(num_voices):
            time = i * (stagger_ms / 1000.0)
            note = base_note + i
            events.append(MIDIEvent.note_on(time, note, velocity))

        # All note offs after hold
        note_off_time = num_voices * (stagger_ms / 1000.0) + hold_seconds
        for i in range(num_voices):
            note = base_note + i
            events.append(MIDIEvent.note_off(note_off_time, note))

        total_duration = note_off_time + 2.0  # Extra decay time

        return self.render(total_duration, events)

    def measure_cpu_per_voice(
        self,
        max_voices: int = 32,
        note: int = 60,
        duration: float = 1.0
    ) -> List[Tuple[int, float]]:
        """
        Measure CPU usage at different polyphony levels.

        Returns:
            List of (num_voices, cpu_percent) tuples
        """
        results = []

        for num_voices in [1, 2, 4, 8, 16, max_voices]:
            if num_voices > max_voices:
                break

            # Create chord
            notes = [note + i for i in range(num_voices)]
            result = self.render_chord(notes, duration_seconds=duration, tail_seconds=0.5)

            # CPU as percentage of realtime
            cpu_percent = result.realtime_ratio * 100

            results.append((num_voices, cpu_percent))

        return results

    def unload(self):
        """Unload the current plugin."""
        self.plugin = None
        self.plugin_path = None
        # Create fresh engine
        self.engine = daw.RenderEngine(self.sample_rate, self.block_size)


def create_test_host(sample_rate: int = 48000) -> DawDreamerHost:
    """Factory function to create a test host."""
    return DawDreamerHost(sample_rate=sample_rate)
