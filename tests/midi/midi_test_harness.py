"""
MIDI Test Harness

Provides utilities for testing MIDI functionality:
- MIDI message generation
- Real-time MIDI I/O
- MIDI file creation and parsing
- Timing measurement
"""

import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

try:
    import rtmidi
    HAS_RTMIDI = True
except ImportError:
    HAS_RTMIDI = False


class MIDIStatus(IntEnum):
    """MIDI status bytes."""
    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    POLY_AFTERTOUCH = 0xA0
    CONTROL_CHANGE = 0xB0
    PROGRAM_CHANGE = 0xC0
    CHANNEL_AFTERTOUCH = 0xD0
    PITCH_BEND = 0xE0


@dataclass
class MIDINote:
    """Represents a MIDI note event."""
    note: int
    velocity: int
    channel: int = 0
    timestamp: float = 0.0


@dataclass
class MIDITestResult:
    """Result of a MIDI test."""
    passed: bool
    latency_ms: float
    notes_triggered: int
    notes_missed: int
    message: str


class MIDIGenerator:
    """Generate MIDI messages and files for testing."""

    @staticmethod
    def note_on(note: int, velocity: int = 100, channel: int = 0) -> bytes:
        """Create a MIDI note-on message."""
        return bytes([MIDIStatus.NOTE_ON | (channel & 0x0F), note & 0x7F, velocity & 0x7F])

    @staticmethod
    def note_off(note: int, velocity: int = 0, channel: int = 0) -> bytes:
        """Create a MIDI note-off message."""
        return bytes([MIDIStatus.NOTE_OFF | (channel & 0x0F), note & 0x7F, velocity & 0x7F])

    @staticmethod
    def control_change(controller: int, value: int, channel: int = 0) -> bytes:
        """Create a MIDI control change message."""
        return bytes([MIDIStatus.CONTROL_CHANGE | (channel & 0x0F), controller & 0x7F, value & 0x7F])

    @staticmethod
    def pitch_bend(value: int, channel: int = 0) -> bytes:
        """
        Create a MIDI pitch bend message.

        Args:
            value: Pitch bend value (0-16383, center = 8192)
            channel: MIDI channel (0-15)
        """
        lsb = value & 0x7F
        msb = (value >> 7) & 0x7F
        return bytes([MIDIStatus.PITCH_BEND | (channel & 0x0F), lsb, msb])

    @staticmethod
    def aftertouch(pressure: int, channel: int = 0) -> bytes:
        """Create a MIDI channel aftertouch message."""
        return bytes([MIDIStatus.CHANNEL_AFTERTOUCH | (channel & 0x0F), pressure & 0x7F])

    @staticmethod
    def poly_aftertouch(note: int, pressure: int, channel: int = 0) -> bytes:
        """Create a MIDI polyphonic aftertouch message."""
        return bytes([MIDIStatus.POLY_AFTERTOUCH | (channel & 0x0F), note & 0x7F, pressure & 0x7F])

    def create_note_sequence(
        self,
        notes: List[int],
        velocities: Optional[List[int]] = None,
        duration_ms: float = 500,
        gap_ms: float = 100
    ) -> List[Tuple[float, bytes]]:
        """
        Create a sequence of MIDI notes with timing.

        Returns:
            List of (timestamp_ms, midi_bytes) tuples
        """
        if velocities is None:
            velocities = [100] * len(notes)

        events = []
        current_time = 0.0

        for note, vel in zip(notes, velocities):
            # Note on
            events.append((current_time, self.note_on(note, vel)))
            # Note off
            events.append((current_time + duration_ms, self.note_off(note)))
            current_time += duration_ms + gap_ms

        return events

    def create_chord(
        self,
        notes: List[int],
        velocity: int = 100,
        duration_ms: float = 1000
    ) -> List[Tuple[float, bytes]]:
        """
        Create a chord (simultaneous notes).

        Returns:
            List of (timestamp_ms, midi_bytes) tuples
        """
        events = []

        # All notes on at time 0
        for note in notes:
            events.append((0.0, self.note_on(note, velocity)))

        # All notes off at duration
        for note in notes:
            events.append((duration_ms, self.note_off(note)))

        return events

    def create_polyphony_test(
        self,
        num_voices: int,
        base_note: int = 48,
        velocity: int = 100,
        stagger_ms: float = 10
    ) -> List[Tuple[float, bytes]]:
        """
        Create a test for polyphony limit.

        Triggers num_voices notes with slight stagger.
        """
        events = []

        for i in range(num_voices):
            time = i * stagger_ms
            note = base_note + i
            events.append((time, self.note_on(note, velocity)))

        return events

    def create_velocity_sweep(
        self,
        note: int = 60,
        num_steps: int = 8,
        duration_ms: float = 500,
        gap_ms: float = 200
    ) -> List[Tuple[float, bytes]]:
        """
        Create a velocity sweep from pianissimo to fortissimo.
        """
        velocities = np.linspace(20, 127, num_steps).astype(int)
        events = []
        current_time = 0.0

        for vel in velocities:
            events.append((current_time, self.note_on(note, vel)))
            events.append((current_time + duration_ms, self.note_off(note)))
            current_time += duration_ms + gap_ms

        return events


class MIDIFileCreator:
    """Create MIDI files for testing."""

    def __init__(self):
        if not HAS_MIDO:
            raise ImportError("mido package required for MIDI file creation")

    def create_test_file(
        self,
        events: List[Tuple[float, bytes]],
        output_path: str,
        tempo: int = 500000  # microseconds per beat (120 BPM)
    ) -> None:
        """
        Create a MIDI file from events.

        Args:
            events: List of (timestamp_ms, midi_bytes) tuples
            output_path: Path to save MIDI file
            tempo: Tempo in microseconds per beat
        """
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))

        # Sort events by time
        events = sorted(events, key=lambda x: x[0])

        # Convert to mido messages with delta times
        last_time = 0

        for time_ms, midi_bytes in events:
            delta = int((time_ms - last_time) * (mid.ticks_per_beat / 1000))
            last_time = time_ms

            # Parse MIDI bytes
            status = midi_bytes[0]
            msg_type = status & 0xF0

            if msg_type == MIDIStatus.NOTE_ON:
                msg = mido.Message('note_on',
                    note=midi_bytes[1],
                    velocity=midi_bytes[2],
                    channel=status & 0x0F,
                    time=delta
                )
            elif msg_type == MIDIStatus.NOTE_OFF:
                msg = mido.Message('note_off',
                    note=midi_bytes[1],
                    velocity=midi_bytes[2],
                    channel=status & 0x0F,
                    time=delta
                )
            else:
                continue  # Skip other message types for now

            track.append(msg)

        mid.save(output_path)


class MIDILatencyTester:
    """
    Measure MIDI latency.

    Based on ALSA MIDI latency test methodology.
    """

    def __init__(self):
        if not HAS_RTMIDI:
            raise ImportError("rtmidi package required for latency testing")

        self.midi_out = rtmidi.MidiOut()
        self.midi_in = rtmidi.MidiIn()

    def setup(self, output_port: int = 0, input_port: int = 0):
        """Open MIDI ports."""
        out_ports = self.midi_out.get_ports()
        in_ports = self.midi_in.get_ports()

        if output_port < len(out_ports):
            self.midi_out.open_port(output_port)
        else:
            self.midi_out.open_virtual_port("Test Output")

        if input_port < len(in_ports):
            self.midi_in.open_port(input_port)
        else:
            self.midi_in.open_virtual_port("Test Input")

    def measure_roundtrip(self, num_tests: int = 100) -> List[float]:
        """
        Measure MIDI round-trip latency.

        Returns:
            List of latency measurements in milliseconds
        """
        generator = MIDIGenerator()
        latencies = []
        received = []

        def callback(message, data):
            received.append((time.perf_counter(), message))

        self.midi_in.set_callback(callback)

        for _ in range(num_tests):
            received.clear()
            note_on = generator.note_on(60, 100)

            start = time.perf_counter()
            self.midi_out.send_message(list(note_on))

            # Wait for response
            timeout = time.perf_counter() + 0.1
            while not received and time.perf_counter() < timeout:
                time.sleep(0.001)

            if received:
                end = received[0][0]
                latency = (end - start) * 1000  # Convert to ms
                latencies.append(latency)

            # Send note off
            self.midi_out.send_message(list(generator.note_off(60)))
            time.sleep(0.05)

        self.midi_in.cancel_callback()
        return latencies

    def cleanup(self):
        """Close MIDI ports."""
        self.midi_out.close_port()
        self.midi_in.close_port()
