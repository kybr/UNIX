"""
Minimal Plugin Host for Testing

A simple Python-based plugin host for testing purposes.
Uses the CLAP or VST3 C API via ctypes.

Note: This is a simplified implementation for testing.
For production use, consider using existing frameworks.
"""

import ctypes
import platform
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    name: str
    vendor: str
    version: str
    format: str  # "VST3", "AU", "CLAP"
    num_inputs: int
    num_outputs: int
    num_params: int


@dataclass
class AudioBuffer:
    """Audio buffer for processing."""
    data: ctypes.Array
    num_channels: int
    num_samples: int


class PluginHost(ABC):
    """Abstract base class for plugin hosting."""

    @abstractmethod
    def load(self, path: Path) -> bool:
        """Load a plugin from path."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the current plugin."""
        pass

    @abstractmethod
    def get_info(self) -> Optional[PluginInfo]:
        """Get information about loaded plugin."""
        pass

    @abstractmethod
    def process(self, input_buffer: AudioBuffer, output_buffer: AudioBuffer) -> bool:
        """Process audio through the plugin."""
        pass

    @abstractmethod
    def set_parameter(self, index: int, value: float) -> bool:
        """Set a parameter value."""
        pass

    @abstractmethod
    def get_parameter(self, index: int) -> float:
        """Get a parameter value."""
        pass

    @abstractmethod
    def send_midi(self, data: bytes) -> bool:
        """Send MIDI data to plugin."""
        pass


class MockPluginHost(PluginHost):
    """
    Mock plugin host for testing.

    Simulates plugin behavior without actually loading plugins.
    Useful for testing the test framework itself.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.loaded = False
        self._params = {}
        self._midi_queue = []

    def load(self, path: Path) -> bool:
        """Simulate loading a plugin."""
        if path.exists() or path.suffix in [".vst3", ".component", ".clap"]:
            self.loaded = True
            self._params = {i: 0.5 for i in range(10)}
            return True
        return False

    def unload(self) -> None:
        """Simulate unloading."""
        self.loaded = False
        self._params = {}
        self._midi_queue = []

    def get_info(self) -> Optional[PluginInfo]:
        """Return mock plugin info."""
        if not self.loaded:
            return None

        return PluginInfo(
            name="Mock Plugin",
            vendor="Test",
            version="1.0.0",
            format="Mock",
            num_inputs=2,
            num_outputs=2,
            num_params=len(self._params)
        )

    def process(self, input_buffer: AudioBuffer, output_buffer: AudioBuffer) -> bool:
        """Pass through audio (mock processing)."""
        if not self.loaded:
            return False

        # Simple passthrough with gain from param 0
        gain = self._params.get(0, 1.0)

        for ch in range(min(input_buffer.num_channels, output_buffer.num_channels)):
            for i in range(min(input_buffer.num_samples, output_buffer.num_samples)):
                output_buffer.data[ch * output_buffer.num_samples + i] = \
                    input_buffer.data[ch * input_buffer.num_samples + i] * gain

        return True

    def set_parameter(self, index: int, value: float) -> bool:
        """Set mock parameter."""
        if not self.loaded or index < 0 or index >= len(self._params):
            return False
        self._params[index] = max(0.0, min(1.0, value))
        return True

    def get_parameter(self, index: int) -> float:
        """Get mock parameter."""
        return self._params.get(index, 0.0)

    def send_midi(self, data: bytes) -> bool:
        """Queue MIDI for mock processing."""
        if not self.loaded:
            return False
        self._midi_queue.append(data)
        return True


def create_audio_buffer(num_channels: int, num_samples: int) -> AudioBuffer:
    """Create an audio buffer."""
    total_samples = num_channels * num_samples
    ArrayType = ctypes.c_float * total_samples
    return AudioBuffer(
        data=ArrayType(),
        num_channels=num_channels,
        num_samples=num_samples
    )


def get_platform_plugin_extension() -> str:
    """Get the typical plugin extension for this platform."""
    system = platform.system()
    if system == "Windows":
        return ".vst3"
    elif system == "Darwin":
        return ".component"  # AU preferred on macOS
    else:
        return ".vst3"


# Note: A full implementation would include:
# - VST3 hosting via the VST3 SDK
# - AU hosting via AudioToolbox (macOS)
# - CLAP hosting via the CLAP API
#
# This is beyond the scope of simple Python testing.
# For real plugin hosting, consider:
# - JUCE AudioPluginHost
# - Carla
# - pedalboard (Spotify's Python plugin host)
