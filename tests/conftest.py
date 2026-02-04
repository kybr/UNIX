"""
UNIX Audio Plugin - Test Suite Configuration

Provides shared fixtures for all test modules:
- Plugin paths (VST3, AU, CLAP, Standalone)
- Build directory locations
- Audio/MIDI test utilities
- Performance monitoring helpers
- Platform detection
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import platform
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

import pytest
import numpy as np

# =============================================================================
# Path Configuration
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_build_dir() -> Path:
    """Get the build directory, checking common locations."""
    root = get_project_root()
    for name in ["build", "cmake-build-release", "cmake-build-debug", "out/build"]:
        build = root / name
        if build.exists():
            return build
    return root / "build"


def get_plugin_dir() -> Path:
    """Get the KarplusStrongPlugin artefacts directory."""
    build = get_build_dir()
    # Check for nested build directory first (cmake -B build/KarplusStrongPlugin)
    nested = build / "KarplusStrongPlugin" / "KarplusStrongPlugin_artefacts"
    if nested.exists():
        return nested
    # Fall back to direct location (cmake -B build with parent CMakeLists)
    return build / "KarplusStrongPlugin_artefacts"


def get_exe_extension() -> str:
    """Get platform-specific executable extension."""
    return ".exe" if platform.system() == "Windows" else ""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PluginPaths:
    """Container for all plugin format paths."""
    vst3: Optional[Path] = None
    au: Optional[Path] = None
    clap: Optional[Path] = None
    standalone: Optional[Path] = None

    def any_exists(self) -> bool:
        """Check if any plugin format exists."""
        return any([
            self.vst3 and self.vst3.exists(),
            self.au and self.au.exists(),
            self.clap and self.clap.exists(),
            self.standalone and self.standalone.exists(),
        ])


@dataclass
class ThresholdConfig:
    """Pass/fail thresholds for tests."""
    # Audio Quality
    thd_db: float = -60.0
    snr_db: float = 90.0
    frequency_accuracy_cents: float = 50.0
    dc_offset_threshold: float = 0.03  # Real synths often have small DC offset
    aliasing_threshold_db: float = -80.0

    # Performance
    cpu_idle_percent: float = 10.0
    cpu_single_voice_percent: float = 1.0
    cpu_8_voices_percent: float = 5.0
    cpu_16_voices_percent: float = 10.0
    cpu_32_voices_percent: float = 20.0
    ram_mb: float = 100.0
    ram_baseline_mb: float = 50.0
    ram_max_mb: float = 200.0
    ram_leak_threshold_mb: float = 10.0
    realtime_ratio: float = 0.5
    latency_ms: float = 10.0

    # MIDI
    note_latency_ms: float = 5.0
    note_on_latency_ms: float = 5.0
    note_off_latency_ms: float = 5.0
    min_polyphony: int = 32
    velocity_response_tolerance: float = 0.05

    # Validation
    pluginval_level: int = 5
    pluginval_min_level: int = 5
    pluginval_target_level: int = 10

    # GUI
    launch_timeout_seconds: float = 10.0
    screenshot_diff_threshold: float = 0.02
    control_response_ms: float = 100.0


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Project root directory."""
    return get_project_root()


@pytest.fixture(scope="session")
def build_dir() -> Path:
    """Build output directory."""
    return get_build_dir()


@pytest.fixture(scope="session")
def reports_dir(project_root) -> Path:
    """Test reports directory."""
    path = project_root / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def report_writer(reports_dir):
    """
    Session-scoped report writer for persisting test results to JSON.

    Usage in tests:
        def test_something(report_writer):
            # ... run test ...
            report_writer.add_audio_result(AudioAnalysisResult(
                test_name="test_something",
                thd_db=-65.0,
                passed=True
            ))
    """
    from tests.tools.report_writer import ReportWriter
    return ReportWriter(reports_dir)


@pytest.fixture(scope="session", autouse=True)
def finalize_reports(request, report_writer):
    """
    Automatically finalize and write all reports at session end.
    This fixture runs automatically due to autouse=True.
    """
    yield  # Let all tests run

    # Gather session info
    session_info = {
        'total_collected': request.session.testscollected if hasattr(request.session, 'testscollected') else 0,
        'exit_status': 'completed',
    }

    # Write all accumulated reports
    paths = report_writer.finalize(session_info)

    # Log the paths
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Test reports written to: {list(paths.values())}")


@pytest.fixture(scope="session")
def thresholds() -> ThresholdConfig:
    """Load test thresholds from config or use defaults."""
    config_path = get_project_root() / "tests" / "thresholds.json"
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
            # Flatten nested structure
            flat_data = {}
            for section, values in data.items():
                if isinstance(values, dict):
                    flat_data.update(values)
                else:
                    flat_data[section] = values
            # Only use keys that ThresholdConfig accepts
            valid_keys = ThresholdConfig.__dataclass_fields__.keys()
            filtered_data = {k: v for k, v in flat_data.items() if k in valid_keys}
            return ThresholdConfig(**filtered_data)
    return ThresholdConfig()


@pytest.fixture(scope="session")
def plugin_paths() -> PluginPaths:
    """
    Locate all plugin format paths.

    Searches common build output locations for each format.
    """
    plugin_dir = get_plugin_dir()
    paths = PluginPaths()
    system = platform.system()

    # Search in Release and Debug configurations
    for config in ["Release", "Debug", ""]:
        config_dir = plugin_dir / config if config else plugin_dir

        # VST3
        if paths.vst3 is None:
            vst3_candidates = [
                config_dir / "VST3" / "Karplus-Strong Synth.vst3",
                config_dir / "VST3" / "KarplusStrongPlugin.vst3",
            ]
            for vst3 in vst3_candidates:
                if vst3.exists():
                    paths.vst3 = vst3
                    break

        # AU (macOS only)
        if system == "Darwin" and paths.au is None:
            au_candidates = [
                config_dir / "AU" / "Karplus-Strong Synth.component",
                config_dir / "AU" / "KarplusStrongPlugin.component",
            ]
            for au in au_candidates:
                if au.exists():
                    paths.au = au
                    break

        # CLAP
        if paths.clap is None:
            clap_candidates = [
                config_dir / "CLAP" / "Karplus-Strong Synth.clap",
                config_dir / "CLAP" / "KarplusStrongPlugin.clap",
            ]
            for clap in clap_candidates:
                if clap.exists():
                    paths.clap = clap
                    break

        # Standalone
        if paths.standalone is None:
            ext = get_exe_extension()
            standalone_candidates = [
                config_dir / "Standalone" / f"Karplus-Strong Synth{ext}",
                config_dir / "Standalone" / f"KarplusStrongPlugin{ext}",
            ]
            if system == "Darwin":
                standalone_candidates.extend([
                    config_dir / "Standalone" / "Karplus-Strong Synth.app",
                    config_dir / "Standalone" / "KarplusStrongPlugin.app",
                ])
            for standalone in standalone_candidates:
                if standalone.exists():
                    paths.standalone = standalone
                    break

    return paths


@pytest.fixture(scope="session")
def cli_tools(build_dir) -> Dict[str, Path]:
    """Dictionary of CLI tool paths."""
    ext = get_exe_extension()
    tools = {}
    tool_names = [
        "karplus-strong",
        "wav-write",
        "wav-read",
        "sine-sweep",
        "phasor",
    ]

    for name in tool_names:
        for subdir in ["", "Debug", "Release"]:
            path = build_dir / subdir / f"{name}{ext}"
            if path.exists():
                tools[name] = path
                break
        else:
            tools[name] = build_dir / f"{name}{ext}"

    return tools


@pytest.fixture(scope="session")
def sample_rate() -> int:
    """Default sample rate for tests."""
    return 48000


@pytest.fixture(scope="session")
def block_size() -> int:
    """Default block size for tests."""
    return 512


# =============================================================================
# Function-Scoped Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def temp_wav(temp_dir) -> Path:
    """Path to a temporary WAV file."""
    return temp_dir / "test.wav"


@pytest.fixture
def temp_midi(temp_dir) -> Path:
    """Path to a temporary MIDI file."""
    return temp_dir / "test.mid"


# =============================================================================
# Plugin Requirement Fixtures
# =============================================================================

@pytest.fixture
def require_vst3(plugin_paths):
    """Skip if VST3 plugin not available."""
    if plugin_paths.vst3 is None or not plugin_paths.vst3.exists():
        pytest.skip("VST3 plugin not built")
    return plugin_paths.vst3


@pytest.fixture
def require_au(plugin_paths):
    """Skip if AU plugin not available."""
    if platform.system() != "Darwin":
        pytest.skip("AU only available on macOS")
    if plugin_paths.au is None or not plugin_paths.au.exists():
        pytest.skip("AU plugin not built")
    return plugin_paths.au


@pytest.fixture
def require_clap(plugin_paths):
    """Skip if CLAP plugin not available."""
    if plugin_paths.clap is None or not plugin_paths.clap.exists():
        pytest.skip("CLAP plugin not built")
    return plugin_paths.clap


@pytest.fixture
def require_standalone(plugin_paths):
    """Skip if standalone not available."""
    if plugin_paths.standalone is None or not plugin_paths.standalone.exists():
        pytest.skip("Standalone app not built")
    return plugin_paths.standalone


@pytest.fixture
def require_any_plugin(plugin_paths):
    """Skip if no plugin format is available."""
    if not plugin_paths.any_exists():
        pytest.skip("No plugin formats built")
    return plugin_paths


# =============================================================================
# Tool Requirement Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def pluginval_path() -> Optional[Path]:
    """Find pluginval executable."""
    ext = ".exe" if platform.system() == "Windows" else ""

    # Check common locations
    candidates = [
        get_project_root() / "tools" / f"pluginval{ext}",
        Path.home() / "bin" / f"pluginval{ext}",
        Path("/usr/local/bin/pluginval"),
        Path(f"C:/Program Files/pluginval/pluginval{ext}"),
    ]

    for path in candidates:
        if path.exists():
            return path

    # Try to find in PATH
    result = shutil.which("pluginval")
    if result:
        return Path(result)

    return None


@pytest.fixture
def require_pluginval(pluginval_path):
    """Skip if pluginval not available."""
    if pluginval_path is None:
        pytest.skip("pluginval not installed. Download from: https://github.com/Tracktion/pluginval/releases")
    return pluginval_path


@pytest.fixture(scope="session")
def clap_validator_path() -> Optional[Path]:
    """Find clap-validator executable."""
    candidates = [
        Path.home() / "bin" / "clap-validator",
        Path("/usr/local/bin/clap-validator"),
        get_project_root() / "tools" / "clap-validator",
    ]

    for path in candidates:
        if path.exists():
            return path

    result = shutil.which("clap-validator")
    if result:
        return Path(result)

    return None


@pytest.fixture
def require_clap_validator(clap_validator_path):
    """Skip if clap-validator not available."""
    if clap_validator_path is None:
        pytest.skip("clap-validator not installed. Download from: https://github.com/free-audio/clap-validator/releases")
    return clap_validator_path


# =============================================================================
# Display & GUI Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def has_display() -> bool:
    """Check if a display is available for GUI tests."""
    if platform.system() == "Windows":
        return True

    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    return display is not None


@pytest.fixture
def require_display(has_display):
    """Skip if no display available."""
    if not has_display:
        pytest.skip("No display available for GUI tests")
    return True


# =============================================================================
# Platform Detection Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def is_windows() -> bool:
    return platform.system() == "Windows"


@pytest.fixture(scope="session")
def is_macos() -> bool:
    return platform.system() == "Darwin"


@pytest.fixture(scope="session")
def is_linux() -> bool:
    return platform.system() == "Linux"


@pytest.fixture(scope="session")
def is_ci() -> bool:
    """Check if running in CI environment."""
    return os.environ.get("CI", "").lower() == "true"


# =============================================================================
# Audio Utility Fixtures
# =============================================================================

@pytest.fixture
def generate_sine(sample_rate):
    """Factory to generate sine wave test signals."""
    def _generate(frequency: float, duration: float, amplitude: float = 1.0) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return amplitude * np.sin(2 * np.pi * frequency * t)
    return _generate


@pytest.fixture
def generate_noise(sample_rate):
    """Factory to generate noise test signals."""
    def _generate(duration: float, amplitude: float = 1.0) -> np.ndarray:
        num_samples = int(sample_rate * duration)
        return amplitude * np.random.uniform(-1, 1, num_samples)
    return _generate


@pytest.fixture
def generate_impulse(sample_rate):
    """Factory to generate impulse test signals."""
    def _generate(duration: float, amplitude: float = 1.0) -> np.ndarray:
        num_samples = int(sample_rate * duration)
        signal = np.zeros(num_samples)
        signal[0] = amplitude
        return signal
    return _generate


# =============================================================================
# Plugin Host Fixtures (DawDreamer)
# =============================================================================

@pytest.fixture(scope="session")
def dawdreamer_available() -> bool:
    """Check if DawDreamer is available."""
    try:
        import dawdreamer
        return True
    except ImportError:
        return False


@pytest.fixture
def plugin_host(dawdreamer_available, sample_rate):
    """
    Create a DawDreamer plugin host for testing.

    Yields the host and cleans up after test.
    """
    if not dawdreamer_available:
        pytest.skip("DawDreamer not installed. Run: pip install dawdreamer")

    from tests.tools.dawdreamer_host import DawDreamerHost
    host = DawDreamerHost(sample_rate=sample_rate)
    yield host
    host.unload()


@pytest.fixture
def loaded_plugin(plugin_host, require_vst3):
    """
    Plugin host with VST3 already loaded.

    Skips if plugin not available or fails to load.
    """
    if not plugin_host.load_plugin(require_vst3):
        pytest.skip(f"Failed to load plugin: {require_vst3}")
    return plugin_host


# =============================================================================
# MIDI Utility Fixtures
# =============================================================================

@pytest.fixture
def midi_note_on():
    """Factory to create MIDI note-on messages."""
    def _create(note: int, velocity: int = 100, channel: int = 0) -> bytes:
        return bytes([0x90 | channel, note & 0x7F, velocity & 0x7F])
    return _create


@pytest.fixture
def midi_note_off():
    """Factory to create MIDI note-off messages."""
    def _create(note: int, velocity: int = 0, channel: int = 0) -> bytes:
        return bytes([0x80 | channel, note & 0x7F, velocity & 0x7F])
    return _create


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest."""
    # Create reports directory
    reports = get_project_root() / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for subdir in ["audio_analysis", "performance", "screenshots", "validation"]:
        (reports / subdir).mkdir(exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    # Skip GUI tests if no display
    if platform.system() != "Windows":
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not display:
            skip_gui = pytest.mark.skip(reason="No display available")
            for item in items:
                if "gui" in item.keywords or "requires_display" in item.keywords:
                    item.add_marker(skip_gui)

    # Skip slow tests unless explicitly requested
    if not config.getoption("--run-slow", default=False):
        skip_slow = pytest.mark.skip(reason="Use --run-slow to run slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--plugin-path",
        action="store",
        default=None,
        help="Override plugin path"
    )
    parser.addoption(
        "--strictness-level",
        action="store",
        default=5,
        type=int,
        help="Pluginval strictness level (1-10)"
    )


# =============================================================================
# Skip Markers for Import
# =============================================================================

skip_on_ci = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Skipped in CI environment"
)

windows_only = pytest.mark.skipif(
    platform.system() != "Windows",
    reason="Windows-only test"
)

macos_only = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="macOS-only test"
)

linux_only = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="Linux-only test"
)
