# UNIX Audio Plugin - Test Suite

Comprehensive testing infrastructure for the Karplus-Strong synthesizer plugin.

## Quick Start

```bash
# Install dependencies
pip install -r tests/requirements.txt

# Run all tests
make test

# Run quick smoke tests
make test-quick
```

## Test Categories

| Category | Command | Description |
|----------|---------|-------------|
| All (non-slow) | `make test` | Default test suite |
| Quick | `make test-quick` | Fast smoke tests |
| Validation | `make test-validation` | Pluginval/CLAP validator |
| Audio | `make test-audio` | Audio quality (THD, SNR) |
| MIDI | `make test-midi` | MIDI functionality |
| Performance | `make test-performance` | CPU/RAM benchmarks |
| GUI | `make test-gui` | UI automation |
| Stress | `make test-stress` | Long-running stress tests |

## Directory Structure

```
tests/
├── conftest.py          # Pytest fixtures
├── pytest.ini           # Pytest configuration
├── requirements.txt     # Python dependencies
├── thresholds.json      # Pass/fail thresholds
│
├── validation/          # Plugin validation
│   ├── test_pluginval.py
│   └── test_clap_validator.py
│
├── audio_quality/       # Audio measurements
│   ├── analyzers/
│   │   └── audio_analyzer.py
│   ├── test_thd.py
│   ├── test_snr.py
│   └── test_frequency_response.py
│
├── midi/                # MIDI functionality
│   ├── midi_test_harness.py
│   ├── test_note_triggering.py
│   ├── test_polyphony.py
│   └── test_pitch_bend.py
│
├── performance/         # Performance profiling
│   ├── test_cpu_usage.py
│   └── test_ram_usage.py
│
├── gui/                 # GUI automation
│   ├── helpers/
│   │   └── gui_automation.py
│   ├── test_standalone_launch.py
│   └── test_plugin_controls.py
│
├── stress/              # Stress tests
│   ├── test_extreme_polyphony.py
│   ├── test_long_session.py
│   └── test_rapid_params.py
│
└── tools/               # Utilities
    ├── download_validators.py
    └── plugin_host.py
```

## Requirements

### Required for All Tests
- Python 3.10+
- pytest

### For Audio Tests
- numpy, scipy, soundfile, librosa

### For GUI Tests
- pyautogui, Pillow, opencv-python
- Display (X11/Wayland on Linux)

### For Validation Tests
- pluginval ([download](https://github.com/Tracktion/pluginval/releases))
- clap-validator ([download](https://github.com/free-audio/clap-validator/releases))

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/validation -v

# Skip slow tests
pytest tests/ -m "not slow"

# Run with specific markers
pytest tests/ -m "audio"
pytest tests/ -m "midi and not slow"
```

### Using Makefile

```bash
# Show all targets
make help

# Install dependencies
make install-deps

# Download validation tools
make download-tools

# Run pluginval directly
make pluginval

# Run strict validation
make pluginval-strict
```

### CI/CD

The GitHub Actions workflow runs automatically on push:

```yaml
# .github/workflows/plugin_tests.yml
- Build on all platforms
- Run pluginval validation
- Run Python test suite
- Upload benchmark results
```

## Writing Tests

### Using Fixtures

```python
def test_my_feature(require_vst3, thresholds):
    """Test with VST3 plugin."""
    # require_vst3 skips if plugin not built
    assert thresholds.thd_db < -60
```

### Markers

```python
@pytest.mark.slow          # Long-running test
@pytest.mark.gui           # Requires display
@pytest.mark.requires_plugin  # Needs built plugin
@pytest.mark.stress        # Stress/endurance test
@pytest.mark.audio         # Audio quality test
@pytest.mark.midi          # MIDI functionality test
```

### Thresholds

Edit `thresholds.json` to adjust pass/fail criteria:

```json
{
  "audio_quality": {
    "thd_db": -60,
    "snr_db": 90
  },
  "performance": {
    "cpu_single_voice_percent": 1.0,
    "ram_max_mb": 200
  }
}
```

## Reports

Test results are saved to `reports/`:

```bash
# Generate HTML report
make report

# View benchmark results
cat reports/benchmark.json
```

## Troubleshooting

### "Plugin not built"
Build the plugin first:
```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build
```

### "pluginval not found"
Download pluginval:
```bash
make download-tools
```

### "No display available"
On Linux without display:
```bash
xvfb-run pytest tests/gui -v
```

Or skip GUI tests:
```bash
pytest tests/ -m "not gui"
```
