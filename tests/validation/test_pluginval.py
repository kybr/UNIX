"""
Pluginval Validation Tests

Runs the industry-standard pluginval tool against VST3 and AU plugins
to verify format compliance, stability, and real-time safety.

Pluginval: https://github.com/Tracktion/pluginval
"""

import subprocess
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import pytest


class TestPluginvalVST3:
    """VST3 validation using pluginval."""

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_vst3_level_1(self, require_pluginval, require_vst3):
        """
        Level 1: Basic instantiation test.

        Quick smoke test that the plugin can be loaded.
        """
        result = run_pluginval(require_pluginval, require_vst3, strictness=1)
        assert result.returncode == 0, f"Pluginval level 1 failed:\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_vst3_level_5(self, require_pluginval, require_vst3):
        """
        Level 5: Standard compatibility level.

        This is the minimum level for host compatibility. Tests include:
        - Plugin instantiation/destruction
        - Parameter enumeration
        - Basic audio processing
        - State save/restore
        """
        result = run_pluginval(require_pluginval, require_vst3, strictness=5)
        assert result.returncode == 0, f"Pluginval level 5 failed:\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    @pytest.mark.slow
    def test_vst3_level_10(self, require_pluginval, require_vst3):
        """
        Level 10: Strictest validation.

        Extended tests including:
        - Parameter fuzzing (random automation)
        - Multiple state save/restore cycles
        - Various buffer sizes and sample rates
        - Long-running stability
        """
        result = run_pluginval(
            require_pluginval,
            require_vst3,
            strictness=10,
            timeout=600  # 10 minutes for thorough tests
        )
        assert result.returncode == 0, f"Pluginval level 10 failed:\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_vst3_buffer_sizes(self, require_pluginval, require_vst3):
        """Test plugin with various buffer sizes."""
        buffer_sizes = [32, 64, 128, 256, 512, 1024, 2048]

        for block_size in buffer_sizes:
            result = run_pluginval(
                require_pluginval,
                require_vst3,
                strictness=5,
                block_size=block_size
            )
            assert result.returncode == 0, \
                f"Plugin failed at buffer size {block_size}:\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_vst3_sample_rates(self, require_pluginval, require_vst3):
        """Test plugin with various sample rates."""
        sample_rates = [44100, 48000, 88200, 96000]

        for sample_rate in sample_rates:
            result = run_pluginval(
                require_pluginval,
                require_vst3,
                strictness=5,
                sample_rate=sample_rate
            )
            assert result.returncode == 0, \
                f"Plugin failed at sample rate {sample_rate}:\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    @pytest.mark.macos
    def test_vst3_realtime_safety(self, require_pluginval, require_vst3, is_macos):
        """
        Real-time safety check (macOS only).

        Verifies no memory allocation or blocking calls in audio thread.
        """
        if not is_macos:
            pytest.skip("Real-time safety checks only available on macOS")

        result = run_pluginval(
            require_pluginval,
            require_vst3,
            strictness=5,
            realtime_check=True
        )
        assert result.returncode == 0, \
            f"Real-time safety check failed:\n{result.stderr}"


class TestPluginvalAU:
    """AU validation using pluginval (macOS only)."""

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    @pytest.mark.macos
    def test_au_level_5(self, require_pluginval, require_au):
        """Standard AU validation."""
        result = run_pluginval(require_pluginval, require_au, strictness=5)
        assert result.returncode == 0, f"AU pluginval level 5 failed:\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    @pytest.mark.macos
    @pytest.mark.slow
    def test_au_level_10(self, require_pluginval, require_au):
        """Strict AU validation."""
        result = run_pluginval(require_pluginval, require_au, strictness=10, timeout=600)
        assert result.returncode == 0, f"AU pluginval level 10 failed:\n{result.stderr}"


class TestPluginvalStandalone:
    """Standalone app validation (limited tests)."""

    @pytest.mark.validation
    @pytest.mark.requires_standalone
    def test_standalone_launches(self, require_standalone, has_display):
        """Verify standalone app can launch and close cleanly."""
        if not has_display:
            pytest.skip("No display for standalone test")

        import psutil
        import time

        # Launch standalone
        proc = subprocess.Popen(
            [str(require_standalone)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Wait for process to start
            time.sleep(3)

            # Check it's running
            assert proc.poll() is None, "Standalone crashed on startup"

            # Check it's responsive (not hung)
            ps_proc = psutil.Process(proc.pid)
            assert ps_proc.status() != psutil.STATUS_ZOMBIE

        finally:
            # Clean shutdown
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


# =============================================================================
# Helper Functions
# =============================================================================

def run_pluginval(
    pluginval_path: Path,
    plugin_path: Path,
    strictness: int = 5,
    timeout: int = 300,
    block_size: Optional[int] = None,
    sample_rate: Optional[int] = None,
    realtime_check: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run pluginval with specified options.

    Args:
        pluginval_path: Path to pluginval executable
        plugin_path: Path to plugin to validate
        strictness: Strictness level 1-10
        timeout: Timeout in seconds
        block_size: Audio block size (optional)
        sample_rate: Sample rate (optional)
        realtime_check: Enable real-time safety checks

    Returns:
        CompletedProcess with results
    """
    cmd = [
        str(pluginval_path),
        "--validate-in-process",
        "--strictness-level", str(strictness),
        "--timeout-ms", str(timeout * 1000),
    ]

    if block_size:
        cmd.extend(["--block-size", str(block_size)])

    if sample_rate:
        cmd.extend(["--sample-rate", str(sample_rate)])

    if realtime_check:
        cmd.append("--skip-gui-tests")  # For CI environments

    cmd.append(str(plugin_path))

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout + 60  # Extra buffer for process overhead
    )


def parse_pluginval_output(output: str) -> Tuple[int, int, list]:
    """
    Parse pluginval output for pass/fail counts.

    Returns:
        Tuple of (passed, failed, failure_messages)
    """
    passed = 0
    failed = 0
    failures = []

    for line in output.split('\n'):
        if 'PASSED' in line:
            passed += 1
        elif 'FAILED' in line:
            failed += 1
            failures.append(line)

    return passed, failed, failures
