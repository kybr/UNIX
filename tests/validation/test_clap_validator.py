"""
CLAP Validator Tests

Runs the official CLAP validator tool against CLAP plugins
to verify format compliance and stability.

CLAP Validator: https://github.com/free-audio/clap-validator
"""

import subprocess
import json
from pathlib import Path
from typing import Optional

import pytest


class TestCLAPValidator:
    """CLAP format validation tests."""

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_clap_basic_validation(self, require_clap_validator, require_clap):
        """
        Basic CLAP validation.

        Runs the validator's standard test suite including:
        - Factory and descriptor validation
        - Parameter discovery
        - Audio processing
        """
        result = run_clap_validator(
            require_clap_validator,
            require_clap,
            only_failed=False
        )

        assert result.returncode == 0, \
            f"CLAP validation failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_clap_fuzzing(self, require_clap_validator, require_clap):
        """
        CLAP fuzzing test.

        Generates random parameter permutations and verifies:
        - No crashes
        - No NaN/Inf values in output
        - Stable processing
        """
        result = run_clap_validator(
            require_clap_validator,
            require_clap,
            tests=["fuzzing"]
        )

        assert result.returncode == 0, \
            f"CLAP fuzzing test failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_clap_presets(self, require_clap_validator, require_clap):
        """
        Test CLAP preset discovery and loading.

        Verifies the preset discovery factory (CLAP 1.1.7+).
        """
        result = run_clap_validator(
            require_clap_validator,
            require_clap,
            tests=["presets"]
        )

        # Preset tests may be skipped if plugin doesn't support presets
        if "skipped" in result.stdout.lower():
            pytest.skip("Plugin doesn't support preset discovery")

        assert result.returncode == 0, \
            f"CLAP preset test failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    @pytest.mark.slow
    def test_clap_full_validation(self, require_clap_validator, require_clap):
        """
        Complete CLAP validation suite.

        Runs all available tests including fuzzing with extended iterations.
        """
        result = run_clap_validator(
            require_clap_validator,
            require_clap,
            only_failed=False,
            timeout=600
        )

        assert result.returncode == 0, \
            f"Full CLAP validation failed:\n{result.stdout}\n{result.stderr}"


class TestCLAPCompatibility:
    """CLAP compatibility and edge case tests."""

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_clap_in_process(self, require_clap_validator, require_clap):
        """
        Run validation in-process for debugging.

        Allows attaching debugger to investigate failures.
        """
        result = run_clap_validator(
            require_clap_validator,
            require_clap,
            in_process=True
        )

        assert result.returncode == 0, \
            f"In-process CLAP validation failed:\n{result.stdout}"

    @pytest.mark.validation
    @pytest.mark.requires_plugin
    def test_clap_list_info(self, require_clap_validator, require_clap):
        """List plugin info to verify metadata."""
        result = subprocess.run(
            [
                str(require_clap_validator),
                "list",
                "plugins",
                str(require_clap)
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, f"Failed to list CLAP info:\n{result.stderr}"

        # Verify expected plugin info in output
        output = result.stdout.lower()
        assert "karplus" in output or "synth" in output, \
            f"Plugin info not found in output:\n{result.stdout}"


# =============================================================================
# Helper Functions
# =============================================================================

def run_clap_validator(
    validator_path: Path,
    plugin_path: Path,
    tests: Optional[list] = None,
    only_failed: bool = True,
    in_process: bool = False,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """
    Run clap-validator with specified options.

    Args:
        validator_path: Path to clap-validator executable
        plugin_path: Path to CLAP plugin
        tests: Specific tests to run (optional)
        only_failed: Only show failed test output
        in_process: Run tests in current process
        timeout: Timeout in seconds

    Returns:
        CompletedProcess with results
    """
    cmd = [str(validator_path), "validate"]

    if only_failed:
        cmd.append("--only-failed")

    if in_process:
        cmd.append("--in-process")

    if tests:
        for test in tests:
            cmd.extend(["--test", test])

    cmd.append(str(plugin_path))

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )


def list_clap_tests(validator_path: Path) -> list:
    """List all available CLAP validator tests."""
    result = subprocess.run(
        [str(validator_path), "list", "tests"],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        return []

    # Parse test names from output
    tests = []
    for line in result.stdout.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            tests.append(line.split()[0])

    return tests
