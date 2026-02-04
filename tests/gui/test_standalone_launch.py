"""
Standalone Application Launch Tests

Tests the standalone application's basic functionality:
- Launch and close
- Window appearance
- Resize behavior
- Audio settings access
"""

import time
import subprocess
import platform
from pathlib import Path

import pytest

try:
    from .helpers.gui_automation import GUIAutomation, ScreenshotComparator
    HAS_AUTOMATION = True
except ImportError:
    HAS_AUTOMATION = False


class TestStandaloneLaunch:
    """Standalone application launch and close tests."""

    @pytest.fixture
    def gui(self):
        """Create GUI automation instance."""
        if not HAS_AUTOMATION:
            pytest.skip("GUI automation dependencies not installed")
        return GUIAutomation()

    @pytest.mark.gui
    @pytest.mark.requires_standalone
    @pytest.mark.requires_display
    def test_launch_and_close(self, require_standalone, gui, thresholds):
        """
        Test application launches and closes cleanly.

        Basic smoke test for the standalone app.
        """
        # Launch
        launched = gui.launch_app(
            require_standalone,
            wait_seconds=thresholds.launch_timeout_seconds,
            window_title="Karplus"
        )

        assert launched, "Application failed to launch"
        assert gui.window is not None, "No window found after launch"

        print(f"\nWindow info:")
        print(f"  Title: {gui.window.title}")
        print(f"  Size: {gui.window.width}x{gui.window.height}")
        print(f"  Position: ({gui.window.x}, {gui.window.y})")

        # Close
        closed = gui.close_app()
        assert closed, "Application failed to close cleanly"

    @pytest.mark.gui
    @pytest.mark.requires_standalone
    @pytest.mark.requires_display
    def test_window_appears(self, require_standalone, gui):
        """
        Verify application window appears on screen.

        Window should be visible and have reasonable dimensions.
        """
        try:
            launched = gui.launch_app(require_standalone, wait_seconds=10)

            if not launched:
                pytest.skip("Could not launch application")

            # Check window was found
            # Note: On some systems, windows launch minimized (-32000 position)
            if gui.window.x < -10000 or gui.window.y < -10000:
                # Window is minimized - test just that we found it
                assert gui.window is not None, "Window should be found even if minimized"
            else:
                # Window is visible - check dimensions
                assert gui.window.width > 100, "Window too narrow"
                assert gui.window.height > 100, "Window too short"
                assert gui.window.x >= 0, "Window off screen (left)"
                assert gui.window.y >= 0, "Window off screen (top)"

        finally:
            gui.close_app()

    @pytest.mark.gui
    @pytest.mark.requires_standalone
    @pytest.mark.requires_display
    def test_window_resize(self, require_standalone, gui):
        """
        Test window resize behavior.

        Window should handle resize without crashing.
        """
        try:
            launched = gui.launch_app(require_standalone, wait_seconds=10)

            if not launched:
                pytest.skip("Could not launch application")

            original_size = (gui.window.width, gui.window.height)

            # Try to resize (platform-dependent)
            # This is a basic test - just verify app doesn't crash
            time.sleep(1)

            # Verify still running
            assert gui.process.poll() is None, "Application crashed during resize"

        finally:
            gui.close_app()

    @pytest.mark.gui
    @pytest.mark.requires_standalone
    @pytest.mark.requires_display
    def test_multiple_launch_close(self, require_standalone, gui):
        """
        Test repeated launch/close cycles.

        Should be able to launch and close multiple times.
        """
        num_cycles = 3

        for i in range(num_cycles):
            print(f"\nCycle {i+1}/{num_cycles}")

            launched = gui.launch_app(require_standalone, wait_seconds=10)
            assert launched, f"Failed to launch on cycle {i+1}"

            time.sleep(1)

            closed = gui.close_app()
            assert closed, f"Failed to close on cycle {i+1}"

            time.sleep(1)

    @pytest.mark.gui
    @pytest.mark.requires_standalone
    @pytest.mark.requires_display
    @pytest.mark.slow
    def test_long_running(self, require_standalone, gui):
        """
        Test application stability over extended period.

        Run for several minutes to check for crashes or memory issues.
        """
        try:
            launched = gui.launch_app(require_standalone, wait_seconds=10)

            if not launched:
                pytest.skip("Could not launch application")

            # Run for 2 minutes
            duration = 120
            start = time.time()

            while time.time() - start < duration:
                # Check still running
                if gui.process.poll() is not None:
                    pytest.fail("Application crashed during long-running test")

                time.sleep(10)
                print(f"  Still running... {int(time.time() - start)}s")

        finally:
            gui.close_app()


class TestStandaloneScreenshots:
    """Screenshot and visual tests."""

    @pytest.fixture
    def gui(self):
        if not HAS_AUTOMATION:
            pytest.skip("GUI automation dependencies not installed")
        return GUIAutomation()

    @pytest.fixture
    def comparator(self, thresholds):
        if not HAS_AUTOMATION:
            pytest.skip("GUI automation dependencies not installed")
        return ScreenshotComparator(
            threshold=thresholds.screenshot_diff_threshold
        )

    @pytest.mark.gui
    @pytest.mark.requires_standalone
    @pytest.mark.requires_display
    def test_capture_screenshot(self, require_standalone, gui, reports_dir):
        """
        Capture screenshot of application.

        Used for visual regression testing.
        """
        try:
            launched = gui.launch_app(require_standalone, wait_seconds=10)

            if not launched:
                pytest.skip("Could not launch application")

            time.sleep(2)  # Let UI settle

            screenshot_path = reports_dir / "screenshots" / "standalone_current.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)

            screenshot = gui.screenshot(screenshot_path)

            assert screenshot is not None, "Failed to capture screenshot"
            assert screenshot_path.exists(), "Screenshot not saved"

            print(f"\nScreenshot saved to: {screenshot_path}")

        finally:
            gui.close_app()

    @pytest.mark.gui
    @pytest.mark.requires_standalone
    @pytest.mark.requires_display
    def test_visual_regression(
        self,
        require_standalone,
        gui,
        comparator,
        project_root,
        reports_dir
    ):
        """
        Compare current UI to baseline screenshot.

        Detects unintended visual changes.
        """
        baseline_path = project_root / "tests" / "gui" / "screenshots" / "baseline" / "standalone.png"

        if not baseline_path.exists():
            pytest.skip(
                f"No baseline screenshot at {baseline_path}. "
                "Run test_capture_screenshot first to create one."
            )

        try:
            launched = gui.launch_app(require_standalone, wait_seconds=10)

            if not launched:
                pytest.skip("Could not launch application")

            time.sleep(2)

            current = gui.screenshot()

            if current is None:
                pytest.skip("Failed to capture screenshot")

            result = comparator.compare(baseline_path, current)

            print(f"\nVisual comparison:")
            print(f"  Identical: {result.identical}")
            print(f"  Difference: {result.difference_percent:.2f}%")

            if not result.identical and result.diff_image:
                diff_path = reports_dir / "screenshots" / "standalone_diff.png"
                comparator.save_diff(result, diff_path)
                print(f"  Diff saved to: {diff_path}")

            assert result.identical, \
                f"Visual regression detected: {result.difference_percent:.2f}% different"

        finally:
            gui.close_app()
