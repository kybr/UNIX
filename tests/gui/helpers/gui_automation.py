"""
GUI Automation Module

Provides utilities for automating GUI testing:
- Application launching and management
- Mouse/keyboard automation
- Screenshot capture and comparison
- Control interaction

Uses PyAutoGUI and Pillow for cross-platform automation.
"""

import os
import time
import subprocess
import platform
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    from PIL import Image, ImageChops
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pygetwindow as gw
    HAS_PYGETWINDOW = True
except ImportError:
    HAS_PYGETWINDOW = False


@dataclass
class WindowInfo:
    """Information about a window."""
    title: str
    x: int
    y: int
    width: int
    height: int
    handle: object  # Platform-specific handle


@dataclass
class ScreenshotDiff:
    """Result of screenshot comparison."""
    identical: bool
    difference_percent: float
    diff_image: Optional[Image.Image]


class GUIAutomation:
    """
    Cross-platform GUI automation for plugin testing.

    Provides methods to:
    - Launch and manage applications
    - Interact with controls (click, drag, type)
    - Capture and compare screenshots
    """

    def __init__(self, failsafe: bool = True):
        if not HAS_PYAUTOGUI:
            raise ImportError("pyautogui required for GUI automation")

        pyautogui.FAILSAFE = failsafe
        pyautogui.PAUSE = 0.1  # Small pause between actions

        self.process: Optional[subprocess.Popen] = None
        self.window: Optional[WindowInfo] = None

    def launch_app(
        self,
        app_path: Path,
        wait_seconds: float = 5.0,
        window_title: Optional[str] = None
    ) -> bool:
        """
        Launch an application and wait for its window.

        Args:
            app_path: Path to executable
            wait_seconds: Time to wait for window
            window_title: Expected window title (partial match)

        Returns:
            True if successfully launched and window found
        """
        try:
            # Handle macOS .app bundles
            if platform.system() == "Darwin" and str(app_path).endswith(".app"):
                self.process = subprocess.Popen(
                    ["open", "-a", str(app_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                self.process = subprocess.Popen(
                    [str(app_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            # Wait for window
            start = time.time()
            while time.time() - start < wait_seconds:
                self.window = self._find_window(window_title or app_path.stem)
                if self.window:
                    return True
                time.sleep(0.5)

            return False

        except Exception as e:
            print(f"Launch error: {e}")
            return False

    def close_app(self, timeout: float = 5.0) -> bool:
        """
        Close the application gracefully.

        Returns:
            True if closed successfully
        """
        if self.process is None:
            return True

        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
            self.process = None
            self.window = None
            return True

        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process = None
            self.window = None
            return False

    def focus_window(self) -> bool:
        """Bring the application window to foreground."""
        if self.window is None:
            return False

        if HAS_PYGETWINDOW:
            try:
                windows = gw.getWindowsWithTitle(self.window.title)
                if windows:
                    windows[0].activate()
                    return True
            except Exception:
                pass

        return False

    def click(self, x: int, y: int, button: str = 'left') -> None:
        """
        Click at absolute screen coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: 'left', 'right', or 'middle'
        """
        pyautogui.click(x, y, button=button)

    def click_relative(self, rel_x: float, rel_y: float, button: str = 'left') -> None:
        """
        Click at position relative to window (0-1 range).

        Args:
            rel_x: Relative X (0 = left edge, 1 = right edge)
            rel_y: Relative Y (0 = top edge, 1 = bottom edge)
            button: Mouse button
        """
        if self.window is None:
            return

        x = self.window.x + int(rel_x * self.window.width)
        y = self.window.y + int(rel_y * self.window.height)
        self.click(x, y, button)

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5
    ) -> None:
        """Drag from start position to end position."""
        pyautogui.moveTo(start_x, start_y)
        pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)

    def drag_relative(
        self,
        start_rel: Tuple[float, float],
        end_rel: Tuple[float, float],
        duration: float = 0.5
    ) -> None:
        """Drag using relative window coordinates."""
        if self.window is None:
            return

        start_x = self.window.x + int(start_rel[0] * self.window.width)
        start_y = self.window.y + int(start_rel[1] * self.window.height)
        end_x = self.window.x + int(end_rel[0] * self.window.width)
        end_y = self.window.y + int(end_rel[1] * self.window.height)

        self.drag(start_x, start_y, end_x, end_y, duration)

    def type_text(self, text: str, interval: float = 0.05) -> None:
        """Type text using keyboard."""
        pyautogui.typewrite(text, interval=interval)

    def press_key(self, key: str) -> None:
        """Press a single key."""
        pyautogui.press(key)

    def hotkey(self, *keys: str) -> None:
        """Press a key combination."""
        pyautogui.hotkey(*keys)

    def screenshot(self, save_path: Optional[Path] = None) -> Optional[Image.Image]:
        """
        Take a screenshot.

        Args:
            save_path: Optional path to save screenshot

        Returns:
            PIL Image or None if failed
        """
        try:
            if self.window:
                # Screenshot just the window
                img = pyautogui.screenshot(region=(
                    self.window.x,
                    self.window.y,
                    self.window.width,
                    self.window.height
                ))
            else:
                # Full screen
                img = pyautogui.screenshot()

            if save_path:
                img.save(str(save_path))

            return img

        except Exception as e:
            print(f"Screenshot error: {e}")
            return None

    def locate_on_screen(
        self,
        image_path: Path,
        confidence: float = 0.9
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate an image on screen.

        Args:
            image_path: Path to reference image
            confidence: Matching confidence (0-1)

        Returns:
            (x, y, width, height) or None if not found
        """
        try:
            location = pyautogui.locateOnScreen(
                str(image_path),
                confidence=confidence
            )
            return location

        except Exception:
            return None

    def click_image(
        self,
        image_path: Path,
        confidence: float = 0.9
    ) -> bool:
        """
        Click on an image if found on screen.

        Returns:
            True if found and clicked
        """
        location = self.locate_on_screen(image_path, confidence)
        if location:
            center = pyautogui.center(location)
            self.click(center.x, center.y)
            return True
        return False

    def _find_window(self, title_substring: str) -> Optional[WindowInfo]:
        """Find a window by partial title match."""
        if not HAS_PYGETWINDOW:
            return None

        try:
            windows = gw.getWindowsWithTitle(title_substring)
            if windows:
                w = windows[0]
                return WindowInfo(
                    title=w.title,
                    x=w.left,
                    y=w.top,
                    width=w.width,
                    height=w.height,
                    handle=w
                )
        except Exception:
            pass

        return None


class ScreenshotComparator:
    """Compare screenshots for visual regression testing."""

    def __init__(self, threshold: float = 0.02):
        """
        Initialize comparator.

        Args:
            threshold: Maximum allowed difference (0-1)
        """
        if not HAS_PIL:
            raise ImportError("Pillow required for screenshot comparison")

        self.threshold = threshold

    def compare(
        self,
        baseline: Path,
        current: Image.Image
    ) -> ScreenshotDiff:
        """
        Compare current screenshot to baseline.

        Args:
            baseline: Path to baseline image
            current: Current screenshot

        Returns:
            ScreenshotDiff with comparison results
        """
        if not baseline.exists():
            return ScreenshotDiff(
                identical=False,
                difference_percent=100.0,
                diff_image=None
            )

        baseline_img = Image.open(baseline)

        # Ensure same size
        if baseline_img.size != current.size:
            current = current.resize(baseline_img.size)

        # Convert to same mode
        if baseline_img.mode != current.mode:
            current = current.convert(baseline_img.mode)

        # Calculate difference
        diff = ImageChops.difference(baseline_img, current)

        # Convert to numpy for analysis
        diff_array = np.array(diff)
        diff_percent = (np.count_nonzero(diff_array) / diff_array.size) * 100

        identical = diff_percent <= self.threshold * 100

        return ScreenshotDiff(
            identical=identical,
            difference_percent=diff_percent,
            diff_image=diff if not identical else None
        )

    def save_diff(self, diff_result: ScreenshotDiff, path: Path) -> None:
        """Save diff image for debugging."""
        if diff_result.diff_image:
            diff_result.diff_image.save(str(path))
