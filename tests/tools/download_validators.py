"""
Download Validation Tools

Downloads pluginval and clap-validator for CI use.
"""

import os
import sys
import platform
import urllib.request
import zipfile
import tarfile
from pathlib import Path


PLUGINVAL_RELEASES = {
    "Windows": "https://github.com/Tracktion/pluginval/releases/latest/download/pluginval_Windows.zip",
    "Darwin": "https://github.com/Tracktion/pluginval/releases/latest/download/pluginval_macOS.zip",
    "Linux": "https://github.com/Tracktion/pluginval/releases/latest/download/pluginval_Linux.zip",
}

CLAP_VALIDATOR_RELEASES = {
    "Windows": "https://github.com/free-audio/clap-validator/releases/latest/download/clap-validator-windows.zip",
    "Darwin": "https://github.com/free-audio/clap-validator/releases/latest/download/clap-validator-macos.zip",
    "Linux": "https://github.com/free-audio/clap-validator/releases/latest/download/clap-validator-ubuntu-22.04.zip",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def make_executable(path: Path) -> None:
    """Make file executable on Unix systems."""
    if platform.system() != "Windows":
        import stat
        path.chmod(path.stat().st_mode | stat.S_IEXEC)


def download_pluginval(dest_dir: Path) -> Path:
    """
    Download pluginval for current platform.

    Returns:
        Path to pluginval executable
    """
    system = platform.system()
    url = PLUGINVAL_RELEASES.get(system)

    if not url:
        raise RuntimeError(f"No pluginval release for {system}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "pluginval.zip"
    if not download_file(url, zip_path):
        raise RuntimeError("Failed to download pluginval")

    if not extract_zip(zip_path, dest_dir):
        raise RuntimeError("Failed to extract pluginval")

    zip_path.unlink()

    # Find executable
    if system == "Windows":
        exe = dest_dir / "pluginval.exe"
    elif system == "Darwin":
        exe = dest_dir / "pluginval.app" / "Contents" / "MacOS" / "pluginval"
    else:
        exe = dest_dir / "pluginval"

    if exe.exists():
        make_executable(exe)
        return exe

    # Search for it
    for path in dest_dir.rglob("pluginval*"):
        if path.is_file():
            make_executable(path)
            return path

    raise RuntimeError("pluginval executable not found after extraction")


def download_clap_validator(dest_dir: Path) -> Path:
    """
    Download clap-validator for current platform.

    Returns:
        Path to clap-validator executable
    """
    system = platform.system()
    url = CLAP_VALIDATOR_RELEASES.get(system)

    if not url:
        raise RuntimeError(f"No clap-validator release for {system}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "clap-validator.zip"
    if not download_file(url, zip_path):
        raise RuntimeError("Failed to download clap-validator")

    if not extract_zip(zip_path, dest_dir):
        raise RuntimeError("Failed to extract clap-validator")

    zip_path.unlink()

    # Find executable
    if system == "Windows":
        exe = dest_dir / "clap-validator.exe"
    else:
        exe = dest_dir / "clap-validator"

    if exe.exists():
        make_executable(exe)
        return exe

    # Search for it
    for path in dest_dir.rglob("clap-validator*"):
        if path.is_file() and not path.suffix:
            make_executable(path)
            return path

    raise RuntimeError("clap-validator executable not found after extraction")


def main():
    """Download all validation tools."""
    tools_dir = Path(__file__).parent.parent.parent / "tools" / "validators"

    print("Downloading validation tools...")
    print(f"Target directory: {tools_dir}")
    print(f"Platform: {platform.system()}")
    print()

    try:
        pluginval = download_pluginval(tools_dir)
        print(f"pluginval: {pluginval}")
    except Exception as e:
        print(f"Failed to download pluginval: {e}")

    try:
        clap_val = download_clap_validator(tools_dir)
        print(f"clap-validator: {clap_val}")
    except Exception as e:
        print(f"Failed to download clap-validator: {e}")


if __name__ == "__main__":
    main()
