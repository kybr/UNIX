"""
Unified report writer for all test categories.
Writes JSON files to reports/ subdirectories with timestamps.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class AudioAnalysisResult:
    """Result from an audio quality test."""
    test_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    thd_db: Optional[float] = None
    snr_db: Optional[float] = None
    fundamental_hz: Optional[float] = None
    pitch_error_cents: Optional[float] = None
    dc_offset: Optional[float] = None
    peak_amplitude: Optional[float] = None
    clipping_samples: int = 0
    passed: bool = True
    notes: str = ""


@dataclass
class PerformanceResult:
    """Result from a performance test."""
    test_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    cpu_percent: Optional[float] = None
    ram_mb: Optional[float] = None
    latency_ms: Optional[float] = None
    realtime_ratio: Optional[float] = None
    voice_count: int = 1
    sample_rate: int = 48000
    buffer_size: int = 512
    passed: bool = True
    notes: str = ""


@dataclass
class MIDITestResult:
    """Result from a MIDI test."""
    test_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    note_on_latency_ms: Optional[float] = None
    note_off_latency_ms: Optional[float] = None
    pitch_accuracy_cents: Optional[float] = None
    velocity_response_error: Optional[float] = None
    polyphony_achieved: int = 0
    passed: bool = True
    notes: str = ""


@dataclass
class ValidationResult:
    """Result from a validation test."""
    test_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validator: str = ""
    level: int = 0
    passed: bool = True
    output: str = ""
    notes: str = ""


class ReportWriter:
    """
    Writes test results to reports directory as JSON files.

    Usage:
        writer = ReportWriter(Path("reports"))
        writer.add_audio_result(AudioAnalysisResult(...))
        writer.add_performance_result(PerformanceResult(...))
        writer.finalize()  # Writes all accumulated results
    """

    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._ensure_dirs()

        # Accumulated results
        self.audio_results: List[AudioAnalysisResult] = []
        self.performance_results: List[PerformanceResult] = []
        self.midi_results: List[MIDITestResult] = []
        self.validation_results: List[ValidationResult] = []

        logger.info(f"ReportWriter initialized: session={self.session_id}")

    def _ensure_dirs(self):
        """Create report subdirectories if they don't exist."""
        subdirs = ['audio_analysis', 'performance', 'validation', 'midi', 'screenshots']
        for subdir in subdirs:
            (self.reports_dir / subdir).mkdir(parents=True, exist_ok=True)

    def add_audio_result(self, result: AudioAnalysisResult):
        """Add an audio analysis result."""
        self.audio_results.append(result)
        logger.debug(f"Added audio result: {result.test_name}")

    def add_performance_result(self, result: PerformanceResult):
        """Add a performance result."""
        self.performance_results.append(result)
        logger.debug(f"Added performance result: {result.test_name}")

    def add_midi_result(self, result: MIDITestResult):
        """Add a MIDI test result."""
        self.midi_results.append(result)
        logger.debug(f"Added MIDI result: {result.test_name}")

    def add_validation_result(self, result: ValidationResult):
        """Add a validation result."""
        self.validation_results.append(result)
        logger.debug(f"Added validation result: {result.test_name}")

    def write_audio_analysis(self) -> Optional[Path]:
        """Write audio analysis results to JSON."""
        if not self.audio_results:
            return None

        path = self.reports_dir / 'audio_analysis' / f'analysis_{self.session_id}.json'
        data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.audio_results],
            'summary': {
                'total': len(self.audio_results),
                'passed': sum(1 for r in self.audio_results if r.passed),
                'failed': sum(1 for r in self.audio_results if not r.passed),
            }
        }
        self._write_json(path, data)
        return path

    def write_performance(self) -> Optional[Path]:
        """Write performance results to JSON."""
        if not self.performance_results:
            return None

        path = self.reports_dir / 'performance' / f'benchmark_{self.session_id}.json'
        data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.performance_results],
            'summary': {
                'total': len(self.performance_results),
                'passed': sum(1 for r in self.performance_results if r.passed),
                'failed': sum(1 for r in self.performance_results if not r.passed),
                'avg_cpu_percent': self._safe_avg([r.cpu_percent for r in self.performance_results]),
                'avg_ram_mb': self._safe_avg([r.ram_mb for r in self.performance_results]),
                'avg_latency_ms': self._safe_avg([r.latency_ms for r in self.performance_results]),
            }
        }
        self._write_json(path, data)
        return path

    def write_midi(self) -> Optional[Path]:
        """Write MIDI test results to JSON."""
        if not self.midi_results:
            return None

        path = self.reports_dir / 'midi' / f'midi_{self.session_id}.json'
        data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.midi_results],
            'summary': {
                'total': len(self.midi_results),
                'passed': sum(1 for r in self.midi_results if r.passed),
                'failed': sum(1 for r in self.midi_results if not r.passed),
            }
        }
        self._write_json(path, data)
        return path

    def write_validation(self) -> Optional[Path]:
        """Write validation results to JSON."""
        if not self.validation_results:
            return None

        path = self.reports_dir / 'validation' / f'validation_{self.session_id}.json'
        data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'results': [asdict(r) for r in self.validation_results],
            'summary': {
                'total': len(self.validation_results),
                'passed': sum(1 for r in self.validation_results if r.passed),
                'failed': sum(1 for r in self.validation_results if not r.passed),
            }
        }
        self._write_json(path, data)
        return path

    def write_summary(self, test_session_info: Dict[str, Any] = None) -> Path:
        """Write aggregated summary report."""
        path = self.reports_dir / f'summary_{self.session_id}.json'

        all_results = (
            self.audio_results +
            self.performance_results +
            self.midi_results +
            self.validation_results
        )

        data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'test_session': test_session_info or {},
            'totals': {
                'audio_tests': len(self.audio_results),
                'performance_tests': len(self.performance_results),
                'midi_tests': len(self.midi_results),
                'validation_tests': len(self.validation_results),
                'total_tests': len(all_results),
                'total_passed': sum(1 for r in all_results if r.passed),
                'total_failed': sum(1 for r in all_results if not r.passed),
            },
            'categories': {
                'audio': {
                    'passed': sum(1 for r in self.audio_results if r.passed),
                    'failed': sum(1 for r in self.audio_results if not r.passed),
                },
                'performance': {
                    'passed': sum(1 for r in self.performance_results if r.passed),
                    'failed': sum(1 for r in self.performance_results if not r.passed),
                },
                'midi': {
                    'passed': sum(1 for r in self.midi_results if r.passed),
                    'failed': sum(1 for r in self.midi_results if not r.passed),
                },
                'validation': {
                    'passed': sum(1 for r in self.validation_results if r.passed),
                    'failed': sum(1 for r in self.validation_results if not r.passed),
                },
            }
        }
        self._write_json(path, data)
        return path

    def finalize(self, test_session_info: Dict[str, Any] = None) -> Dict[str, Path]:
        """Write all accumulated results to files."""
        paths = {}

        audio_path = self.write_audio_analysis()
        if audio_path:
            paths['audio_analysis'] = audio_path

        perf_path = self.write_performance()
        if perf_path:
            paths['performance'] = perf_path

        midi_path = self.write_midi()
        if midi_path:
            paths['midi'] = midi_path

        val_path = self.write_validation()
        if val_path:
            paths['validation'] = val_path

        paths['summary'] = self.write_summary(test_session_info)

        logger.info(f"Reports written: {list(paths.keys())}")
        return paths

    def _write_json(self, path: Path, data: Dict):
        """Write data to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Wrote report: {path}")

    def _safe_avg(self, values: List[Optional[float]]) -> Optional[float]:
        """Calculate average, ignoring None values."""
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None


# Convenience function for quick result creation
def create_audio_result(test_name: str, **kwargs) -> AudioAnalysisResult:
    """Create an AudioAnalysisResult with the given parameters."""
    return AudioAnalysisResult(test_name=test_name, **kwargs)


def create_performance_result(test_name: str, **kwargs) -> PerformanceResult:
    """Create a PerformanceResult with the given parameters."""
    return PerformanceResult(test_name=test_name, **kwargs)


def create_midi_result(test_name: str, **kwargs) -> MIDITestResult:
    """Create a MIDITestResult with the given parameters."""
    return MIDITestResult(test_name=test_name, **kwargs)


def create_validation_result(test_name: str, **kwargs) -> ValidationResult:
    """Create a ValidationResult with the given parameters."""
    return ValidationResult(test_name=test_name, **kwargs)
