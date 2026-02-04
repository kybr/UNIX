"""
HTML Report Generator

Generates HTML reports from JSON test results with charts and visualizations.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# HTML template with embedded CSS and Chart.js
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KarplusStrongPlugin Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eee;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --success: #4ade80;
            --warning: #fbbf24;
            --error: #f87171;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--bg-card);
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .timestamp {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .summary-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }

        .summary-card h3 {
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .summary-card .value {
            font-size: 2.5rem;
            font-weight: 700;
        }

        .summary-card.passed .value { color: var(--success); }
        .summary-card.failed .value { color: var(--error); }
        .summary-card.skipped .value { color: var(--warning); }

        section {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }

        section h2 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent);
            display: inline-block;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-card);
        }

        th {
            background: var(--bg-card);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }

        tr:hover {
            background: var(--bg-card);
        }

        .status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .status.pass { background: rgba(74, 222, 128, 0.2); color: var(--success); }
        .status.fail { background: rgba(248, 113, 113, 0.2); color: var(--error); }

        .metric {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin: 2rem 0;
        }

        .category-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .category-item {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }

        .category-item h4 {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .category-item .numbers {
            font-size: 1.25rem;
        }

        .category-item .pass-count { color: var(--success); }
        .category-item .fail-count { color: var(--error); }

        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        footer a {
            color: var(--accent);
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>KarplusStrongPlugin Test Report</h1>
            <p class="timestamp">Generated: {{ timestamp }}</p>
        </header>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{{ total_tests }}</div>
            </div>
            <div class="summary-card passed">
                <h3>Passed</h3>
                <div class="value">{{ passed }}</div>
            </div>
            <div class="summary-card failed">
                <h3>Failed</h3>
                <div class="value">{{ failed }}</div>
            </div>
            <div class="summary-card skipped">
                <h3>Skipped</h3>
                <div class="value">{{ skipped }}</div>
            </div>
        </div>

        <section>
            <h2>Test Categories</h2>
            <div class="category-breakdown">
                {% for cat in categories %}
                <div class="category-item">
                    <h4>{{ cat.name }}</h4>
                    <div class="numbers">
                        <span class="pass-count">{{ cat.passed }}</span> /
                        <span class="fail-count">{{ cat.failed }}</span>
                    </div>
                </div>
                {% endfor %}
            </div>
        </section>

        {% if audio_metrics %}
        <section>
            <h2>Audio Quality</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Value</th>
                        <th>Notes</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric in audio_metrics %}
                    <tr>
                        <td>{{ metric.test_name }}</td>
                        <td class="metric">
                            {% if metric.thd_db %}THD: {{ metric.thd_db|round(1) }} dB{% endif %}
                            {% if metric.snr_db %}SNR: {{ metric.snr_db|round(1) }} dB{% endif %}
                            {% if metric.pitch_error_cents %}Pitch: {{ metric.pitch_error_cents|round(1) }} cents{% endif %}
                            {% if metric.fundamental_hz %}Freq: {{ metric.fundamental_hz|round(1) }} Hz{% endif %}
                        </td>
                        <td>{{ metric.notes }}</td>
                        <td><span class="status {{ 'pass' if metric.passed else 'fail' }}">
                            {{ 'PASS' if metric.passed else 'FAIL' }}
                        </span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        {% endif %}

        {% if performance_metrics %}
        <section>
            <h2>Performance</h2>
            <div class="chart-container">
                <canvas id="cpuChart"></canvas>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Voices</th>
                        <th>CPU %</th>
                        <th>RAM MB</th>
                        <th>Latency ms</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric in performance_metrics %}
                    <tr>
                        <td>{{ metric.test_name }}</td>
                        <td class="metric">{{ metric.voice_count }}</td>
                        <td class="metric">{{ metric.cpu_percent|round(2) if metric.cpu_percent else '-' }}</td>
                        <td class="metric">{{ metric.ram_mb|round(1) if metric.ram_mb else '-' }}</td>
                        <td class="metric">{{ metric.latency_ms|round(1) if metric.latency_ms else '-' }}</td>
                        <td><span class="status {{ 'pass' if metric.passed else 'fail' }}">
                            {{ 'PASS' if metric.passed else 'FAIL' }}
                        </span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        {% endif %}

        {% if midi_metrics %}
        <section>
            <h2>MIDI Response</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Latency</th>
                        <th>Pitch Accuracy</th>
                        <th>Notes</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric in midi_metrics %}
                    <tr>
                        <td>{{ metric.test_name }}</td>
                        <td class="metric">
                            {{ metric.note_on_latency_ms|round(1) if metric.note_on_latency_ms else '-' }} ms
                        </td>
                        <td class="metric">
                            {{ metric.pitch_accuracy_cents|round(1) if metric.pitch_accuracy_cents else '-' }} cents
                        </td>
                        <td>{{ metric.notes }}</td>
                        <td><span class="status {{ 'pass' if metric.passed else 'fail' }}">
                            {{ 'PASS' if metric.passed else 'FAIL' }}
                        </span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        {% endif %}

        {% if validation_metrics %}
        <section>
            <h2>Validation</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test</th>
                        <th>Validator</th>
                        <th>Level</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric in validation_metrics %}
                    <tr>
                        <td>{{ metric.test_name }}</td>
                        <td>{{ metric.validator }}</td>
                        <td class="metric">{{ metric.level }}</td>
                        <td><span class="status {{ 'pass' if metric.passed else 'fail' }}">
                            {{ 'PASS' if metric.passed else 'FAIL' }}
                        </span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>
        {% endif %}

        <footer>
            <p>Generated by KarplusStrongPlugin Test Suite</p>
            <p><a href="https://github.com/anthropics/claude-code">Built with Claude Code</a></p>
        </footer>
    </div>

    <script>
        // CPU Chart
        {% if cpu_chart_data %}
        const ctx = document.getElementById('cpuChart');
        if (ctx) {
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: {{ cpu_chart_labels | safe }},
                    datasets: [{
                        label: 'CPU Usage %',
                        data: {{ cpu_chart_data | safe }},
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: { color: '#eee' }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#a0a0a0' }
                        },
                        x: {
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            ticks: { color: '#a0a0a0' }
                        }
                    }
                }
            });
        }
        {% endif %}
    </script>
</body>
</html>"""


class ReportGenerator:
    """
    Generates HTML reports from JSON test results.

    Reads all JSON files from reports subdirectories and generates
    a comprehensive HTML report with charts and visualizations.
    """

    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)

    def generate(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate HTML report from all JSON results.

        Args:
            output_path: Where to write the HTML file. Defaults to reports/report.html

        Returns:
            Path to generated report
        """
        data = self._collect_results()
        html = self._render_template(data)

        if output_path is None:
            output_path = self.reports_dir / 'report.html'

        output_path.write_text(html, encoding='utf-8')
        return output_path

    def _collect_results(self) -> Dict[str, Any]:
        """Collect all JSON results into report data."""
        audio_metrics = []
        performance_metrics = []
        midi_metrics = []
        validation_metrics = []

        # Read audio analysis results
        audio_dir = self.reports_dir / 'audio_analysis'
        if audio_dir.exists():
            for json_file in audio_dir.glob('*.json'):
                data = self._read_json(json_file)
                if data and 'results' in data:
                    audio_metrics.extend(data['results'])

        # Read performance results
        perf_dir = self.reports_dir / 'performance'
        if perf_dir.exists():
            for json_file in perf_dir.glob('*.json'):
                data = self._read_json(json_file)
                if data and 'results' in data:
                    performance_metrics.extend(data['results'])

        # Read MIDI results
        midi_dir = self.reports_dir / 'midi'
        if midi_dir.exists():
            for json_file in midi_dir.glob('*.json'):
                data = self._read_json(json_file)
                if data and 'results' in data:
                    midi_metrics.extend(data['results'])

        # Read validation results
        val_dir = self.reports_dir / 'validation'
        if val_dir.exists():
            for json_file in val_dir.glob('*.json'):
                data = self._read_json(json_file)
                if data and 'results' in data:
                    validation_metrics.extend(data['results'])

        # Calculate totals
        all_metrics = audio_metrics + performance_metrics + midi_metrics + validation_metrics
        total = len(all_metrics)
        passed = sum(1 for m in all_metrics if m.get('passed', False))
        failed = total - passed

        # Build categories
        categories = [
            {'name': 'Audio', 'passed': sum(1 for m in audio_metrics if m.get('passed')),
             'failed': sum(1 for m in audio_metrics if not m.get('passed'))},
            {'name': 'Performance', 'passed': sum(1 for m in performance_metrics if m.get('passed')),
             'failed': sum(1 for m in performance_metrics if not m.get('passed'))},
            {'name': 'MIDI', 'passed': sum(1 for m in midi_metrics if m.get('passed')),
             'failed': sum(1 for m in midi_metrics if not m.get('passed'))},
            {'name': 'Validation', 'passed': sum(1 for m in validation_metrics if m.get('passed')),
             'failed': sum(1 for m in validation_metrics if not m.get('passed'))},
        ]

        # Build CPU chart data
        cpu_chart_labels = []
        cpu_chart_data = []
        for m in performance_metrics:
            if m.get('cpu_percent') is not None:
                label = m.get('test_name', 'Unknown')
                if m.get('voice_count'):
                    label = f"{m['voice_count']} voices"
                cpu_chart_labels.append(label)
                cpu_chart_data.append(m['cpu_percent'])

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'skipped': 0,  # Would need to track this separately
            'categories': categories,
            'audio_metrics': audio_metrics,
            'performance_metrics': performance_metrics,
            'midi_metrics': midi_metrics,
            'validation_metrics': validation_metrics,
            'cpu_chart_labels': json.dumps(cpu_chart_labels),
            'cpu_chart_data': json.dumps(cpu_chart_data),
        }

    def _render_template(self, data: Dict[str, Any]) -> str:
        """Render the HTML template with data."""
        try:
            from jinja2 import Template
            template = Template(HTML_TEMPLATE)
            return template.render(**data)
        except ImportError:
            # Fallback: simple string replacement for basic functionality
            return self._simple_render(data)

    def _simple_render(self, data: Dict[str, Any]) -> str:
        """Simple template rendering without Jinja2."""
        html = HTML_TEMPLATE

        # Replace simple placeholders
        replacements = {
            '{{ timestamp }}': str(data.get('timestamp', '')),
            '{{ total_tests }}': str(data.get('total_tests', 0)),
            '{{ passed }}': str(data.get('passed', 0)),
            '{{ failed }}': str(data.get('failed', 0)),
            '{{ skipped }}': str(data.get('skipped', 0)),
        }

        for key, value in replacements.items():
            html = html.replace(key, value)

        # Remove Jinja2 control structures for simple render
        # This is a basic fallback - full functionality requires Jinja2
        import re
        html = re.sub(r'{%.*?%}', '', html, flags=re.DOTALL)
        html = re.sub(r'{{.*?}}', '', html)

        return html

    def _read_json(self, path: Path) -> Optional[Dict]:
        """Read a JSON file safely."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None


def generate_report(reports_dir: str = 'reports') -> Path:
    """
    Convenience function to generate a report.

    Args:
        reports_dir: Path to reports directory

    Returns:
        Path to generated HTML report
    """
    generator = ReportGenerator(Path(reports_dir))
    return generator.generate()


if __name__ == '__main__':
    import sys
    reports_path = sys.argv[1] if len(sys.argv) > 1 else 'reports'
    output = generate_report(reports_path)
    print(f"Report generated: {output}")
