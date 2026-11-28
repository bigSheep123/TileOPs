"""Run a lint on a probe file written under a test's ``tmp_path``."""

import subprocess
import sys
from pathlib import Path

LINT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "lint"


def probe_findings(lint, tmp_path: Path, source: str, name: str = "probe.py") -> list[str]:
    """The findings of the *lint* module on *source*, written to *name* under *tmp_path*."""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return lint.findings(path)


def run_lint_script(script: str, tmp_path: Path, content: str) -> subprocess.CompletedProcess:
    """Run ``scripts/lint/<script>`` as a subprocess on *content* written to a tmp file."""
    target = tmp_path / "fixture.py"
    target.write_text(content)
    return subprocess.run(
        [sys.executable, str(LINT_DIR / script), str(target)], capture_output=True, text=True
    )
