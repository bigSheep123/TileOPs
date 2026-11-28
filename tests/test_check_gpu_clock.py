"""Tests for scripts/ci/check_gpu_clock.sh."""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci" / "check_gpu_clock.sh"


def _run(tmp_path: Path, query_output: str, status: int = 0) -> int:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvidia-smi").write_text(f"#!/bin/sh\nprintf '{query_output}'\nexit {status}\n")
    (bin_dir / "sleep").write_text("#!/bin/sh\n")
    for tool in bin_dir.iterdir():
        tool.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True).returncode


@pytest.mark.parametrize(
    ("query_output", "status", "returncode"),
    [
        ("0, 1500\\n1, 1500\\n", 0, 0),
        ("0, 1500\\n1, 1980\\n", 0, 1),
        # A query that prints nothing must not read as every GPU locked.
        ("", 0, 1),
        # A failed query fails even with locked rows.
        ("0, 1500\\n", 1, 1),
    ],
)
def test_check_gpu_clock(tmp_path: Path, query_output: str, status: int, returncode: int) -> None:
    assert _run(tmp_path, query_output, status=status) == returncode
