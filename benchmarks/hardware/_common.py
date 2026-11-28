"""Helpers the hardware microbenchmark scripts share."""

import statistics
import subprocess
import sys
import threading


def compile_cuda(cu_path, binary_path, arch="sm_90"):
    """Compile the CUDA source; exit the process on failure."""
    cmd = [
        "nvcc",
        "-O3",
        f"-arch={arch}",
        "-Wno-deprecated-gpu-targets",
        "-o",
        str(binary_path),
        str(cu_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"nvcc compilation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


class ClockSampler(threading.Thread):
    """Sample SM clock, power and temperature while the benchmark runs.

    A calibration is only meaningful alongside the clocks it was taken at;
    recording them turns an unreproducible number into a conditional one.
    """

    def __init__(self, gpu_index, interval_ms=200):
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples = []  # (sm_mhz, watts, celsius)
        self._stopped = threading.Event()
        self._proc = None

    def run(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"-lms={self.interval_ms}",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError:
            return  # no nvidia-smi: telemetry is best-effort
        self._proc = proc
        for line in proc.stdout:
            if self._stopped.is_set():
                break
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                self.samples.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
        proc.terminate()

    def stop(self):
        self._stopped.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)  # reap; terminate alone leaves a zombie
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def summary(self):
        """Return (sm_min, sm_median, sm_max, watts_max, temp_max) or None."""
        if not self.samples:
            return None
        sm = sorted(s[0] for s in self.samples)
        return (
            sm[0],
            statistics.median(sm),
            sm[-1],
            max(s[1] for s in self.samples),
            max(s[2] for s in self.samples),
        )
