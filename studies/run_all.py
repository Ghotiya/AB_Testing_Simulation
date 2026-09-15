"""Reproduce every result and figure from scratch: tests, Studies 1-4 (each gated), figures.

Stops at the first failing step (a failed gate exits 1). Takes about 36 s on one CPU thread.

Run: python -m studies.run_all   (via scripts/run_bg.sh on the shared machine)
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("tests", ["-m", "pytest", "-q"]),
    ("study1", ["-m", "studies.study1_baseline"]),
    ("study2", ["-m", "studies.study2_peeking"]),
    ("study3", ["-m", "studies.study3_power"]),
    ("study4", ["-m", "studies.study4_winners_curse"]),
    ("figures", ["-m", "absim.plots"]),
]


def main() -> int:
    start = time.perf_counter()
    for name, args in STEPS:
        t0 = time.perf_counter()
        print(f"== {name}: python {' '.join(args)}", flush=True)
        code = subprocess.run([sys.executable, *args], cwd=ROOT).returncode
        if code != 0:
            print(f"== FAILED at {name} (exit {code})", flush=True)
            return 1
        print(f"== {name} ok ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"== ALL STEPS PASSED ({time.perf_counter() - start:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
