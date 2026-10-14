import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"
STEPS = ["sanity_checks.py", "gradient_check.py", "train.py", "timing.py"]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    versions = subprocess.run([sys.executable, str(SRC.parents[1] / "lab01" / "src" / "print_versions.py")],
                              capture_output=True, text=True)
    (RESULTS / "versions.txt").write_text(versions.stdout, encoding="utf-8")
    for step in STEPS:
        print(f"\n {step}", flush=True)
        r = subprocess.run([sys.executable, str(SRC / step)], cwd=SRC)
        if r.returncode != 0:
            sys.exit(f"{step} failed")
    print("\nAll steps done. Results are in", RESULTS)


if __name__ == "__main__":
    main()
