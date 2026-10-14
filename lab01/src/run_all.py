"""Run Lab 01: print library versions, then train, measure and check deployment budgets."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    out = subprocess.run([sys.executable, str(SRC / "print_versions.py")], capture_output=True, text=True)
    (RESULTS / "versions.txt").write_text(out.stdout, encoding="utf-8")
    print(out.stdout)
    r = subprocess.run([sys.executable, str(SRC / "baselines.py")], cwd=SRC)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
