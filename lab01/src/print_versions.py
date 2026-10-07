"""Lab 01, step 1: print Python and every pinned library version.

Usage (from repo root, with .venv active):
    python lab01/src/print_versions.py > lab01/results/versions.txt
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"


def main() -> None:
    print(f"Python {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform {platform.platform()}")
    print("-" * 40)

    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, pinned = line.partition("==")
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "NOT INSTALLED"
        matches = installed != "NOT INSTALLED" and Version(installed) == Version(pinned)
        flag = "" if matches else f"   <-- expected {pinned}"
        print(f"{name}=={installed}{flag}")

    try:
        import torch

        print("-" * 40)
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
