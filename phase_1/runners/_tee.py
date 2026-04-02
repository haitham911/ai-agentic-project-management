"""Shared helper: tees stdout to both terminal and a file."""
import sys
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class Tee:
    """Writes to both stdout and a file simultaneously."""
    def __init__(self, file):
        self._file = file
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def run_with_tee(filename, fn):
    """Run fn() while teeing all stdout output to results/<filename>."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            fn()
        finally:
            sys.stdout = sys.stdout._stdout
    print(f"Results saved to results/{filename}")
