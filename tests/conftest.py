"""Skip the dataset-backed tests when the flag survey has not been fetched.

The imagery and annotations live in a separate gated dataset, so a fresh clone
has code but no `data/`. The synthetic tests (ground plane, QC, calibration
data handling, roll identifiability) still run and still catch real regressions;
only the two files that read real photos are skipped.
"""
from pathlib import Path

DATASET_ROOT = Path("data/flaglabel-dataset")
NEEDS_DATA = {"test_paper_numbers.py", "test_alignment.py"}

collect_ignore = []
if not DATASET_ROOT.is_dir():
    collect_ignore = sorted(NEEDS_DATA)


def pytest_report_header(config):
    if not DATASET_ROOT.is_dir():
        return (f"flag survey not found at {DATASET_ROOT} -- skipping "
                f"{', '.join(sorted(NEEDS_DATA))}. "
                "Fetch it with: python scripts/fetch_data.py")
    return None
