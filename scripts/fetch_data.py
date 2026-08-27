"""Download the flag survey into data/, where the rest of the code expects it.

The imagery and annotations are distributed as a gated HuggingFace dataset
rather than in this repository, so access has to be granted to your account
first. Visit the dataset page, accept the terms, then authenticate once with
`hf auth login` (or set HF_TOKEN) before running this.

    python scripts/fetch_data.py

Writes data/flaglabel-dataset/ (photos + annotation JSONs) and
data/calibrations/ (per-photo fitted parameters). Re-running is cheap: the
HuggingFace cache means only changed files are downloaded.
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "toqi/camtrap-distance-flags"
DEST = Path("data")
SUBDIRS = ("flaglabel-dataset", "calibrations")


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    try:
        local = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                                  revision=None)
    except Exception as e:                       # gated, unauthenticated, offline
        sys.exit(f"could not download {REPO_ID}: {e}\n\n"
                 "This dataset is gated. Request access on its HuggingFace page, "
                 "then run `hf auth login` (or set HF_TOKEN) and try again.")

    DEST.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        src = Path(local) / sub
        if not src.is_dir():
            sys.exit(f"{REPO_ID} has no {sub}/ -- wrong repo or revision?")
        out = DEST / sub
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(src, out)
        n = sum(1 for _ in out.rglob("*") if _.is_file())
        print(f"{out}: {n} files")

    print("\nready. `python -m pytest tests/ -q` now includes the dataset-backed tests.")


if __name__ == "__main__":
    main()
