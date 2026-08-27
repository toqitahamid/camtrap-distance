"""Fit all flag photos, run QC, write data/calibrations/ and outputs/qc/."""
import argparse
import json
from pathlib import Path

from src.calibration.data import PHOTOS_DIR, load_dataset
from src.calibration.ground_plane import GroundPlaneFit
from src.calibration.qc import cross_photo, leave_one_flag_out, monotonicity
from src.calibration import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(PHOTOS_DIR))
    ap.add_argument("--out", default=".")
    ap.add_argument("--skip-overlays", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    photos = load_dataset(args.dataset)
    print(f"{len(photos)} photos")

    cv_rows, mono_rows, skipped_rows, insufficient = [], [], [], []
    site_models = {}
    for ph in photos:
        stem = Path(ph.image).stem
        m = GroundPlaneFit.fit(ph)
        site_models.setdefault(ph.site, []).append(m)
        cpath = out / "data" / "calibrations" / ph.site / f"{stem}.json"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(m.to_dict(), indent=1))
        if not m.ok:
            insufficient.append(f"{ph.site}/{ph.image}")
        for row in monotonicity(ph):
            mono_rows.append({"site": ph.site, "image": ph.image, **row})
        for sk in ph.skipped:
            skipped_rows.append({"site": ph.site, "image": ph.image, **sk})
        cv_rows.extend(leave_one_flag_out(ph))
        if not args.skip_overlays:
            img = Path(args.dataset) / ph.site / ph.image
            if img.exists() and m.ok:
                report.overlay(ph, m, img,
                               out / "outputs" / "qc" / "overlays" / f"{ph.site}__{stem}.png")
        print(f"  {ph.site}/{ph.image}: ok={m.ok}")

    cross = cross_photo(site_models)
    report.write_csv(out / "outputs" / "qc" / "loo_cv.csv", cv_rows)
    report.write_csv(out / "outputs" / "qc" / "monotonicity.csv", mono_rows)
    report.write_csv(out / "outputs" / "qc" / "skipped_leans.csv", skipped_rows)
    report.write_csv(out / "outputs" / "qc" / "cross_photo.csv", cross)
    (out / "outputs" / "qc" / "summary.md").write_text(
        report.summarize(cv_rows, mono_rows, skipped_rows, cross, insufficient))
    print("wrote outputs/qc/summary.md")


if __name__ == "__main__":
    main()
