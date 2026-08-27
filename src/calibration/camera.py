"""A surveyed camera: its flag photos, their calibrations, and the file
conventions that tie a photo to its distance map and aligned prompts."""
from dataclasses import dataclass, field
from pathlib import Path

from src.calibration.data import PHOTOS_DIR, TARGETS_DIR, load_dataset
from src.calibration.ground_plane import load_calibration


@dataclass
class Camera:
    site: str
    photos: list = field(default_factory=list)   # PhotoData, in annotation-file order
    root: Path = PHOTOS_DIR

    def photo_path(self, photo):
        return self.root / self.site / photo.image

    def calibration(self, photo):
        """GroundPlaneFit for one flag photo, or None if missing or failed."""
        return load_calibration(self.site, Path(photo.image).stem)

    def references_for(self, photo):
        """The camera's other flag photos: candidates for `photo`'s reference."""
        return [p for p in self.photos if p.image != photo.image]

    def distance_map(self, photo):
        """D_R of `photo` in its own frame (targets_d entry, keyed by the photo)."""
        return TARGETS_DIR / f"{self.site}__{Path(photo.image).stem}.npz"

    def aligned_prompt(self, target, prompts_dir):
        """The sibling's D_R aligned into `target`'s frame (keyed by the TARGET)."""
        return Path(prompts_dir) / f"{self.site}__{Path(target.image).stem}.npz"


def load_cameras(sites=None, root=PHOTOS_DIR):
    """site -> Camera, sorted by site, optionally restricted to `sites`."""
    cams = {}
    for ph in load_dataset(root):
        if sites is None or ph.site in sites:
            cams.setdefault(ph.site, Camera(ph.site, root=root)).photos.append(ph)
    return dict(sorted(cams.items()))
