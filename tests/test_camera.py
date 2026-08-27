"""The camera module's file conventions, on a two-photo fixture."""
import json

from src.calibration.camera import load_cameras


def _annotation(root, site, image):
    d = root / site
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{image.rsplit('.', 1)[0]}.json").write_text(json.dumps(
        dict(site=site, image=image, image_w=1920, image_h=1080)))


def test_cameras_pair_their_own_photos_and_name_their_files(tmp_path):
    for site, image in (("B_CAM02", "IMG_0002.JPG"), ("A_CAM01", "IMG_0001.JPG"), ("A_CAM01", "IMG_0009.JPG")):
        _annotation(tmp_path, site, image)
    cams = load_cameras(root=tmp_path)
    assert list(cams) == ["A_CAM01", "B_CAM02"]                     # sorted
    a = cams["A_CAM01"]
    first, second = a.photos
    assert a.references_for(first) == [second] and a.references_for(second) == [first]
    assert cams["B_CAM02"].references_for(cams["B_CAM02"].photos[0]) == []
    assert a.photo_path(first) == tmp_path / "A_CAM01" / "IMG_0001.JPG"
    assert a.distance_map(first).name == "A_CAM01__IMG_0001.npz"     # keyed by the photo itself
    assert a.aligned_prompt(second, "prompts").as_posix() == "prompts/A_CAM01__IMG_0009.npz"  # keyed by the target
    assert list(load_cameras({"B_CAM02"}, root=tmp_path)) == ["B_CAM02"]
