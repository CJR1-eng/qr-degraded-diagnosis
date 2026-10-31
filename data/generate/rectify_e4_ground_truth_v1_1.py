from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "e4_perspective_v1_1"
OUTPUT = ROOT / "records" / "e4_gt_rectified_v1_1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    manifest = pd.read_csv(DATASET / "manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    images = OUTPUT / "images"
    images.mkdir(parents=True, exist_ok=False)
    for row in manifest.to_dict("records"):
        image = cv2.imread(str(DATASET / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(row["image_path"])
        H = np.array([[float(row[f"H{i}{j}"]) for j in range(3)] for i in range(3)])
        rectified = cv2.warpPerspective(
            image, np.linalg.inv(H), (640, 640), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        filename = Path(row["image_path"]).name
        if not cv2.imwrite(str(images / filename), rectified):
            raise IOError(filename)
    manifest.to_csv(OUTPUT / "manifest.csv", index=False, encoding="utf-8-sig")
    config = {
        "method": "Ground-Truth inverse-Homography image rectification",
        "purpose": "Geometry sanity check and upper-reference preprocessing; not a deployable method and not a grid-sampling oracle.",
        "rows": len(manifest),
        "note": "Blur is not removed. P4/P5 may still fail after perfect geometric rectification.",
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
