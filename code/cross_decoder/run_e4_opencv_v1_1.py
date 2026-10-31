from __future__ import annotations

import csv
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
IMG_DIR = ROOT / "data" / "e4_perspective_v1_1" / "images"
OUT = ROOT / "records" / "e4_opencv_v1_1"


def main() -> None:
    man = pd.read_csv(MAN, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    man["filename"] = man["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]

    OUT.mkdir(parents=True, exist_ok=True)
    detector = cv2.QRCodeDetector()
    rows = []

    for _, r in man.iterrows():
        fn = r["filename"]
        img = cv2.imread(str(IMG_DIR / fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append((fn, "FAIL", "READ_ERROR"))
            continue
        data, _, _ = detector.detectAndDecode(img)
        if data:
            rows.append((fn, "DECODED", data))
        else:
            rows.append((fn, "FAIL", ""))

    df = pd.DataFrame(rows, columns=["filename", "opencv_status", "opencv_payload"])
    m = man[["filename", "sample_id", "condition", "payload"]].merge(df, on="filename", validate="one_to_one")
    m["correct"] = (m["opencv_status"].eq("DECODED") & m["opencv_payload"].eq(m["payload"]))
    m["wrong"] = m["opencv_status"].eq("DECODED") & m["opencv_payload"].ne(m["payload"])

    m.to_csv(OUT / "batch_opencv.csv", index=False, encoding="utf-8-sig")

    print("=== OpenCV QRCodeDetector 在 E4（600 张）上的结果 ===")
    print(f"总正确: {int(m['correct'].sum())}/600 = {m['correct'].mean()*100:.2f}%")
    print(f"错误载荷: {int(m['wrong'].sum())}")
    print()
    print("=== 分条件 ===")
    g = m.groupby("condition").agg(n=("correct", "size"), correct=("correct", "sum")).reset_index()
    g["rate"] = (g["correct"] / g["n"] * 100).round(2)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
