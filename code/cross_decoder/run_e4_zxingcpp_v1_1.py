from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import zxingcpp

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
IMG_DIR = ROOT / "data" / "e4_perspective_v1_1" / "images"
OUT = ROOT / "records" / "e4_zxingcpp_v1_1"


def main() -> None:
    man = pd.read_csv(MAN, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    man["filename"] = man["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in man.iterrows():
        fn = r["filename"]
        img = cv2.imread(str(IMG_DIR / fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append((fn, "FAIL", "READ_ERROR"))
            continue
        try:
            results = zxingcpp.read_barcodes(img)
            if results and results[0].text:
                rows.append((fn, "DECODED", results[0].text))
            else:
                rows.append((fn, "FAIL", ""))
        except Exception as e:
            rows.append((fn, "FAIL", type(e).__name__))

    df = pd.DataFrame(rows, columns=["filename", "zxingcpp_status", "zxingcpp_payload"])
    m = man[["filename", "sample_id", "condition", "payload"]].merge(df, on="filename", validate="one_to_one")
    m["correct"] = m["zxingcpp_status"].eq("DECODED") & m["zxingcpp_payload"].eq(m["payload"])
    m["wrong"] = m["zxingcpp_status"].eq("DECODED") & m["zxingcpp_payload"].ne(m["payload"])
    m.to_csv(OUT / "batch_zxingcpp.csv", index=False, encoding="utf-8-sig")

    print("=== zxing-cpp 在 E4（600 张）上的结果 ===")
    print(f"总正确: {int(m['correct'].sum())}/600 = {m['correct'].mean()*100:.2f}%")
    print(f"错误载荷: {int(m['wrong'].sum())}")
    g = m.groupby("condition").agg(n=("correct", "size"), correct=("correct", "sum")).reset_index()
    g["rate"] = (g["correct"] / g["n"] * 100).round(2)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
