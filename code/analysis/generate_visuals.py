# -*- coding: utf-8 -*-
"""Generate illustrative figures for the GitHub README gallery (pure OpenCV,
no matplotlib dependency). Produces three images under
qr-degraded-diagnosis/docs/images/:
  1. degradation_gallery.png  — one QR under 6 degradation conditions
  2. diagnosis_annotated.png  — geometric diagnosis on a hard P5 sample
  3. cross_decoder.png        — decoder accuracy comparison bar chart
"""
import os
import math
import numpy as np
import pandas as pd
import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "qr-degraded-diagnosis", "docs", "images")
os.makedirs(OUT_DIR, exist_ok=True)

MAN = os.path.join(BASE, "data", "e4_perspective_v1_1", "manifest.csv")
E5_CSV = os.path.join(BASE, "records", "e4_e5_adaptive_v1_1", "batch_e5_adaptive_timing.csv")
IMG_DIR = os.path.join(BASE, "data", "e4_perspective_v1_1", "images")

# OpenCV colour constants (BGR)
GREEN = (0, 255, 0)
BLUE = (255, 140, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_manifest():
    m = pd.read_csv(MAN, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    m["filename"] = m["image_path"].str.replace("\\", "/").str.split("/").str[-1]
    return m


def draw_text(img, text, org, color, size=0.55, thick=2):
    """Draw outlined text for legibility on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, BLACK, thick + 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick,
                cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Figure 1: degradation gallery
# ----------------------------------------------------------------------------
def fig_degradation_gallery():
    m = load_manifest()
    qr = m[m["sample_id"].str.startswith("E4QR_0001__")].sort_values("sample_id")
    cond_labels = {
        "C0_CLEAN": "CLEAN",
        "P1_MILD": "P1: mild perspective",
        "P2_MEDIUM": "P2: medium perspective",
        "P3_STRONG": "P3: strong perspective",
        "P4_MEDIUM_BLUR_S2": "P4: perspective + blur s=2",
        "P5_MEDIUM_BLUR_S3": "P5: perspective + blur s=3",
    }
    cell = 300
    header = 34
    tiles = []
    for _, row in qr.iterrows():
        img = cv2.imread(os.path.join(IMG_DIR, row["filename"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        head = np.full((header, cell, 3), 245, dtype=np.uint8)
        label = cond_labels.get(row["condition"], row["condition"])
        cv2.putText(head, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    BLACK, 2, cv2.LINE_AA)
        tiles.append(np.vstack([head, img]))

    rows = [np.hstack(tiles[0:3]), np.hstack(tiles[3:6])]
    # top banner
    banner = np.full((50, rows[0].shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(banner, "One QR code under six synthetic degradation conditions (V4)",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2, cv2.LINE_AA)
    canvas = np.vstack([banner, rows[0], rows[1]])
    out = os.path.join(OUT_DIR, "degradation_gallery.png")
    cv2.imwrite(out, canvas)
    print("[1/3] degradation_gallery.png", canvas.shape)


# ----------------------------------------------------------------------------
# Figure 2: geometric diagnosis annotation
# ----------------------------------------------------------------------------
def fig_diagnosis():
    m = load_manifest()
    e5 = pd.read_csv(E5_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    mm = m.merge(e5, on="filename", validate="one_to_one")
    t = mm[mm["sample_id"] == "E4QR_0084__P5_MEDIUM_BLUR_S3"].iloc[0]

    def f(name):
        return float(t[name])

    gt = (int(f("gt_alignment_dst_x")), int(f("gt_alignment_dst_y")))
    zx = (int(f("gt_zxing_prior_dst_x")), int(f("gt_zxing_prior_dst_y")))
    proj = (int(f("projective_prior_x")), int(f("projective_prior_y")))
    tl = (int(f("gt_finder_tl_dst_x")), int(f("gt_finder_tl_dst_y")))
    tr = (int(f("gt_finder_tr_dst_x")), int(f("gt_finder_tr_dst_y")))
    bl = (int(f("gt_finder_bl_dst_x")), int(f("gt_finder_bl_dst_y")))

    img = cv2.imread(os.path.join(IMG_DIR, t["filename"]))
    H, W = img.shape[:2]

    # three finder centres (green circle + cross)
    for p, name in [(tl, "TL"), (tr, "TR"), (bl, "BL")]:
        cv2.circle(img, p, 12, GREEN, 3, cv2.LINE_AA)
        cv2.line(img, (p[0] - 16, p[1]), (p[0] + 16, p[1]), GREEN, 2, cv2.LINE_AA)
        cv2.line(img, (p[0], p[1] - 16), (p[0], p[1] + 16), GREEN, 2, cv2.LINE_AA)
        draw_text(img, name, (p[0] + 14, p[1] - 10), GREEN)

    # GT alignment centre (blue circle)
    cv2.circle(img, gt, 14, BLUE, 3, cv2.LINE_AA)

    # ZXing affine prior (red X) — the wrong extrapolation
    s = 13
    cv2.line(img, (zx[0] - s, zx[1] - s), (zx[0] + s, zx[1] + s), RED, 4, cv2.LINE_AA)
    cv2.line(img, (zx[0] - s, zx[1] + s), (zx[0] + s, zx[1] - s), RED, 4, cv2.LINE_AA)

    # our projective prior (magenta star) — nearly on top of GT
    cv2.drawMarker(img, proj, MAGENTA, cv2.MARKER_STAR, 26, 3, cv2.LINE_AA)

    # zoom box around the bottom-right region
    cx, cy = (gt[0] + zx[0]) // 2, (gt[1] + zx[1]) // 2
    half = 60
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), YELLOW, 2)

    # title bar
    bar = np.full((46, W, 3), 255, dtype=np.uint8)
    cv2.putText(bar, "Geometric diagnosis on a hard sample (P5: strong perspective + blur s=3)",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLACK, 2, cv2.LINE_AA)

    # legend bar (below the image) — avoids label overlap inside the code
    legend = np.full((150, W, 3), 255, dtype=np.uint8)
    lx, y0 = 20, 34
    cv2.putText(legend, "Legend:", (lx, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                BLACK, 2, cv2.LINE_AA)

    # green finder
    cv2.circle(legend, (lx + 6, y0 + 28), 9, GREEN, 2, cv2.LINE_AA)
    cv2.putText(legend, "three finder centres (localised)", (lx + 26, y0 + 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, cv2.LINE_AA)
    # blue GT
    cv2.circle(legend, (lx + 6, y0 + 62), 9, BLUE, 2, cv2.LINE_AA)
    cv2.putText(legend, "GT alignment centre (ground truth)", (lx + 26, y0 + 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, cv2.LINE_AA)
    # red affine prior
    cv2.line(legend, (lx - 3, y0 + 88), (lx + 15, y0 + 106), RED, 3, cv2.LINE_AA)
    cv2.line(legend, (lx - 3, y0 + 106), (lx + 15, y0 + 88), RED, 3, cv2.LINE_AA)
    cv2.putText(legend, "ZXing affine prior - 4.5 modules off", (lx + 26, y0 + 101),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, cv2.LINE_AA)
    # magenta projective prior
    cv2.drawMarker(legend, (lx + 6, y0 + 128), MAGENTA, cv2.MARKER_STAR, 20, 2,
                   cv2.LINE_AA)
    cv2.putText(legend, "Ours: projective prior - 0.4 modules off", (lx + 26, y0 + 133),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, cv2.LINE_AA)

    canvas = np.vstack([bar, img, legend])
    out = os.path.join(OUT_DIR, "diagnosis_annotated.png")
    cv2.imwrite(out, canvas)
    print("[2/3] diagnosis_annotated.png", canvas.shape)


# ----------------------------------------------------------------------------
# Figure 3: cross-decoder bar chart
# ----------------------------------------------------------------------------
def fig_cross_decoder():
    W, H = 900, 560
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    data = [
        ("ZXing (Java) baseline", 57.67, RED),
        ("Ours (E5 + perturb)", 81.67, BLUE),
        ("OpenCV", 88.17, (0, 165, 255)),
        ("zxing-cpp", 99.83, (0, 180, 0)),
    ]
    n = len(data)
    left_margin, right_margin = 90, 40
    top_margin, bottom_margin = 70, 90
    plot_w = W - left_margin - right_margin
    plot_h = H - top_margin - bottom_margin
    max_val = 110.0

    def y_of(v):
        return top_margin + int(plot_h * (1 - v / max_val))

    # baseline axis at 0
    cv2.line(img, (left_margin, y_of(0)), (left_margin + plot_w, y_of(0)),
             BLACK, 2)
    # gridlines at 0/25/50/75/100
    for gv in [0, 25, 50, 75, 100]:
        y = y_of(gv)
        cv2.line(img, (left_margin, y), (left_margin + plot_w, y), (220, 220, 220), 1)
        cv2.putText(img, f"{gv}%", (left_margin - 48, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    slot = plot_w // n
    bw = int(slot * 0.52)
    for i, (name, val, color) in enumerate(data):
        x0 = left_margin + i * slot + (slot - bw) // 2
        y0 = y_of(val)
        ybase = y_of(0)
        cv2.rectangle(img, (x0, y0), (x0 + bw, ybase), color, -1)
        cv2.rectangle(img, (x0, y0), (x0 + bw, ybase), BLACK, 1)
        # value label
        cv2.putText(img, f"{val:.2f}%", (x0 + bw // 2 - 32, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, cv2.LINE_AA)
        # wrapped name (two lines)
        lines = name.split(" ")
        if len(lines) > 2:
            l1, l2 = " ".join(lines[:-1]), lines[-1]
        else:
            l1, l2 = " ".join(lines), ""
        cx = x0 + bw // 2
        cv2.putText(img, l1, (cx - len(l1) * 5, ybase + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)
        if l2:
            cv2.putText(img, l2, (cx - len(l2) * 5, ybase + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)

    cv2.putText(img, "Cross-decoder accuracy on E4 (600 perspective images)",
                (left_margin, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLACK, 2,
                cv2.LINE_AA)
    out = os.path.join(OUT_DIR, "cross_decoder.png")
    cv2.imwrite(out, img)
    print("[3/3] cross_decoder.png", img.shape)


if __name__ == "__main__":
    fig_degradation_gallery()
    fig_diagnosis()
    fig_cross_decoder()
    print("Done. Images written to:", OUT_DIR)
