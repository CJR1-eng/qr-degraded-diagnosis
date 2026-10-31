#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the E4 controlled-perspective QR dataset, geometry-corrected V1.1.

Default output (relative to this script's directory): data/e4_perspective_v1_1

Examples from the project root:
  python generate_e4_perspective_v1_1.py --n-identities 6 --output-dir data/e4_perspective_v1_1_smoke
  python generate_e4_perspective_v1_1.py --n-identities 100 --output-dir data/e4_perspective_v1_1

The generator is intentionally non-destructive: it refuses to write into an
existing non-empty output directory unless --overwrite is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M


# ----------------------------- Frozen experiment config --------------------
IMAGE_SIZE = 640
BOX_SIZE = 8
QUIET_ZONE_MODULES = 4
CAMERA_FOCAL_PX = 800.0
CAMERA_DISTANCE = 800.0
CAMERA_CX = (IMAGE_SIZE - 1) / 2.0
CAMERA_CY = (IMAGE_SIZE - 1) / 2.0
RANDOM_SEED = 20261027

# Standard QR alignment-pattern center module coordinates for versions 4-6.
# ZXing's detector's lower-right alignment target is the final/final pair.
ALIGNMENT_POSITIONS = {
    4: [6, 26],
    5: [6, 30],
    6: [6, 34],
}

CONDITIONS = [
    ("C0_CLEAN", 0.0, "clean"),
    ("P1_MILD", 0.0, "mild"),
    ("P2_MEDIUM", 0.0, "medium"),
    ("P3_STRONG", 0.0, "strong"),
    ("P4_MEDIUM_BLUR_S2", 2.0, "medium"),
    ("P5_MEDIUM_BLUR_S3", 3.0, "medium"),
]

CSV_FIELDS = [
    "sample_id", "source_sample_id", "payload", "qr_version", "module_count",
    "condition", "yaw_deg", "pitch_deg", "roll_deg", "blur_sigma",
    "source_qr_path", "source_canvas_path", "image_path",
    "source_sha256", "source_canvas_sha256", "image_sha256",
    "H00", "H01", "H02", "H10", "H11", "H12", "H20", "H21", "H22",
    "gt_alignment_module_x", "gt_alignment_module_y",
    "gt_alignment_src_x", "gt_alignment_src_y",
    "gt_alignment_dst_x", "gt_alignment_dst_y",
    "gt_finder_tl_src_x", "gt_finder_tl_src_y", "gt_finder_tr_src_x", "gt_finder_tr_src_y",
    "gt_finder_bl_src_x", "gt_finder_bl_src_y",
    "gt_finder_tl_dst_x", "gt_finder_tl_dst_y", "gt_finder_tr_dst_x", "gt_finder_tr_dst_y",
    "gt_finder_bl_dst_x", "gt_finder_bl_dst_y",
    "gt_symbol_corners_src_json", "gt_symbol_corners_dst_json",
    "gt_module_basis_vx_x", "gt_module_basis_vx_y",
    "gt_module_basis_vy_x", "gt_module_basis_vy_y",
    "gt_affine_alignment_dst_x", "gt_affine_alignment_dst_y",
    "gt_zxing_prior_dst_x", "gt_zxing_prior_dst_y",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_camera_homography(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Map 640x640 source-canvas pixel coordinates to output pixel coordinates.

    Coordinate convention: image x points right, image y points down, and the
    QR plane is initially at camera depth D. R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
    The source-plane origin is the image center (319.5, 319.5).
    """
    R = rot_z(roll_deg) @ rot_y(yaw_deg) @ rot_x(pitch_deg)
    t = np.array([[0.0], [0.0], [CAMERA_DISTANCE]], dtype=np.float64)
    K = np.array(
        [[CAMERA_FOCAL_PX, 0.0, CAMERA_CX],
         [0.0, CAMERA_FOCAL_PX, CAMERA_CY],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # [x, y, 1] -> [X, Y, 1] centered coordinates on the QR plane.
    T = np.array(
        [[1.0, 0.0, -CAMERA_CX],
         [0.0, 1.0, -CAMERA_CY],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    plane_to_camera = np.column_stack((R[:, 0], R[:, 1], t[:, 0]))
    H = K @ plane_to_camera @ T
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("Generated an invalid/singular homography")
    return H / H[2, 2]


def project_points(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite projected coordinates")
    return out


def choose_angles(rng: np.random.Generator, level: str, shared_signs: tuple[int, int], roll: float) -> tuple[float, float, float]:
    ranges = {
        "mild": (8.0, 12.0),
        "medium": (18.0, 22.0),
        "strong": (28.0, 32.0),
    }
    if level == "clean":
        return 0.0, 0.0, 0.0
    lo, hi = ranges[level]
    yaw = shared_signs[0] * float(rng.uniform(lo, hi))
    pitch = shared_signs[1] * float(rng.uniform(lo, hi))
    return yaw, pitch, roll


def make_versions(n: int) -> list[int]:
    if n == 100:
        return [4] * 20 + [5] * 60 + [6] * 20
    # Smoke tests use a balanced repeating sequence; only the 100-ID run is
    # the pre-registered 20/60/20 version distribution.
    return [[4, 5, 6][i % 3] for i in range(n)]


def create_qr(version: int, payload: str) -> ImageLike:
    qr = qrcode.QRCode(
        version=version,
        error_correction=ERROR_CORRECT_M,
        box_size=BOX_SIZE,
        border=QUIET_ZONE_MODULES,
    )
    qr.add_data(payload, optimize=0)
    qr.make(fit=False)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    expected = (17 + 4 * version + 2 * QUIET_ZONE_MODULES) * BOX_SIZE
    if arr.shape != (expected, expected):
        raise RuntimeError(f"Unexpected QR raster size for V{version}: {arr.shape}, expected {(expected, expected)}")
    return arr


# A small alias avoids importing PIL types solely for annotations.
ImageLike = np.ndarray


def make_source_canvas(qr_arr: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = qr_arr.shape
    ox = (IMAGE_SIZE - w) // 2
    oy = (IMAGE_SIZE - h) // 2
    if ox < 0 or oy < 0:
        raise ValueError("QR raster does not fit in the output canvas")
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
    canvas[oy:oy + h, ox:ox + w] = qr_arr
    return canvas, ox, oy


def build_geometry(version: int, offset_x: int, offset_y: int, H: np.ndarray) -> dict[str, Any]:
    N = 17 + 4 * version
    p = ALIGNMENT_POSITIONS[version][-1]  # lower-right Alignment Pattern used by ZXing

    # Module-center locations in the source 640x640 canvas.
    def module_center(mx: float, my: float) -> tuple[float, float]:
        return (
            offset_x + (QUIET_ZONE_MODULES + mx + 0.5) * BOX_SIZE,
            offset_y + (QUIET_ZONE_MODULES + my + 0.5) * BOX_SIZE,
        )

    A_src = np.array(module_center(float(p), float(p)), dtype=np.float64)
    # module_center() accepts a zero-based module index and adds the 0.5
    # center offset internally. Finder centers are therefore at module indices
    # 3 and N-4, equivalent to continuous module coordinates 3.5 and N-3.5.
    F_src = np.array(
        [module_center(3, 3), module_center(N - 4, 3), module_center(3, N - 4)],
        dtype=np.float64,
    )
    A_dst = project_points(H, A_src.reshape(1, 2))[0]
    F_dst = project_points(H, F_src)

    # QR symbol outer boundary, excluding the four-module quiet zone.
    x0 = offset_x + QUIET_ZONE_MODULES * BOX_SIZE
    y0 = offset_y + QUIET_ZONE_MODULES * BOX_SIZE
    x1 = offset_x + (QUIET_ZONE_MODULES + N) * BOX_SIZE
    y1 = offset_y + (QUIET_ZONE_MODULES + N) * BOX_SIZE
    corners_src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    corners_dst = project_points(H, corners_src)

    # Local image-space basis for one QR module in +x and +y at Alignment.
    basis_src = np.array(
        [A_src, A_src + [BOX_SIZE, 0.0], A_src + [0.0, BOX_SIZE]],
        dtype=np.float64,
    )
    basis_dst = project_points(H, basis_src)
    vx = basis_dst[1] - basis_dst[0]
    vy = basis_dst[2] - basis_dst[0]
    B = np.column_stack((vx, vy))
    if abs(np.linalg.det(B)) < 1e-8:
        raise ValueError("Local module basis is singular")

    # Affine prediction from the three projected Finder centers. ZXing moves
    # three modules inward from the inferred bottom-right Finder center.
    # This is a geometric reference, not the exact projective Alignment truth.
    alpha = 1.0 - 3.0 / (N - 7)
    beta = alpha
    F_tl, F_tr, F_bl = F_dst
    A_affine = F_tl + alpha * (F_tr - F_tl) + beta * (F_bl - F_tl)
    # Detector.java casts the floating-point prior to int before searching.
    # All generated coordinates are positive, so truncation equals floor here.
    A_zxing_prior = np.array([math.trunc(A_affine[0]), math.trunc(A_affine[1])], dtype=np.float64)

    return {
        "N": N,
        "align_module": p,
        "A_src": A_src,
        "A_dst": A_dst,
        "F_src": F_src,
        "F_dst": F_dst,
        "corners_src": corners_src,
        "corners_dst": corners_dst,
        "vx": vx,
        "vy": vy,
        "A_affine": A_affine,
        "A_zxing_prior": A_zxing_prior,
    }


def within_canvas(points: np.ndarray, margin: float = 2.0) -> bool:
    return bool(
        np.all(points[:, 0] >= margin)
        and np.all(points[:, 0] <= IMAGE_SIZE - 1 - margin)
        and np.all(points[:, 1] >= margin)
        and np.all(points[:, 1] <= IMAGE_SIZE - 1 - margin)
    )


def json_compact(obj: Any) -> str:
    return json.dumps(np.asarray(obj).round(8).tolist(), ensure_ascii=False, separators=(",", ":"))


def safe_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_dataset(root: Path, expected_n: int) -> dict[str, Any]:
    manifest_path = root / "manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    expected_rows = expected_n * len(CONDITIONS)
    assert len(rows) == expected_rows, (len(rows), expected_rows)
    ids = sorted({r["source_sample_id"] for r in rows})
    assert len(ids) == expected_n, len(ids)
    assert len({r["payload"] for r in rows}) == expected_n

    cond_counts: dict[str, int] = {}
    by_key = {(r["source_sample_id"], r["condition"]): r for r in rows}
    for cond, _, _ in CONDITIONS:
        subset = [r for r in rows if r["condition"] == cond]
        cond_counts[cond] = len(subset)
        assert len(subset) == expected_n, (cond, len(subset))

    for row in rows:
        source_qr_path = root / row["source_qr_path"]
        source_canvas_path = root / row["source_canvas_path"]
        assert source_qr_path.is_file(), str(source_qr_path)
        assert source_canvas_path.is_file(), str(source_canvas_path)
        assert sha256_file(source_qr_path) == row["source_sha256"], f"Source QR SHA mismatch: {source_qr_path}"
        assert sha256_file(source_canvas_path) == row["source_canvas_sha256"], f"Source canvas SHA mismatch: {source_canvas_path}"

        image_path = root / row["image_path"]
        assert image_path.is_file(), str(image_path)
        assert sha256_file(image_path) == row["image_sha256"], f"SHA mismatch: {image_path}"
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        assert img is not None and img.shape == (IMAGE_SIZE, IMAGE_SIZE), f"Bad image shape: {image_path}"
        assert img.dtype == np.uint8 and img.ndim == 2, f"Bad image type: {image_path}"
        H = np.array([[float(row[f"H{i}{j}"]) for j in range(3)] for i in range(3)], dtype=np.float64)
        assert abs(np.linalg.det(H)) > 1e-12
        src_A = np.array([[float(row["gt_alignment_src_x"]), float(row["gt_alignment_src_y"])]])
        dst_A_expected = np.array([float(row["gt_alignment_dst_x"]), float(row["gt_alignment_dst_y"])])
        dst_A_actual = project_points(H, src_A)[0]
        assert np.linalg.norm(dst_A_actual - dst_A_expected) < 1e-4
        assert 0 <= dst_A_expected[0] < IMAGE_SIZE and 0 <= dst_A_expected[1] < IMAGE_SIZE

        src_F = np.array([
            [float(row["gt_finder_tl_src_x"]), float(row["gt_finder_tl_src_y"])],
            [float(row["gt_finder_tr_src_x"]), float(row["gt_finder_tr_src_y"])],
            [float(row["gt_finder_bl_src_x"]), float(row["gt_finder_bl_src_y"])],
        ])
        dst_F_expected = np.array([
            [float(row["gt_finder_tl_dst_x"]), float(row["gt_finder_tl_dst_y"])],
            [float(row["gt_finder_tr_dst_x"]), float(row["gt_finder_tr_dst_y"])],
            [float(row["gt_finder_bl_dst_x"]), float(row["gt_finder_bl_dst_y"])],
        ])
        assert np.max(np.linalg.norm(project_points(H, src_F) - dst_F_expected, axis=1)) < 1e-4

        n = int(row["module_count"])
        alpha = 1.0 - 3.0 / (n - 7)
        affine_expected = dst_F_expected[0] + alpha * (
            dst_F_expected[1] - dst_F_expected[0] + dst_F_expected[2] - dst_F_expected[0])
        affine_actual = np.array([
            float(row["gt_affine_alignment_dst_x"]), float(row["gt_affine_alignment_dst_y"])])
        assert np.linalg.norm(affine_actual - affine_expected) < 1e-4
        zxing_prior = np.array([
            float(row["gt_zxing_prior_dst_x"]), float(row["gt_zxing_prior_dst_y"])])
        assert np.array_equal(zxing_prior, np.trunc(affine_expected))

        src_corners = np.asarray(json.loads(row["gt_symbol_corners_src_json"]), dtype=np.float64)
        dst_corners = np.asarray(json.loads(row["gt_symbol_corners_dst_json"]), dtype=np.float64)
        assert np.max(np.linalg.norm(project_points(H, src_corners) - dst_corners, axis=1)) < 1e-4
        if row["condition"] == "C0_CLEAN":
            assert np.linalg.norm(dst_A_expected - affine_actual) < 1e-8
            assert np.array_equal(zxing_prior, dst_A_expected)

    # P2/P4/P5 must share identical homography per identity.
    for sid in ids:
        r2 = by_key[(sid, "P2_MEDIUM")]
        h2 = [float(r2[f"H{i}{j}"]) for i in range(3) for j in range(3)]
        for cond in ("P4_MEDIUM_BLUR_S2", "P5_MEDIUM_BLUR_S3"):
            rr = by_key[(sid, cond)]
            hh = [float(rr[f"H{i}{j}"]) for i in range(3) for j in range(3)]
            assert np.allclose(h2, hh, rtol=0.0, atol=1e-12), f"H mismatch: {sid} {cond}"

    return {
        "rows": len(rows),
        "identities": len(ids),
        "condition_counts": cond_counts,
        "sha256_verified": {"source_qr": expected_n, "source_canvas": expected_n, "final_images": len(rows)},
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "dtype": "uint8 grayscale",
        "validation": "PASS",
    }


def generate(args: argparse.Namespace) -> None:
    root = Path(args.output_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root

    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {root}\n"
            "Choose a new output directory or pass --overwrite explicitly."
        )
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("source_qr", "source_canvas", "images"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    versions = make_versions(args.n_identities)
    rows: list[dict[str, Any]] = []
    rng_master = np.random.default_rng(args.seed)

    for idx, version in enumerate(versions, start=1):
        sid = f"E4QR_{idx:04d}"
        payload = f"E4-QR-{idx:04d}-SEED-{args.seed}"
        qr_arr = create_qr(version, payload)
        canvas, ox, oy = make_source_canvas(qr_arr)

        qr_path = root / "source_qr" / f"{sid}.png"
        canvas_path = root / "source_canvas" / f"{sid}_640.png"
        if not cv2.imwrite(str(qr_path), qr_arr):
            raise IOError(f"Could not write {qr_path}")
        if not cv2.imwrite(str(canvas_path), canvas):
            raise IOError(f"Could not write {canvas_path}")
        qr_hash = sha256_file(qr_path)
        canvas_hash = sha256_file(canvas_path)

        # Stable per-identity RNG: all six conditions can be regenerated.
        identity_rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        signs = (int(identity_rng.choice([-1, 1])), int(identity_rng.choice([-1, 1])))
        shared_roll = float(identity_rng.uniform(-5.0, 5.0))

        angle_cache: dict[str, tuple[float, float, float]] = {}
        for cond, blur_sigma, level in CONDITIONS:
            if level not in angle_cache:
                angle_cache[level] = choose_angles(identity_rng, level, signs, shared_roll)
            yaw, pitch, roll = angle_cache[level]
            if level == "clean":
                H = np.eye(3, dtype=np.float64)
            else:
                H = make_camera_homography(yaw, pitch, roll)

            geometry = build_geometry(version, ox, oy, H)
            if not within_canvas(geometry["corners_dst"], margin=1.0):
                raise RuntimeError(
                    f"Projected QR symbol is clipped for {sid}/{cond}: "
                    f"{geometry['corners_dst'].round(2).tolist()}"
                )

            warped = cv2.warpPerspective(
                canvas,
                H,
                (IMAGE_SIZE, IMAGE_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            if blur_sigma > 0:
                warped = cv2.GaussianBlur(
                    warped, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma,
                    borderType=cv2.BORDER_REPLICATE,
                )
            warped = np.ascontiguousarray(warped, dtype=np.uint8)

            image_name = f"{sid}__{cond}.png"
            image_path = root / "images" / image_name
            if not cv2.imwrite(str(image_path), warped):
                raise IOError(f"Could not write {image_path}")
            image_hash = sha256_file(image_path)

            row: dict[str, Any] = {
                "sample_id": f"{sid}__{cond}",
                "source_sample_id": sid,
                "payload": payload,
                "qr_version": version,
                "module_count": geometry["N"],
                "condition": cond,
                "yaw_deg": f"{yaw:.8f}",
                "pitch_deg": f"{pitch:.8f}",
                "roll_deg": f"{roll:.8f}",
                "blur_sigma": f"{blur_sigma:.4f}",
                "source_qr_path": str(Path("source_qr") / qr_path.name),
                "source_canvas_path": str(Path("source_canvas") / canvas_path.name),
                "image_path": str(Path("images") / image_name),
                "source_sha256": qr_hash,
                "source_canvas_sha256": canvas_hash,
                "image_sha256": image_hash,
                "gt_alignment_module_x": geometry["align_module"],
                "gt_alignment_module_y": geometry["align_module"],
                "gt_alignment_src_x": f"{geometry['A_src'][0]:.10f}",
                "gt_alignment_src_y": f"{geometry['A_src'][1]:.10f}",
                "gt_alignment_dst_x": f"{geometry['A_dst'][0]:.10f}",
                "gt_alignment_dst_y": f"{geometry['A_dst'][1]:.10f}",
                "gt_finder_tl_src_x": f"{geometry['F_src'][0,0]:.10f}",
                "gt_finder_tl_src_y": f"{geometry['F_src'][0,1]:.10f}",
                "gt_finder_tr_src_x": f"{geometry['F_src'][1,0]:.10f}",
                "gt_finder_tr_src_y": f"{geometry['F_src'][1,1]:.10f}",
                "gt_finder_bl_src_x": f"{geometry['F_src'][2,0]:.10f}",
                "gt_finder_bl_src_y": f"{geometry['F_src'][2,1]:.10f}",
                "gt_finder_tl_dst_x": f"{geometry['F_dst'][0,0]:.10f}",
                "gt_finder_tl_dst_y": f"{geometry['F_dst'][0,1]:.10f}",
                "gt_finder_tr_dst_x": f"{geometry['F_dst'][1,0]:.10f}",
                "gt_finder_tr_dst_y": f"{geometry['F_dst'][1,1]:.10f}",
                "gt_finder_bl_dst_x": f"{geometry['F_dst'][2,0]:.10f}",
                "gt_finder_bl_dst_y": f"{geometry['F_dst'][2,1]:.10f}",
                "gt_symbol_corners_src_json": json_compact(geometry["corners_src"]),
                "gt_symbol_corners_dst_json": json_compact(geometry["corners_dst"]),
                "gt_module_basis_vx_x": f"{geometry['vx'][0]:.10f}",
                "gt_module_basis_vx_y": f"{geometry['vx'][1]:.10f}",
                "gt_module_basis_vy_x": f"{geometry['vy'][0]:.10f}",
                "gt_module_basis_vy_y": f"{geometry['vy'][1]:.10f}",
                "gt_affine_alignment_dst_x": f"{geometry['A_affine'][0]:.10f}",
                "gt_affine_alignment_dst_y": f"{geometry['A_affine'][1]:.10f}",
                "gt_zxing_prior_dst_x": f"{geometry['A_zxing_prior'][0]:.10f}",
                "gt_zxing_prior_dst_y": f"{geometry['A_zxing_prior'][1]:.10f}",
            }
            for i in range(3):
                for j in range(3):
                    row[f"H{i}{j}"] = f"{H[i, j]:.14f}"
            rows.append(row)

        print(f"Generated {sid} (V{version}) [{idx}/{args.n_identities}]")

    safe_write_csv(root / "manifest.csv", rows)
    config = {
        "dataset_name": "E4_Perspective_Stress_Test_V1_1",
        "seed": args.seed,
        "n_identities": args.n_identities,
        "n_images": len(rows),
        "version_distribution": {str(v): versions.count(v) for v in sorted(set(versions))},
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "pixel_format": "8-bit grayscale PNG",
        "error_correction": "M",
        "box_size": BOX_SIZE,
        "quiet_zone_modules": QUIET_ZONE_MODULES,
        "camera": {
            "focal_px": CAMERA_FOCAL_PX,
            "distance": CAMERA_DISTANCE,
            "cx": CAMERA_CX,
            "cy": CAMERA_CY,
            "rotation_order": "Rz(roll) @ Ry(yaw) @ Rx(pitch)",
            "yaw_pitch_ranges_deg": {"mild": [8, 12], "medium": [18, 22], "strong": [28, 32]},
            "roll_range_deg": [-5, 5],
        },
        "conditions": [
            {"name": c, "blur_sigma": s, "perspective_level": lvl}
            for c, s, lvl in CONDITIONS
        ],
        "ground_truth": {
            "alignment_target": "lower-right Alignment Pattern center",
            "alignment_module_positions": ALIGNMENT_POSITIONS,
            "finder_centers_module_coordinates": {
                "TL": [3.5, 3.5], "TR": ["N-3.5", 3.5], "BL": [3.5, "N-3.5"]
            },
            "homography_direction": "source_640_canvas_to_output_640_canvas",
            "module_basis": "local projected +x and +y one-module displacement at the Alignment center",
        },
        "notes": [
            "P2/P4/P5 share the exact same per-identity Homography; P4/P5 apply blur after perspective warping.",
            "The 100-identity release uses V4/V5/V6 counts 20/60/20. Smoke tests cycle versions and are not the final dataset.",
            "The image canvas is 640x640; the QR raster is centered before perspective projection.",
            "V1.1 fixes Finder centers to zero-based module indices 3 and N-4 before adding the 0.5 center offset.",
            "gt_zxing_prior_dst_* reproduces Detector.java integer truncation of the three-Finder affine prior.",
        ],
    }
    (root / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = validate_dataset(root, args.n_identities)
    summary = {
        "dataset_name": config["dataset_name"],
        "output_dir": str(root.resolve()),
        "validation": validation,
        "manifest_sha256": sha256_file(root / "manifest.csv"),
        "config_sha256": sha256_file(root / "experiment_config.json"),
    }
    (root / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nE4 generation and validation: PASS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled E4 perspective QR dataset.")
    parser.add_argument("--n-identities", type=int, default=100, help="100 for the formal dataset; use 3-6 for smoke testing")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=str, default="data/e4_perspective_v1_1")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory")
    args = parser.parse_args()
    if args.n_identities < 1:
        parser.error("--n-identities must be >= 1")
    return args


if __name__ == "__main__":
    try:
        generate(parse_args())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
