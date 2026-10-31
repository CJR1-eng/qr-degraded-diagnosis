#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""退化二维码高斯模糊参数扫描 V1.1（先 dry-run，确认后再正式生成）

设计：
1. 固定保留 V1.0 的 20 个 source_sample_id；
2. 按 QR Version 分层补充 80 个身份，使总数达到 Version4/5/6 = 15/80/5；
3. 每个身份生成 1 张灰度 CLEAN + 9 张高斯模糊图，共 1000 张；
4. 所有输出统一为 8-bit 单通道灰度 PNG；
5. 不覆盖原始数据和 V1.0 数据；
6. 默认先运行 --dry-run，只检查抽样和输入，不生成图片。

在项目根目录运行：
python pretest_gaussian_blur_v1_1.py --dry-run
确认输出后再运行：
python pretest_gaussian_blur_v1_1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请在当前 Python 环境执行：python -m pip install opencv-python numpy pandas"
    ) from exc


# -----------------------------
# 固定实验配置（V1.1 参数扫描计划）
# -----------------------------
SOURCE_MANIFEST = Path("records/clean_400/manifest.csv")
SOURCE_IMAGE_DIR = Path("records/clean_400/qr_images")
V1_CONFIG = Path("records/pretest_gaussian_blur_v1/experiment_config.json")
OUTPUT_DIR = Path("data/pretest_gaussian_blur_v1_1")

SELECTION_SEED = 20261020
TARGET_VERSION_COUNTS = {4: 15, 5: 80, 6: 5}

# ID、Sigma、Kernel Size。K = 2 * ceil(3*sigma) + 1。
BLUR_CONFIGS = [
    {"id": "S1p50", "sigma": 1.50, "ksize": 11},
    {"id": "S1p75", "sigma": 1.75, "ksize": 13},
    {"id": "S2p00", "sigma": 2.00, "ksize": 13},
    {"id": "S2p25", "sigma": 2.25, "ksize": 15},
    {"id": "S2p50", "sigma": 2.50, "ksize": 17},
    {"id": "S2p75", "sigma": 2.75, "ksize": 19},
    {"id": "S3p00", "sigma": 3.00, "ksize": 19},
    {"id": "S3p50", "sigma": 3.50, "ksize": 23},
    {"id": "S4p00", "sigma": 4.00, "ksize": 25},
]

REQUIRED_COLUMNS = {
    "sample_id", "filename", "payload", "seed", "qr_version",
    "error_correction", "box_size", "border_modules", "width_px", "height_px"
}


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 配置：{path}")
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层结构不是对象：{path}")
    return data


def load_source_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"找不到原始 Manifest：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"原始 Manifest 缺少字段：{missing}")
    if df.empty:
        raise ValueError("原始 Manifest 为空。")
    if df["sample_id"].isna().any() or df["filename"].isna().any():
        raise ValueError("原始 Manifest 的 sample_id 或 filename 存在空值。")
    if df["sample_id"].astype(str).duplicated().any():
        raise ValueError("原始 Manifest 存在重复 sample_id。")
    if df["filename"].astype(str).duplicated().any():
        raise ValueError("原始 Manifest 存在重复 filename。")
    return df


def choose_identities(df: pd.DataFrame, v1_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[int, int]]:
    """保留 V1.0 的20个身份，再按版本分层补足到100个。"""
    v1_ids_raw = v1_config.get("selected_source_sample_ids")
    if not isinstance(v1_ids_raw, list) or not v1_ids_raw:
        raise ValueError("V1.0 配置中没有有效的 selected_source_sample_ids。")

    v1_ids = [str(item) for item in v1_ids_raw]
    if len(v1_ids) != len(set(v1_ids)):
        raise ValueError("V1.0 身份清单存在重复 sample_id。")

    source_ids = set(df["sample_id"].astype(str))
    missing_v1 = sorted(set(v1_ids) - source_ids)
    if missing_v1:
        raise ValueError(f"V1.0 身份在 clean_400 Manifest 中不存在：{missing_v1}")

    by_id = df.assign(sample_id=df["sample_id"].astype(str)).set_index("sample_id", drop=False)
    selected_v1 = by_id.loc[v1_ids].copy()

    actual_v1_counts = {
        int(version): int(count)
        for version, count in selected_v1.groupby("qr_version").size().items()
    }

    selected_rows = [selected_v1]
    extra_counts: dict[int, int] = {}
    already_selected = set(v1_ids)

    for version, target_count in TARGET_VERSION_COUNTS.items():
        current_count = actual_v1_counts.get(version, 0)
        need = target_count - current_count
        if need < 0:
            raise ValueError(
                f"V1.0 已有 Version {version} 身份 {current_count} 个，"
                f"超过 V1.1 目标 {target_count} 个；请重新审定分层配额。"
            )
        candidates = df[
            (df["qr_version"].astype(int) == version)
            & (~df["sample_id"].astype(str).isin(already_selected))
        ].copy()
        candidates = candidates.sort_values("sample_id")
        if len(candidates) < need:
            raise ValueError(
                f"Version {version} 可用新增身份不足：需要 {need}，只有 {len(candidates)}。"
            )
        # 每个版本独立使用确定性随机数；保存实际选中清单供复现。
        rng = random.Random(SELECTION_SEED + int(version))
        candidate_ids = candidates["sample_id"].astype(str).tolist()
        chosen_ids = rng.sample(candidate_ids, need) if need else []
        chosen = candidates[candidates["sample_id"].astype(str).isin(chosen_ids)].copy()
        selected_rows.append(chosen)
        already_selected.update(chosen_ids)
        extra_counts[version] = need

    selected = pd.concat(selected_rows, ignore_index=True)
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected = selected.sort_values("sample_id").reset_index(drop=True)

    if len(selected) != 100:
        raise RuntimeError(f"身份总数应为100，实际为{len(selected)}。")
    if selected["sample_id"].duplicated().any():
        raise RuntimeError("最终身份清单存在重复 sample_id。")

    actual_final = {
        int(version): int(count)
        for version, count in selected.groupby("qr_version").size().items()
    }
    if actual_final != TARGET_VERSION_COUNTS:
        raise RuntimeError(
            f"最终版本分布与目标不一致：实际 {actual_final}，目标 {TARGET_VERSION_COUNTS}。"
        )
    return selected, extra_counts


def check_source_images(selected: pd.DataFrame, image_dir: Path) -> list[tuple[Path, pd.Series, np.ndarray, str]]:
    """逐张读取并校验源图；返回灰度图及原图哈希，不生成输出。"""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"找不到原始二维码图片目录：{image_dir}")

    checked: list[tuple[Path, pd.Series, np.ndarray, str]] = []
    for _, row in selected.iterrows():
        source_path = image_dir / str(row["filename"])
        if not source_path.is_file():
            raise FileNotFoundError(f"原始二维码图片不存在：{source_path}")

        original = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if original is None:
            raise ValueError(f"OpenCV 无法读取原图：{source_path}")
        if original.ndim == 2:
            gray = original
        elif original.ndim == 3 and original.shape[2] in (3, 4):
            gray = cv2.cvtColor(
                original,
                cv2.COLOR_BGRA2GRAY if original.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
            )
        else:
            raise ValueError(
                f"不支持的原图通道结构：{source_path.name}，shape={original.shape}"
            )

        expected_width = int(row["width_px"])
        expected_height = int(row["height_px"])
        height, width = gray.shape
        if width != expected_width or height != expected_height:
            raise ValueError(
                f"图片尺寸不匹配：{source_path.name}；实际 {width}x{height}，"
                f"Manifest {expected_width}x{expected_height}。"
            )
        if gray.dtype != np.uint8:
            raise ValueError(f"原图转灰度后不是 uint8：{source_path.name}，dtype={gray.dtype}")

        checked.append((source_path, row, gray, sha256_file(source_path)))
    return checked


def print_plan(selected: pd.DataFrame, extra_counts: dict[int, int], dry_run: bool) -> None:
    print("=" * 68)
    print("退化二维码高斯模糊参数扫描 V1.1")
    print("=" * 68)
    print(f"运行模式：{'仅检查（dry-run，不生成图片）' if dry_run else '正式生成'}")
    print(f"总身份数：{len(selected)}")
    print("V1.0 已有身份：20（全部保留）")
    print(f"新增身份：{sum(extra_counts.values())}")
    print("新增身份分层数量：" + ", ".join(
        f"Version {version}: {count}" for version, count in sorted(extra_counts.items())
    ))
    print("最终版本分布：")
    for version, count in sorted(selected.groupby("qr_version").size().items()):
        print(f"  Version {int(version)}：{int(count)}")
    print("模糊参数：")
    for config in BLUR_CONFIGS:
        print(f"  {config['id']}: sigma={config['sigma']}, kernel={config['ksize']}x{config['ksize']}")
    print(f"预计图片数：{len(selected)} x (1 CLEAN + {len(BLUR_CONFIGS)} BLUR) = "
          f"{len(selected) * (len(BLUR_CONFIGS) + 1)}")
    print("身份清单：")
    print(", ".join(selected["sample_id"].astype(str).tolist()))


def generate_dataset(
    selected: pd.DataFrame,
    checked_sources: list[tuple[Path, pd.Series, np.ndarray, str]],
    source_manifest_path: Path,
    v1_config_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"输出目录已存在且非空：{output_dir}\n"
            "为保护已有实验，程序拒绝覆盖。请检查目录或更换输出目录。"
        )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, (source_path, source_row, gray, source_hash) in enumerate(checked_sources, start=1):
        source_id = str(source_row["sample_id"])
        width = int(source_row["width_px"])
        height = int(source_row["height_px"])

        # CLEAN 也保存为单通道灰度 PNG，不直接复制原始 RGB PNG。
        clean_filename = f"{source_id}_CLEAN.png"
        clean_path = images_dir / clean_filename
        if not cv2.imwrite(str(clean_path), gray):
            raise IOError(f"CLEAN 图片保存失败：{clean_path}")

        rows.append({
            "sample_id": f"{source_id}_CLEAN",
            "source_sample_id": source_id,
            "filename": clean_filename,
            "payload": str(source_row["payload"]),
            "original_seed": int(source_row["seed"]),
            "selection_seed": SELECTION_SEED,
            "qr_version": int(source_row["qr_version"]),
            "error_correction": str(source_row["error_correction"]),
            "box_size": int(source_row["box_size"]),
            "border_modules": int(source_row["border_modules"]),
            "width_px": width,
            "height_px": height,
            "degradation_type": "none",
            "degradation_id": "CLEAN",
            "degradation_strength": "clean_control",
            "kernel_size": 0,
            "sigma": 0.0,
            "border_type": "none",
            "channel_format": "8-bit_grayscale_PNG",
            "source_filename": str(source_row["filename"]),
            "source_sha256": source_hash,
            "image_sha256": sha256_file(clean_path),
        })

        for config in BLUR_CONFIGS:
            blurred = cv2.GaussianBlur(
                gray,
                (int(config["ksize"]), int(config["ksize"])),
                sigmaX=float(config["sigma"]),
                sigmaY=float(config["sigma"]),
                borderType=cv2.BORDER_REFLECT_101,
            )
            filename = f"{source_id}_{config['id']}.png"
            image_path = images_dir / filename
            if not cv2.imwrite(str(image_path), blurred):
                raise IOError(f"模糊图片保存失败：{image_path}")

            rows.append({
                "sample_id": f"{source_id}_{config['id']}",
                "source_sample_id": source_id,
                "filename": filename,
                "payload": str(source_row["payload"]),
                "original_seed": int(source_row["seed"]),
                "selection_seed": SELECTION_SEED,
                "qr_version": int(source_row["qr_version"]),
                "error_correction": str(source_row["error_correction"]),
                "box_size": int(source_row["box_size"]),
                "border_modules": int(source_row["border_modules"]),
                "width_px": width,
                "height_px": height,
                "degradation_type": "gaussian_blur",
                "degradation_id": str(config["id"]),
                "degradation_strength": "unassigned",
                "kernel_size": int(config["ksize"]),
                "sigma": float(config["sigma"]),
                "border_type": "BORDER_REFLECT_101",
                "channel_format": "8-bit_grayscale_PNG",
                "source_filename": str(source_row["filename"]),
                "source_sha256": source_hash,
                "image_sha256": sha256_file(image_path),
            })

        if index % 10 == 0 or index == len(checked_sources):
            print(f"已完成 {index}/{len(checked_sources)} 个二维码身份")

    manifest = pd.DataFrame(rows)
    expected_rows = len(selected) * (len(BLUR_CONFIGS) + 1)
    if len(manifest) != expected_rows:
        raise RuntimeError(f"Manifest 行数异常：应为 {expected_rows}，实际 {len(manifest)}")
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("新 Manifest 中存在重复 sample_id。")

    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    config = {
        "experiment_name": "gaussian_blur_parameter_scan_v1_1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": SELECTION_SEED,
        "sample_count": len(selected),
        "clean_image_count": len(selected),
        "blur_image_count": len(selected) * len(BLUR_CONFIGS),
        "total_image_count": expected_rows,
        "target_qr_version_counts": {str(k): v for k, v in TARGET_VERSION_COUNTS.items()},
        "actual_qr_version_counts": {
            str(int(k)): int(v) for k, v in selected.groupby("qr_version").size().items()
        },
        "v1_config_path": str(v1_config_path),
        "v1_config_sha256": sha256_file(v1_config_path),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "selected_source_sample_ids": selected["sample_id"].astype(str).tolist(),
        "v1_0_source_sample_ids_preserved": True,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version,
        "image_format": "8-bit single-channel grayscale PNG for CLEAN and all blur images",
        "blur_border_type": "BORDER_REFLECT_101",
        "degradation_parameters": BLUR_CONFIGS,
        "kernel_rule": "K = 2 * ceil(3*sigma) + 1",
        "parameter_level": "预先设定的扫描点，未按解码结果划分轻中重",
        "note": "This dataset is for parameter exploration; do not use it as a final held-out test set.",
    }
    config_path = output_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存后复核：数量、图像通道、尺寸和 SHA256。
    png_files = sorted(images_dir.glob("*.png"))
    if len(png_files) != expected_rows:
        raise RuntimeError(f"图片数量异常：应为 {expected_rows}，实际 {len(png_files)}")
    for _, row in manifest.iterrows():
        image_path = images_dir / str(row["filename"])
        saved = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if saved is None or saved.ndim != 2 or saved.dtype != np.uint8:
            raise RuntimeError(f"输出图片不是 8-bit 单通道灰度图：{image_path.name}")
        if saved.shape != (int(row["height_px"]), int(row["width_px"])):
            raise RuntimeError(f"输出图片尺寸异常：{image_path.name}")
        if sha256_file(image_path) != str(row["image_sha256"]):
            raise RuntimeError(f"输出图片 SHA256 不匹配：{image_path.name}")

    print("\nV1.1 图片生成及保存后复核完成。")
    print(f"图片目录：{images_dir}")
    print(f"Manifest：{manifest_path}")
    print(f"实验配置：{config_path}")
    print(f"总图片数：{expected_rows}")
    print("检查结果：全部为 8-bit 单通道灰度 PNG，Manifest 与 SHA256 校验通过。")


def main() -> int:
    parser = argparse.ArgumentParser(description="高斯模糊参数扫描 V1.1（先 dry-run）")
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--images", type=Path, default=SOURCE_IMAGE_DIR)
    parser.add_argument("--v1-config", type=Path, default=V1_CONFIG)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只检查路径、身份抽样和源图片，不生成任何新图片。",
    )
    args = parser.parse_args()

    source_manifest_path = args.manifest
    v1_config_path = args.v1_config
    manifest_df = load_source_manifest(source_manifest_path)
    v1_config = load_json(v1_config_path)
    selected, extra_counts = choose_identities(manifest_df, v1_config)
    checked_sources = check_source_images(selected, args.images)
    print_plan(selected, extra_counts, args.dry_run)
    print(f"\n已检查原始图片：{len(checked_sources)}/100 张，尺寸与 Manifest 一致。")

    if args.dry_run:
        print("\nDRY-RUN 结束：没有创建输出目录，也没有生成或修改任何图片。")
        print("确认抽样清单和版本分布无误后，再运行不带 --dry-run 的命令。")
        return 0

    generate_dataset(
        selected=selected,
        checked_sources=checked_sources,
        source_manifest_path=source_manifest_path,
        v1_config_path=v1_config_path,
        output_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n程序发生错误：", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
