#!/usr/bin/env python3
"""Generate reproducible QR-code identities and a ground-truth manifest."""
from __future__ import annotations

import argparse
import csv
import json
import random
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import qrcode
    import PIL
    from qrcode.constants import ERROR_CORRECT_M
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Activate your virtual environment and run: "
        "python -m pip install -r requirements.txt"
    ) from exc


def build_payload(index: int, rng: random.Random) -> str:
    """Create a unique, ASCII-only payload with modest length variation."""
    alphabet = string.ascii_uppercase + string.digits
    token = "".join(rng.choice(alphabet) for _ in range(24))
    optional_len = (index * 7) % 41
    suffix = "".join(rng.choice(alphabet) for _ in range(optional_len))
    return f"QR-MVP-20261019|ID={index:04d}|TOKEN={token}|EXTRA={suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a reproducible QR test set.")
    parser.add_argument("--count", type=int, default=50, help="Number of unique QR identities (20-500 recommended).")
    parser.add_argument("--seed", type=int, default=20261019, help="Random seed for exact reproducibility.")
    parser.add_argument("--out", type=Path, default=Path("data/qr"), help="Output directory.")
    parser.add_argument("--box-size", type=int, default=8, help="Pixels per QR module.")
    parser.add_argument("--border", type=int, default=4, help="Quiet-zone width in modules; 4 is the QR minimum.")
    args = parser.parse_args()

    if not 1 <= args.count <= 5000:
        parser.error("--count must be between 1 and 5000.")
    if args.box_size < 2:
        parser.error("--box-size must be >= 2.")
    if args.border < 4:
        parser.error("--border must be >= 4 for a standards-compliant quiet zone.")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    manifest_path = out_dir / "manifest.csv"
    rows: list[dict[str, object]] = []
    payloads: set[str] = set()

    for index in range(1, args.count + 1):
        payload = build_payload(index, rng)
        if payload in payloads:
            raise RuntimeError(f"Duplicate payload unexpectedly generated at index {index}.")
        payloads.add(payload)

        qr = qrcode.QRCode(
            version=None,
            error_correction=ERROR_CORRECT_M,
            box_size=args.box_size,
            border=args.border,
        )
        qr.add_data(payload)
        qr.make(fit=True)
        image = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        sample_id = f"QR{index:04d}"
        filename = f"{sample_id}.png"
        image.save(out_dir / filename, format="PNG", optimize=False)

        rows.append({
            "sample_id": sample_id,
            "filename": filename,
            "payload": payload,
            "seed": args.seed,
            "qr_version": qr.version,
            "error_correction": "M",
            "box_size": args.box_size,
            "border_modules": args.border,
            "width_px": image.width,
            "height_px": image.height,
        })

    fieldnames = list(rows[0].keys()) if rows else []
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "count": args.count,
        "seed": args.seed,
        "generator": "Python qrcode",
        "qrcode_version": getattr(qrcode, "__version__", "see pip metadata"),
        "pillow_version": PIL.__version__,
        "error_correction": "M",
        "box_size": args.box_size,
        "border_modules": args.border,
        "manifest": manifest_path.name,
        "note": "Ground-truth payloads are ASCII strings; image identities are unique.",
    }
    (out_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Generated {len(rows)} QR images: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Unique payloads: {len(payloads)} / {args.count}")
    print(f"Image dimensions range: {min(int(r['width_px']) for r in rows)}-"
          f"{max(int(r['width_px']) for r in rows)} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
