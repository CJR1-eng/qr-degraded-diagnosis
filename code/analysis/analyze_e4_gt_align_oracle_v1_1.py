from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
ORACLE = ROOT / "records" / "e4_gt_align_oracle_v1_1" / "batch_gt_align_oracle.csv"
REF = ROOT / "results" / "e4_strategy_analysis_v1_1" / "per_image_comparison.csv"
OUTPUT = ROOT / "results" / "e4_gt_align_oracle_v1_1"
CONDITIONS = ["C0_CLEAN", "P1_MILD", "P2_MEDIUM", "P3_STRONG", "P4_MEDIUM_BLUR_S2", "P5_MEDIUM_BLUR_S3"]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    manifest = read(MANIFEST)
    oracle = read(ORACLE)
    ref = read(REF)

    manifest["filename"] = manifest["image_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    assert len(manifest) == 600 and manifest["filename"].is_unique
    assert len(oracle) == 600 and oracle["filename"].is_unique

    m = manifest[["filename", "sample_id", "condition", "payload"]].merge(
        oracle.rename(columns={c: f"{c}_oracle" for c in oracle.columns if c != "filename"}),
        on="filename", validate="one_to_one")
    m = m.merge(ref[["sample_id", "original_correct", "gate_decision_original"]],
                on="sample_id", validate="one_to_one")

    decoded = m["decoder_status_oracle"].eq("DECODED")
    m["oracle_correct"] = decoded & m["decoded_payload_oracle"].eq(m["payload"])
    m["oracle_wrong_payload"] = decoded & m["decoded_payload_oracle"].ne(m["payload"])
    orig_correct = m["original_correct"].astype(str).str.lower().eq("true")

    noalign = m["gate_decision_original"].eq("NO_ALIGNMENT_FOUND")
    mislock = m["gate_decision_original"].eq("ORIGINAL_KEEP_ALIGNMENT") & (~orig_correct)
    target = noalign | mislock

    by_condition = []
    for c in CONDITIONS:
        g = m[m["condition"] == c]
        by_condition.append({
            "condition": c,
            "n": len(g),
            "original_correct": int(orig_correct[g.index].sum()),
            "oracle_correct": int(g["oracle_correct"].sum()),
            "rescued": int((g["oracle_correct"] & (~orig_correct[g.index])).sum()),
            "oracle_wrong_payload": int(g["oracle_wrong_payload"].sum()),
        })
    by_condition = pd.DataFrame(by_condition)

    summary = {
        "validation": "PASS",
        "rows": 600,
        "mode": "GT_ALIGN_ORACLE",
        "gt_alignment_injected": int(m["gate_decision_oracle"].eq("GT_ALIGNMENT_INJECTED").sum()),
        "overall": {
            "oracle_correct": int(m["oracle_correct"].sum()),
            "oracle_wrong_payload": int(m["oracle_wrong_payload"].sum()),
            "oracle_correct_rate": float(m["oracle_correct"].mean()),
            "original_correct": int(orig_correct.sum()),
            "original_correct_rate": float(orig_correct.mean()),
        },
        "recovery_on_144_target": {
            "no_alignment_target": int(noalign.sum()),
            "no_alignment_rescued": int((noalign & m["oracle_correct"]).sum()),
            "mislock_target": int(mislock.sum()),
            "mislock_rescued": int((mislock & m["oracle_correct"]).sum()),
            "total_target": int(target.sum()),
            "total_rescued": int((target & m["oracle_correct"]).sum()),
        },
        "regression": int((orig_correct & (~m["oracle_correct"])).sum()),
        "injected_but_failed": int((m["gate_decision_oracle"].eq("GT_ALIGNMENT_INJECTED")
                                     & (~m["oracle_correct"])).sum()),
        "finder_not_found_ceiling": int(m["gate_decision_oracle"].eq("FINDER_PATTERN_NOT_FOUND").sum()),
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)
    by_condition.to_csv(OUTPUT / "by_condition.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== BY CONDITION ===")
    print(by_condition.to_string(index=False))


if __name__ == "__main__":
    main()
