from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
E4_MANIFEST = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
E4_E5 = ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_r3.csv"
E4_REF = ROOT / "results" / "e4_strategy_analysis_v1_1" / "per_image_comparison.csv"
E4_ORACLE = ROOT / "records" / "e4_gt_align_oracle_v1_1" / "batch_gt_align_oracle.csv"
E2_MANIFEST = ROOT / "data" / "pretest_gaussian_blur_v1_1" / "manifest.csv"
E2_E5 = ROOT / "records" / "e2_e5_adaptive_v1_1" / "batch_e5_adaptive_r3.csv"
OUTPUT = ROOT / "results" / "e5_adaptive_v1_1"
CONDITIONS = ["C0_CLEAN", "P1_MILD", "P2_MEDIUM", "P3_STRONG", "P4_MEDIUM_BLUR_S2", "P5_MEDIUM_BLUR_S3"]
SIGMAS = ["0.0", "1.5", "1.75", "2.0", "2.25", "2.5", "2.75", "3.0", "3.5", "4.0"]
# Verified E2 baselines from the E2 phase (records/e2_*): original 692, affine P0 gate 900.
E2_BASELINE = {"original": 692, "affine_gate": 900, "force_three_point": 900}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def correct(m: pd.DataFrame, prefix: str) -> pd.Series:
    return (m[f"decoder_status_{prefix}"].eq("DECODED")
            & m[f"decoded_payload_{prefix}"].eq(m["payload"]))


def mcnemar_p(improved: int, worsened: int) -> float:
    import math
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def analyze_e4() -> dict:
    man = read(E4_MANIFEST)
    man["filename"] = man["image_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    e5 = read(E4_E5).rename(columns={c: f"{c}_e5" for c in read(E4_E5).columns if c != "filename"})
    ref = read(E4_REF)[["sample_id", "original_correct", "gate_correct", "force_correct",
                        "gate_decision_original"]]
    oracle = read(E4_ORACLE).rename(columns={c: f"{c}_ora" for c in read(E4_ORACLE).columns if c != "filename"})

    m = man[["filename", "sample_id", "condition", "payload"]].merge(e5, on="filename", validate="one_to_one")
    m = m.merge(ref, on="sample_id", validate="one_to_one")
    m = m.merge(oracle, on="filename", validate="one_to_one")

    m["e5_correct"] = correct(m, "e5")
    m["e5_wrong"] = (m["decoder_status_e5"].eq("DECODED") & m["decoded_payload_e5"].ne(m["payload"]))
    m["orig_correct"] = m["original_correct"].astype(str).str.lower().eq("true")
    m["gate_correct"] = m["gate_correct"].astype(str).str.lower().eq("true")
    m["force_correct"] = m["force_correct"].astype(str).str.lower().eq("true")
    m["ora_correct"] = (m["decoder_status_ora"].eq("DECODED") & m["decoded_payload_ora"].eq(m["payload"]))

    by_condition = []
    for c in CONDITIONS:
        g = m[m["condition"] == c]
        by_condition.append({
            "condition": c, "n": len(g),
            "original_correct": int(g["orig_correct"].sum()),
            "affine_gate_correct": int(g["gate_correct"].sum()),
            "force_three_point_correct": int(g["force_correct"].sum()),
            "e5_correct": int(g["e5_correct"].sum()),
            "oracle_correct": int(g["ora_correct"].sum()),
            "e5_wrong_payload": int(g["e5_wrong"].sum()),
        })
    by_condition = pd.DataFrame(by_condition)

    improved = int((~m["orig_correct"] & m["e5_correct"]).sum())
    worsened = int((m["orig_correct"] & ~m["e5_correct"]).sum())

    return {
        "n": 600,
        "original_correct": int(m["orig_correct"].sum()),
        "affine_gate_correct": int(m["gate_correct"].sum()),
        "force_three_point_correct": int(m["force_correct"].sum()),
        "e5_correct": int(m["e5_correct"].sum()),
        "oracle_correct": int(m["ora_correct"].sum()),
        "e5_wrong_payload": int(m["e5_wrong"].sum()),
        "e5_vs_original_improved": improved,
        "e5_vs_original_worsened": worsened,
        "e5_vs_original_mcnemar_p": mcnemar_p(improved, worsened),
        "routing": m["gate_decision_e5"].value_counts().to_dict(),
        "by_condition": by_condition,
    }


def analyze_e2() -> dict:
    man = read(E2_MANIFEST)
    e5 = read(E2_E5).rename(columns={c: f"{c}_e5" for c in read(E2_E5).columns if c != "filename"})
    m = man[["filename", "payload", "sigma"]].merge(e5, on="filename", validate="one_to_one")
    m["e5_correct"] = correct(m, "e5")
    m["e5_wrong"] = (m["decoder_status_e5"].eq("DECODED") & m["decoded_payload_e5"].ne(m["payload"]))

    by_sigma = []
    for s in SIGMAS:
        g = m[m["sigma"] == s]
        by_sigma.append({"sigma": s, "n": len(g), "e5_correct": int(g["e5_correct"].sum()),
                         "e5_wrong_payload": int(g["e5_wrong"].sum())})
    by_sigma = pd.DataFrame(by_sigma)

    return {
        "n": 1000,
        "original_correct": E2_BASELINE["original"],
        "affine_gate_correct": E2_BASELINE["affine_gate"],
        "force_three_point_correct": E2_BASELINE["force_three_point"],
        "e5_correct": int(m["e5_correct"].sum()),
        "e5_wrong_payload": int(m["e5_wrong"].sum()),
        "by_sigma": by_sigma,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    e4 = analyze_e4()
    e2 = analyze_e2()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    e4["by_condition"].to_csv(OUTPUT / "e4_by_condition.csv", index=False, encoding="utf-8-sig")
    e2["by_sigma"].to_csv(OUTPUT / "e2_by_sigma.csv", index=False, encoding="utf-8-sig")

    summary = {
        "method": "E5_ADAPTIVE (projective-prior search + P0_proj gate + projective fallback + near-affine degeneration)",
        "threshold_modules": 2.0,
        "perspective_ratio_epsilon": 0.03,
        "e2": {k: v for k, v in e2.items() if k != "by_sigma"},
        "e4": {k: v for k, v in e4.items() if k != "by_condition"},
        "combined_1600": {
            "e5_total_correct": e2["e5_correct"] + e4["e5_correct"],
            "e5_total_rate": (e2["e5_correct"] + e4["e5_correct"]) / 1600.0,
            "affine_gate_total": e2["affine_gate_correct"] + e4["affine_gate_correct"],
            "original_total": e2["original_correct"] + e4["original_correct"],
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== E4 BY CONDITION ===")
    print(e4["by_condition"].to_string(index=False))
    print("\n=== E2 BY SIGMA ===")
    print(e2["by_sigma"].to_string(index=False))


if __name__ == "__main__":
    main()
