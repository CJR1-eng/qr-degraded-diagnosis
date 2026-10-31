from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
E4_MANIFEST = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
E4_PERT = ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_perturb_r2.csv"
E4_R3 = ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_r3.csv"
E4_ORACLE = ROOT / "records" / "e4_gt_align_oracle_v1_1" / "batch_gt_align_oracle.csv"
E2_MANIFEST = ROOT / "data" / "pretest_gaussian_blur_v1_1" / "manifest.csv"
E2_PERT = ROOT / "records" / "e2_e5_adaptive_v1_1" / "batch_e5_adaptive_perturb_r2.csv"
OUT = ROOT / "results" / "e5_perturb_final_v1_1"
CONDITIONS = ["C0_CLEAN", "P1_MILD", "P2_MEDIUM", "P3_STRONG", "P4_MEDIUM_BLUR_S2", "P5_MEDIUM_BLUR_S3"]


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main() -> None:
    man4 = read(E4_MANIFEST)
    man4["filename"] = man4["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    pert = read(E4_PERT).rename(columns={c: f"{c}_p" for c in read(E4_PERT).columns if c != "filename"})
    r3 = read(E4_R3).rename(columns={c: f"{c}_r3" for c in read(E4_R3).columns if c != "filename"})
    ora = read(E4_ORACLE).rename(columns={c: f"{c}_o" for c in read(E4_ORACLE).columns if c != "filename"})

    m4 = man4[["filename", "sample_id", "condition", "payload"]].merge(pert, on="filename", validate="one_to_one")
    m4 = m4.merge(r3, on="filename", validate="one_to_one")
    m4 = m4.merge(ora, on="filename", validate="one_to_one")

    def corr(m, pre):
        return (m[f"decoder_status_{pre}"].eq("DECODED") & m[f"decoded_payload_{pre}"].eq(m["payload"]))

    m4["pert"] = corr(m4, "p")
    m4["r3"] = corr(m4, "r3")
    m4["ora"] = corr(m4, "o")
    m4["pert_wrong"] = m4["decoder_status_p"].eq("DECODED") & m4["decoded_payload_p"].ne(m4["payload"])

    by_condition = []
    for c in CONDITIONS:
        g = m4[m4["condition"] == c]
        by_condition.append({
            "condition": c, "n": len(g),
            "e5_r3": int(g["r3"].sum()),
            "e5_perturb": int(g["pert"].sum()),
            "oracle": int(g["ora"].sum()),
        })
    by_condition = pd.DataFrame(by_condition)

    rescued = m4[~m4["r3"] & m4["pert"]]
    regressed = m4[m4["r3"] & ~m4["pert"]]
    remaining = m4[m4["ora"] & ~m4["pert"]]

    # E2
    man2 = read(E2_MANIFEST)
    e2p = read(E2_PERT)
    m2 = man2[["filename", "payload", "sigma"]].merge(e2p, on="filename", validate="one_to_one")
    m2["pert"] = (m2["decoder_status"].eq("DECODED") & m2["decoded_payload"].eq(m2["payload"]))
    m2["pert_wrong"] = m2["decoder_status"].eq("DECODED") & m2["decoded_payload"].ne(m2["payload"])

    e2_correct = int(m2["pert"].sum())
    e4_correct = int(m4["pert"].sum())

    summary = {
        "method": "E5_ADAPTIVE + checksum-triggered fourth-point perturbation retry",
        "perturb_steps_modules": [0.25, 0.5, 1.0],
        "perturb_directions": "8 (4 axis + 4 diagonal) along finder (u,v) basis",
        "e2": {"n": 1000, "perturb_correct": e2_correct, "rate": e2_correct / 1000.0,
               "wrong_payload": int(m2["pert_wrong"].sum())},
        "e4": {
            "n": 600, "r3_correct": int(m4["r3"].sum()), "perturb_correct": e4_correct,
            "rate": e4_correct / 600.0, "oracle_correct": int(m4["ora"].sum()),
            "wrong_payload": int(m4["pert_wrong"].sum()),
            "rescued_vs_r3": int(len(rescued)), "regressed_vs_r3": int(len(regressed)),
            "remaining_gap_vs_oracle": int(len(remaining)),
        },
        "combined_1600": {"perturb_total": e2_correct + e4_correct,
                          "rate": (e2_correct + e4_correct) / 1600.0},
        "e4_by_condition": by_condition.to_dict(orient="records"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    by_condition.to_csv(OUT / "e4_by_condition.csv", index=False, encoding="utf-8-sig")
    rescued[["sample_id", "condition", "gate_decision_r3", "perturb_retry_outcome_p"]].to_csv(
        OUT / "rescued_vs_r3.csv", index=False, encoding="utf-8-sig")
    remaining[["sample_id", "condition", "gate_decision_p", "perturb_retry_outcome_p"]].to_csv(
        OUT / "remaining_gap.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== E4 BY CONDITION ===")
    print(by_condition.to_string(index=False))
    print(f"\n微扰救回 {len(rescued)} 张, 回退 {len(regressed)} 张, 剩余 gap {len(remaining)} 张")


if __name__ == "__main__":
    main()
