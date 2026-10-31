from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
PERTURB = ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_perturb.csv"
R3 = ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_r3.csv"
ORACLE = ROOT / "records" / "e4_gt_align_oracle_v1_1" / "batch_gt_align_oracle.csv"
OUT = ROOT / "results" / "e5_perturb_analysis_v1_1"


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main() -> None:
    man = read(MANIFEST)
    man["filename"] = man["image_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]

    pert = read(PERTURB).rename(columns={c: f"{c}_p" for c in read(PERTURB).columns if c != "filename"})
    r3 = read(R3).rename(columns={c: f"{c}_r3" for c in read(R3).columns if c != "filename"})
    ora = read(ORACLE).rename(columns={c: f"{c}_ora" for c in read(ORACLE).columns if c != "filename"})

    m = man[["filename", "sample_id", "condition", "payload"]].merge(pert, on="filename", validate="one_to_one")
    m = m.merge(r3, on="filename", validate="one_to_one")
    m = m.merge(ora, on="filename", validate="one_to_one")

    def correct(prefix: str) -> pd.Series:
        return (m[f"decoder_status_{prefix}"].eq("DECODED")
                & m[f"decoded_payload_{prefix}"].eq(m["payload"]))

    m["pert_correct"] = correct("p")
    m["r3_correct"] = correct("r3")
    m["ora_correct"] = correct("ora")

    m["pert_wrong"] = m["decoder_status_p"].eq("DECODED") & m["decoded_payload_p"].ne(m["payload"])

    print(f"R3 (无微扰) E4 正确: {int(m['r3_correct'].sum())}")
    print(f"Perturb (微扰) E4 正确: {int(m['pert_correct'].sum())}")
    print(f"Oracle E4 正确: {int(m['ora_correct'].sum())}")
    print(f"Perturb 错误 payload: {int(m['pert_wrong'].sum())}\n")

    rescued = m[~m["r3_correct"] & m["pert_correct"]]
    print(f"微扰新增救回 (r3 失败 -> perturb 成功): {len(rescued)} 张")
    print(rescued.groupby("condition").size().to_string())
    print()

    regressed = m[m["r3_correct"] & ~m["pert_correct"]]
    print(f"微扰回退 (r3 成功 -> perturb 失败): {len(regressed)} 张")

    # perturb 结果 vs r3 的 routing 变化
    print("\n=== 救回样本的 r3 路由 与 perturb outcome ===")
    print(rescued[["sample_id", "condition", "gate_decision_r3", "perturb_retry_outcome_p"]].to_string(index=False))
    print()

    # 剩余 gap（Oracle 成功但 perturb 失败）
    remain = m[m["ora_correct"] & ~m["pert_correct"]]
    print(f"=== 剩余 gap (Oracle 成功 & perturb 失败): {len(remain)} 张 ===")
    print(remain.groupby("condition").size().to_string())
    print(remain[["sample_id", "condition", "gate_decision_p", "perturb_retry_outcome_p",
                  "error_type_p"]].to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "r3_correct": int(m["r3_correct"].sum()),
        "perturb_correct": int(m["pert_correct"].sum()),
        "oracle_correct": int(m["ora_correct"].sum()),
        "perturb_wrong_payload": int(m["pert_wrong"].sum()),
        "rescued_vs_r3": int((~m["r3_correct"] & m["pert_correct"]).sum()),
        "regressed_vs_r3": int((m["r3_correct"] & ~m["pert_correct"]).sum()),
        "remaining_gap_vs_oracle": int((m["ora_correct"] & ~m["pert_correct"]).sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rescued.to_csv(OUT / "rescued.csv", index=False, encoding="utf-8-sig")
    remain.to_csv(OUT / "remaining_gap.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
