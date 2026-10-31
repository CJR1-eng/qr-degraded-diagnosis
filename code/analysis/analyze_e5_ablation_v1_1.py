from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
OUT = ROOT / "results" / "e5_ablation_v1_1"

CONFIGS = {
    "full": ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_timing.csv",
    "no_search": ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_abl_nosearch.csv",
    "no_gate": ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_abl_nogate.csv",
    "no_fallback": ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_abl_nofallback.csv",
    "no_perturb": ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_abl_noperturb.csv",
}
CONDITIONS = ["C0_CLEAN", "P1_MILD", "P2_MEDIUM", "P3_STRONG", "P4_MEDIUM_BLUR_S2", "P5_MEDIUM_BLUR_S3"]


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def main() -> None:
    man = read(MAN)
    man["filename"] = man["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    base = man[["filename", "sample_id", "condition", "payload"]]

    results = {}
    for name, path in CONFIGS.items():
        df = read(path)
        m = base.merge(df, on="filename", validate="one_to_one")
        m["correct"] = (m["decoder_status"].eq("DECODED") & m["decoded_payload"].eq(m["payload"]))
        m["wrong"] = m["decoder_status"].eq("DECODED") & m["decoded_payload"].ne(m["payload"])
        results[name] = m

    full = results["full"]
    n_full = int(full["correct"].sum())

    rows = []
    for name, m in results.items():
        rows.append({
            "config": name,
            "correct": int(m["correct"].sum()),
            "wrong": int(m["wrong"].sum()),
            "delta_vs_full": int(m["correct"].sum()) - n_full,
        })
    table = pd.DataFrame(rows)
    table["rate"] = (table["correct"] / 600.0 * 100).round(2)

    print("=== E4 消融结果（600 张）===")
    print(table.to_string(index=False))
    print()

    # 组件增量（leave-one-out）
    comp = {
        "A 投影搜索引导": n_full - int(results["no_search"]["correct"].sum()),
        "B P0_proj 门控": n_full - int(results["no_gate"]["correct"].sum()),
        "C 投影回退": n_full - int(results["no_fallback"]["correct"].sum()),
        "D 微扰重试": n_full - int(results["no_perturb"]["correct"].sum()),
    }
    print("=== 各组件增量贡献（移除后损失，张）===")
    for k, v in sorted(comp.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} 张")
    print()

    # 分条件消融（关键配置）
    print("=== 分条件（full / no_gate / no_fallback / no_perturb）===")
    by_cond = []
    for c in CONDITIONS:
        row = {"condition": c}
        for name in ["full", "no_gate", "no_fallback", "no_perturb"]:
            g = results[name][results[name]["condition"] == c]
            row[name] = int(g["correct"].sum())
        by_cond.append(row)
    by_cond = pd.DataFrame(by_cond)
    print(by_cond.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    by_cond.to_csv(OUT / "ablation_by_condition.csv", index=False, encoding="utf-8-sig")
    summary = {
        "full_correct": n_full,
        "ablation": {name: int(m["correct"].sum()) for name, m in results.items()},
        "component_delta_vs_full": comp,
        "note": "search guidance (A) contributes 0: allowance loop (4..16 modules) is already robust; "
                "geometry decision (gate/fallback) and sub-module correction (perturb) are the key drivers.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
