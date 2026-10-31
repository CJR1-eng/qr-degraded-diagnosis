from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
E4_MAN = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
E2_MAN = ROOT / "data" / "pretest_gaussian_blur_v1_1" / "manifest.csv"
E4_E5 = ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_timing.csv"
E2_E5 = ROOT / "records" / "e2_e5_adaptive_v1_1" / "batch_e5_adaptive_timing.csv"
E4_ORIG = ROOT / "records" / "e4_original_baseline_v1_1" / "batch_original_timing.csv"
E2_ORIG = ROOT / "records" / "e2_detector_original_v2" / "batch_original.csv"
OUT = ROOT / "results" / "e5_latency_v1_1"


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    man4 = read(E4_MAN)
    man4["filename"] = man4["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    man2 = read(E2_MAN)

    e4e5 = read(E4_E5)
    e2e5 = read(E2_E5)
    e4orig = read(E4_ORIG)

    m4 = man4[["filename", "sample_id", "condition", "payload"]].merge(e4e5, on="filename", validate="one_to_one")
    m2 = man2[["filename", "payload", "sigma"]].merge(e2e5, on="filename", validate="one_to_one")
    o4 = man4[["filename"]].merge(e4orig, on="filename", validate="one_to_one")

    def corr(m):
        return (m["decoder_status"].eq("DECODED") & m["decoded_payload"].eq(m["payload"]))

    m4["correct"] = corr(m4)
    m2["correct"] = corr(m2)

    # --- 1. 投影先验计算耗时（E5 新增几何成本） ---
    pp4 = num(m4["projective_prior_ms"])
    pp2 = num(m2["projective_prior_ms"])
    print("=== 投影先验 computeProjectivePrior 耗时 (ms) ===")
    print(f"E4 (n={int(pp4.notna().sum())}): mean={pp4.mean():.4f}  median={pp4.median():.4f}  "
          f"p95={pp4.quantile(0.95):.4f}  max={pp4.max():.4f}")
    print(f"E2 (n={int(pp2.notna().sum())}): mean={pp2.mean():.4f}  median={pp2.median():.4f}  "
          f"p95={pp2.quantile(0.95):.4f}  max={pp2.max():.4f}")
    print()

    # --- 2. 微扰重试耗时（按 outcome 分组） ---
    print("=== 微扰重试耗时 (ms) 按 outcome ===")
    for name, m in [("E4", m4), ("E2", m2)]:
        trig = m[m["perturb_retry_outcome"].ne("")]
        if len(trig) == 0:
            print(f"{name}: 无微扰触发")
            continue
        for oc in ["SUCCESS", "EXHAUSTED"]:
            g = trig[trig["perturb_retry_outcome"] == oc]
            if len(g) == 0:
                continue
            rt = num(g["perturb_retry_ms"])
            cc = num(g["perturb_candidate_count"])
            print(f"{name} {oc}: n={len(g)}  retry_ms mean={rt.mean():.3f} median={rt.median():.3f} "
                  f"max={rt.max():.3f}  candidates mean={cc.mean():.1f} median={cc.median():.1f}")
    print()

    # --- 3. 净延迟对比（平均 elapsed_ms：ORIGINAL vs E5） ---
    print("=== 平均端到端解码延迟 elapsed_ms ===")
    e4orig_ms = num(o4["elapsed_ms"])
    e4e5_ms = num(m4["elapsed_ms"])
    print(f"E4 ORIGINAL: mean={e4orig_ms.mean():.2f}  median={e4orig_ms.median():.2f}")
    print(f"E4 E5_ADAPTIVE: mean={e4e5_ms.mean():.2f}  median={e4e5_ms.median():.2f}")
    print(f"  E5 相对 ORIGINAL 平均额外开销: {e4e5_ms.mean() - e4orig_ms.mean():.2f} ms "
          f"({(e4e5_ms.mean()/e4orig_ms.mean()-1)*100:.1f}%)")

    # --- 4. 收益（解码率提升）汇总 ---
    print()
    print("=== 收益（解码率） ===")
    e4_correct = int(m4["correct"].sum())
    print(f"E4 E5+微扰 正确: {e4_correct}/600 = {e4_correct/600*100:.2f}%")
    # 微扰 SUCCESS 样本 = 救回的（本次 run 中）
    success = m4[m4["perturb_retry_outcome"] == "SUCCESS"]
    print(f"E4 微扰救回（SUCCESS）: {len(success)} 张")

    # --- 汇总写入 ---
    summary = {
        "projective_prior_ms": {
            "e4_mean": round(float(pp4.mean()), 4),
            "e4_median": round(float(pp4.median()), 4),
            "e2_mean": round(float(pp2.mean()), 4),
            "e2_median": round(float(pp2.median()), 4),
        },
        "perturb_retry": {},
        "elapsed_ms": {
            "e4_original_mean": round(float(e4orig_ms.mean()), 2),
            "e4_e5_mean": round(float(e4e5_ms.mean()), 2),
            "e4_overhead_ms": round(float(e4e5_ms.mean() - e4orig_ms.mean()), 2),
        },
        "gain": {"e4_correct": e4_correct, "e4_success_rescued": len(success)},
    }
    for name, m in [("e4", m4), ("e2", m2)]:
        trig = m[m["perturb_retry_outcome"].ne("")]
        for oc in ["SUCCESS", "EXHAUSTED"]:
            g = trig[trig["perturb_retry_outcome"] == oc]
            if len(g):
                summary["perturb_retry"][f"{name}_{oc}"] = {
                    "n": int(len(g)),
                    "retry_ms_mean": round(float(num(g["perturb_retry_ms"]).mean()), 3),
                    "retry_ms_median": round(float(num(g["perturb_retry_ms"]).median()), 3),
                    "candidates_mean": round(float(num(g["perturb_candidate_count"]).mean()), 1),
                }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保存微扰触发明细
    trig_all = pd.concat([m4[m4["perturb_retry_outcome"].ne("")][["sample_id", "condition", "perturb_retry_outcome", "perturb_retry_ms", "perturb_candidate_count"]],
                          m2[m2["perturb_retry_outcome"].ne("")][["filename", "sigma", "perturb_retry_outcome", "perturb_retry_ms", "perturb_candidate_count"]]],
                         ignore_index=True)
    trig_all.to_csv(OUT / "perturb_triggered.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
