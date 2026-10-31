from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "e4_perspective_v1_1" / "manifest.csv"
E5 = ROOT / "records" / "e4_e5_adaptive_v1_1" / "batch_e5_adaptive_r3.csv"
ORACLE = ROOT / "records" / "e4_gt_align_oracle_v1_1" / "batch_gt_align_oracle.csv"
OUT = ROOT / "results" / "e5_gap_diagnosis_v1_1"


def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def fnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    man = read(MANIFEST)
    man["filename"] = man["image_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]

    e5 = read(E5).rename(columns={c: f"{c}_e5" for c in read(E5).columns if c != "filename"})
    ora = read(ORACLE).rename(columns={c: f"{c}_ora" for c in read(ORACLE).columns if c != "filename"})

    m = man[["filename", "sample_id", "condition", "payload", "module_count",
             "gt_alignment_dst_x", "gt_alignment_dst_y",
             "gt_module_basis_vx_x", "gt_module_basis_vx_y",
             "gt_module_basis_vy_x", "gt_module_basis_vy_y",
             "gt_finder_tl_dst_x", "gt_finder_tl_dst_y",
             "gt_finder_tr_dst_x", "gt_finder_tr_dst_y",
             "gt_finder_bl_dst_x", "gt_finder_bl_dst_y"]].merge(e5, on="filename", validate="one_to_one")
    m = m.merge(ora, on="filename", validate="one_to_one")

    # module size (px) from GT module basis: average of |vx| and |vy|
    m["module_px"] = 0.5 * (
        (fnum(m["gt_module_basis_vx_x"]) ** 2 + fnum(m["gt_module_basis_vx_y"]) ** 2) ** 0.5
        + (fnum(m["gt_module_basis_vy_x"]) ** 2 + fnum(m["gt_module_basis_vy_y"]) ** 2) ** 0.5
    )

    # proj_prior_error: |p_proj - gt_alignment_dst| in modules
    ppx = fnum(m["projective_prior_x_e5"])
    ppy = fnum(m["projective_prior_y_e5"])
    gtx = fnum(m["gt_alignment_dst_x"])
    gty = fnum(m["gt_alignment_dst_y"])
    m["proj_prior_error_modules"] = (
        ((ppx - gtx) ** 2 + (ppy - gty) ** 2) ** 0.5 / m["module_px"]
    )

    # correctness
    def correct(prefix: str) -> pd.Series:
        return (m[f"decoder_status_{prefix}"].eq("DECODED")
                & m[f"decoded_payload_{prefix}"].eq(m["payload"]))

    m["e5_correct"] = correct("e5")
    m["ora_correct"] = correct("ora")

    gap = m[m["ora_correct"] & ~m["e5_correct"]].copy()

    print(f"Oracle 成功总数: {int(m['ora_correct'].sum())}")
    print(f"E5 成功总数: {int(m['e5_correct'].sum())}")
    print(f"Gap (Oracle 成功 & E5 失败): {len(gap)} 张\n")

    # gap by condition
    print("=== Gap 分条件 ===")
    print(gap.groupby("condition").size().to_string())
    print()

    # routing distribution of gap
    print("=== Gap 的最终路由 (gate_decision_e5) ===")
    print(gap["gate_decision_e5"].value_counts(dropna=False).to_string())
    print()

    # proj_prior_error distribution of gap
    pe = gap["proj_prior_error_modules"]
    print("=== Gap 的 proj_prior_error (模块) 分布 ===")
    print(f"  n={len(pe)}  min={pe.min():.3f}  median={pe.median():.3f}  max={pe.max():.3f}")
    buckets = [0, 0.5, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
    labels = ["<0.5", "0.5-1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0-3.0", "3.0-5.0", "5.0-10.0", ">10.0"]
    cut = pd.cut(pe, bins=buckets, labels=labels, right=False)
    print(cut.value_counts().reindex(labels).to_string())
    print()

    # 关键断言：1.2~3.0 区间占比
    in_range = int(((pe >= 1.2) & (pe < 3.0)).sum())
    print(f"proj_prior_error ∈ [1.2, 3.0) 模块的 Gap 数: {in_range}/{len(pe)} = {in_range/len(pe)*100:.1f}%")
    print()

    # 路由 × proj_prior_error 交叉
    print("=== Gap: 路由 × proj_prior_error 均值 ===")
    print(gap.groupby("gate_decision_e5")["proj_prior_error_modules"]
          .agg(["count", "mean", "median", "min", "max"]).to_string())
    print()

    # residual_norm_modules (detected vs p_proj) 对非 fallback 的 gap
    nonfb = gap[gap["gate_decision_e5"] != "PROJECTIVE_FALLBACK_NO_ALIGNMENT"]
    print(f"=== Gap 中非 PROJECTIVE_FALLBACK 样本 ({len(nonfb)} 张) 的 residual_norm_modules_e5 (detected vs p_proj) ===")
    if len(nonfb):
        print(nonfb["residual_norm_modules_e5"].astype(float).describe().to_string())
    print()

    # 明细
    cols = ["sample_id", "condition", "gate_decision_e5", "proj_prior_error_modules",
            "residual_norm_modules_e5", "projective_scale_ratio_x_e5", "projective_scale_ratio_y_e5",
            "decoder_status_e5", "error_type_e5"]
    detail = gap[cols].copy()
    detail["proj_prior_error_modules"] = detail["proj_prior_error_modules"].round(3)
    print("=== Gap 明细（前 40 条）===")
    print(detail.sort_values("proj_prior_error_modules").to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    gap_out = gap.copy()
    gap_out.to_csv(OUT / "gap_per_image.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "gap_detail.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {OUT}")


if __name__ == "__main__":
    main()
