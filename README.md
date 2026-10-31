# QR-Degraded-Diagnosis

**White-box geometric diagnosis of degraded QR decoding — and an honest cross-decoder reality check.**

A research-grade case study that (1) diagnoses *why* ZXing fails on perspective-distorted + blurred QR codes, (2) proposes a zero-training, interpretable geometric fix, and (3) — just as importantly — shows through a cross-decoder benchmark that the problem is largely specific to ZXing's Java implementation, not a general QR decoding difficulty.

> **Bottom line up front:** the proposed method lifts ZXing from 57.67% → 81.67% on a perspective-degraded benchmark (matching its own oracle upper bound), but `zxing-cpp` already solves the *same* dataset at **99.83%** and OpenCV at **88.17%**. The project is therefore published as an **honest negative result + reusable benchmark**, not as a "new SOTA decoder".

---

## Cross-decoder benchmark (E4 perspective-degraded set, 600 images)

| Decoder | Correct | Nature |
|---|---|---|
| ZXing **Java 3.5.4** (baseline) | 346 / 600 = **57.67%** | the decoder being diagnosed |
| Proposed **E5 + perturbation** | 490 / 600 = **81.67%** | in-framework fix for ZXing Java |
| OpenCV 5.0 | 529 / 600 = **88.17%** | independent implementation |
| **zxing-cpp 3.1.1** | **599 / 600 = 99.83%** | ZXing's rewritten C++ port |

Per-condition (ZXing Java / E5 / OpenCV / zxing-cpp):

| Condition | ZXing | E5 | OpenCV | zxing-cpp |
|---|---|---|---|---|
| P3 strong perspective | 35 | 38 | 86 | 100 |
| P4 medium perspective + σ2 blur | 31 | 92 | 98 | 100 |
| P5 medium perspective + σ3 blur | 0 | 69 | 46 | 99 |

---

## What this project contains

### 1. Root-cause diagnosis

ZXing's affine prior for the bottom-right alignment pattern conflates two error sources under perspective:

```
P0 ≈ ε_struct + ε_true
     ε_struct ≈ 1.86–3.77 modules   (structural error of the affine prior)
     ε_true   ≈ 0.03–0.09 modules   (true localization error)
```

This is why affine-P0 gating *worsens* ZXing on perspective data (346 → 161), despite helping on near-axis blur (692 → 900).

### 2. The proposed method (E5)

- **Projective prior `p_proj`** — closed-form reconstruction of the alignment center (module `D−6.5`) from three finder centers + local directional module sizes, no corner extraction, no SVD. Error 0.0 modules on the synthetic perspective set (affine prior: median 2.88).
- **`P0_proj` gate** — detects alignment-pattern mislocks using `p_proj` instead of the affine prior.
- **Projective fallback** — uses `p_proj` as the 4th point when no alignment pattern is found.
- **Checksum-triggered perturbation retry** — re-samples the 4th point with sub-module offsets only when decoding fails (zero cost on successes).

**Result:** E4 490/600 = 81.67%, matching the oracle upper bound (ground-truth alignment center injected); E2 (near-axis) 900/1000, zero regression; vs original ZXing: +144 recovered, 0 worsened.

### 3. Ablation & latency

| Component removed (E4) | Loss |
|---|---|
| Projective fallback | 94 images |
| `P0_proj` gate | 37 images |
| Perturbation retry | 28 images |
| Search guidance | 0 images |

Latency: **+1.4%** wall-clock overhead for **+41.6%** relative decode-rate gain; projective prior ≈ 6 µs median; perturbation only on failures (~0.7 ms).

### 4. Why the direction was terminated

`zxing-cpp`'s `QRDetector` already does what the proposed `p_proj` aims to do — a perspective-aware bottom-right extrapolation — via `TraceLine` (intersecting the finders' inner edge lines) plus sub-pixel finder refinement and multi-binarizer fallback. The "alignment-pattern mislock" problem is therefore a **Java-ZXing-specific architectural weakness** (single global perspective transform + affine prior), not a universal QR difficulty. This is exactly the "guard against ZXing-bug-fix criticism" risk flagged at project inception, now confirmed by cross-decoder evidence.

---

## Repository layout

```
├── code/
│   ├── detector/        # E5 patch on ZXing Java's Detector.java (E5_ADAPTIVE + perturbation)
│   ├── analysis/        # ablation, latency, gap-diagnosis, adaptive-decoder analysis scripts
│   └── cross_decoder/   # OpenCV & zxing-cpp batch decoders + three-way comparison
├── data/
│   └── generate/        # scripts to regenerate the synthetic datasets (with GT homography)
├── results/             # per-condition / ablation / latency summary CSVs & JSONs
├── docs/
│   ├── direction_review.md   # full experiment summary & termination assessment
│   └── paper_draft.md        # paper draft (methods + synthetic experiments, data-verified)
└── README.md
```

## Datasets

Two synthetic degraded-QR sets with ground-truth homography (alignment-center + finder positions per image):

- **E2** — near-axis Gaussian blur, 1000 images, σ ∈ {0 … 4.0}.
- **E4** — perspective distortion, 600 images across 6 conditions (clean / mild / medium / strong perspective, plus two joint perspective+blur levels).

Regenerate via `data/generate/*.py` (requires the QR generator + `pyzbar`-free pipeline; see `docs/`). Raw PNGs are ~79 MB and are therefore not committed — scripts reproduce them deterministically.

## Reproduce the key numbers

```bash
# 1. build the patched ZXing (Java)
mvn -f zxing-core-e2/pom.xml clean compile

# 2. run E5_ADAPTIVE on E4
java -Dzxing.e2.geometryMode=E5_ADAPTIVE -Dzxing.e2.thresholdModules=2.0 \
     -cp zxing-core-e2/target/classes edu.wuyi.qr.BatchDecodeE2 \
     data/e4_perspective_v1_1/images records/e4_e5.csv

# 3. cross-decoder benchmark (OpenCV / zxing-cpp)
python code/cross_decoder/run_e4_opencv_v1_1.py
python code/cross_decoder/run_e4_zxingcpp_v1_1.py

# 4. analysis
python code/analysis/analyze_e5_ablation_v1_1.py
python code/analysis/analyze_e5_latency_v1_1.py
```

## License

See `LICENSE`. The ZXing-derived detector code retains its Apache-2.0 lineage.
