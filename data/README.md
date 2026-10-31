# 数据集说明

本项目使用两个**带真值的合成退化二维码数据集**。原始 PNG 图片（约 79 MB，3278 张）未随仓库提交，请用以下脚本本地生成（生成结果可复现）。

## 依赖

```bash
pip install qrcode pillow numpy opencv-python pandas zxing-cpp
```

## E2 — 近轴高斯模糊集（1000 张）

```bash
python data/generate/generate_qr.py
python data/generate/pretest_gaussian_blur_v1_1.py
```

- 10 档高斯模糊 σ ∈ {0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0}，每档 100 张。
- 真值（`manifest.csv`）：`payload`、`sigma`、`filename` 等。

## E4 — 透视退化集（600 张）

```bash
python data/generate/generate_e4_perspective_v1_1.py
python data/generate/rectify_e4_ground_truth_v1_1.py
```

- 6 条件：C0 清晰 / P1 轻透视 / P2 中透视 / P3 强透视 / P4 中透视+σ2 模糊 / P5 中透视+σ3 模糊，各 100 张。
- 真值（`manifest.csv`）关键字段：
  - `gt_alignment_dst_x/y` — 右下对齐图案中心（640×640 画布坐标）
  - `gt_finder_*_dst_*` — 三个 Finder 中心
  - `gt_module_basis_vx/vy` — 局部方向模块基
  - `H00..H22` — 生成过程的 Homography 矩阵
  - `payload`、`condition`、`module_count` 等

## 复现关键数字

完整复现流程见根目录 `README.md` 的 "Reproduce the key numbers" 一节。

> 说明：跨解码器基准（`code/cross_decoder/`）依赖 OpenCV（`cv2`）与 `zxing-cpp` 两个独立实现；E5 方法的 Java 补丁位于 `code/detector/`，需用 Maven 编译 `zxing-core-e2` 工程。
