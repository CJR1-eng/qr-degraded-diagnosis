/*
 * Copyright 2007 ZXing authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.zxing.qrcode.detector;

import com.google.zxing.DecodeHintType;
import com.google.zxing.FormatException;
import com.google.zxing.NotFoundException;
import com.google.zxing.ResultPoint;
import com.google.zxing.ResultPointCallback;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.common.DetectorResult;
import com.google.zxing.common.GridSampler;
import com.google.zxing.common.PerspectiveTransform;
import com.google.zxing.common.detector.MathUtils;
import com.google.zxing.qrcode.decoder.Version;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * <p>Encapsulates logic that can detect a QR Code in an image, even if the QR Code
 * is rotated or skewed, or partially obscured.</p>
 *
 * @author Sean Owen
 */
public class Detector {

  /**
   * E2/E5 experiment controls. Set as JVM properties before Detector is loaded:
   *   -Dzxing.e2.geometryMode=ORIGINAL|FORCE_THREE_POINT|P0_GATE|GT_ALIGN_ORACLE
   *   -Dzxing.e2.thresholdModules=2.0
   *
   * ORIGINAL is the default and preserves the upstream 3.5.4 geometry choice.
   * GT_ALIGN_ORACLE replaces the detected Alignment with the ground-truth center
   * supplied per-image via {@link #setGtAlignment(float, float)}, isolating the
   * pure four-point geometry upper bound.
   */
  private enum GeometryMode {
    ORIGINAL,
    FORCE_THREE_POINT,
    P0_GATE,
    GT_ALIGN_ORACLE,
    E5_ADAPTIVE;

    private static GeometryMode fromSystemProperty() {
      String value = System.getProperty("zxing.e2.geometryMode", "ORIGINAL").trim();
      try {
        return GeometryMode.valueOf(value.toUpperCase(java.util.Locale.ROOT));
      } catch (IllegalArgumentException e) {
        throw new IllegalArgumentException(
            "Unsupported -Dzxing.e2.geometryMode=" + value
                + ". Expected ORIGINAL, FORCE_THREE_POINT, P0_GATE, GT_ALIGN_ORACLE, or E5_ADAPTIVE.", e);
      }
    }
  }

  private static final GeometryMode GEOMETRY_MODE = GeometryMode.fromSystemProperty();
  private static final double P0_THRESHOLD_MODULES = readThresholdModules();
  /** Max per-finder scale-ratio deviation from unity below which the scene is treated as affine. */
  private static final float PERSPECTIVE_RATIO_EPSILON = 0.03f;
  /**
   * E5 component switches for ablation. All default to enabled; disable one at a time via
   * -Dzxing.e2.projectiveSearch=false / -Dzxing.e2.p0projGate=false /
   * -Dzxing.e2.projectiveFallback=false to isolate each component's incremental contribution.
   */
  private static final boolean ENABLE_PROJECTIVE_SEARCH =
      readBoolFlag("zxing.e2.projectiveSearch", true);
  private static final boolean ENABLE_P0PROJ_GATE =
      readBoolFlag("zxing.e2.p0projGate", true);
  private static final boolean ENABLE_PROJECTIVE_FALLBACK =
      readBoolFlag("zxing.e2.projectiveFallback", true);
  private static final ThreadLocal<Diagnostics> LAST_DIAGNOSTICS =
      ThreadLocal.withInitial(Diagnostics::new);
  /** Per-image ground-truth alignment center {x, y} in image pixels, for GT_ALIGN_ORACLE. */
  private static final ThreadLocal<float[]> GT_ALIGNMENT =
      ThreadLocal.withInitial(() -> null);
  /**
   * E5 perturbation retry: the last sampled four-point geometry, retained so the reader can
   * re-sample with a small perturbation of the fourth point when the first decode fails its
   * checksum. Only populated under E5_ADAPTIVE when a fourth point was actually selected.
   */
  private static final ThreadLocal<SamplingContext> LAST_SAMPLING_CONTEXT =
      ThreadLocal.withInitial(() -> null);

  private final BitMatrix image;
  private ResultPointCallback resultPointCallback;

  /** Supplies the ground-truth alignment center for the next decode on this thread. */
  public static void setGtAlignment(float x, float y) {
    GT_ALIGNMENT.set(new float[]{x, y});
  }

  /** Clears any pending ground-truth alignment for this thread. */
  public static void clearGtAlignment() {
    GT_ALIGNMENT.remove();
  }

  /** Records the perturbation-retry outcome for the current decode into the live diagnostics. */
  public static void recordPerturbRetryOutcome(String outcome) {
    LAST_DIAGNOSTICS.get().perturbRetryOutcome = outcome;
  }

  /** Records the perturbation-retry wall-clock cost (ms) and candidate count into diagnostics. */
  public static void recordPerturbRetryCost(double nanosMs, int candidateCount) {
    Diagnostics d = LAST_DIAGNOSTICS.get();
    d.perturbRetryNanos = nanosMs;
    d.perturbCandidateCount = candidateCount;
  }

  /**
   * Returns alternative detections obtained by re-sampling with a small perturbation of the
   * fourth point, for the reader to try when the primary decode fails its checksum. Returns an
   * empty list when no perturbable four-point geometry was captured on this thread.
   */
  public static List<DetectorResult> perturbedDetections() {
    SamplingContext ctx = LAST_SAMPLING_CONTEXT.get();
    return ctx == null ? Collections.emptyList() : ctx.perturbedDetections();
  }

  private static double readThresholdModules() {
    String value = System.getProperty("zxing.e2.thresholdModules", "2.0").trim();
    final double parsed;
    try {
      parsed = Double.parseDouble(value);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException("Invalid -Dzxing.e2.thresholdModules=" + value, e);
    }
    if (!Double.isFinite(parsed) || parsed < 0.0) {
      throw new IllegalArgumentException(
          "-Dzxing.e2.thresholdModules must be finite and >= 0: " + value);
    }
    return parsed;
  }

  private static boolean readBoolFlag(String key, boolean defaultValue) {
    String value = System.getProperty(key);
    if (value == null) {
      return defaultValue;
    }
    return Boolean.parseBoolean(value.trim());
  }

  /** Clears per-thread telemetry before a new decode attempt. */
  public static void resetLastDiagnostics() {
    LAST_DIAGNOSTICS.set(new Diagnostics());
  }

  /** Returns a snapshot of telemetry from the most recent detector invocation on this thread. */
  public static Diagnostics getLastDiagnostics() {
    return LAST_DIAGNOSTICS.get().copy();
  }

  public static String getConfiguredGeometryMode() {
    return GEOMETRY_MODE.name();
  }

  public static double getConfiguredThresholdModules() {
    return P0_THRESHOLD_MODULES;
  }

  /** Immutable-by-interface snapshot of one Detector invocation. */
  public static final class Diagnostics {
    private boolean detectorReached;
    private boolean finderPatternsFound;
    private int candidateDimension = -1;
    private int sampledBitsDimension = -1;
    private int selectedPointCount = -1;
    private int detectedPointCount = -1;
    private boolean alignmentSearchAttempted;
    private boolean originalAlignmentFound;
    private double alignmentX = Double.NaN;
    private double alignmentY = Double.NaN;
    private double searchPriorX = Double.NaN;
    private double searchPriorY = Double.NaN;
    private double residualDxPx = Double.NaN;
    private double residualDyPx = Double.NaN;
    private double residualDxModules = Double.NaN;
    private double residualDyModules = Double.NaN;
    private double residualNormModules = Double.NaN;
    private double projectivePriorX = Double.NaN;
    private double projectivePriorY = Double.NaN;
    private double projectiveScaleRatioX = Double.NaN;
    private double projectiveScaleRatioY = Double.NaN;
    private String selectedGeometry = "UNKNOWN";
    private String gateDecision = "DETECTOR_NOT_REACHED";
    private boolean gridSampled;
    private String perturbRetryOutcome = "";
    private double projectivePriorNanos = Double.NaN;
    private double perturbRetryNanos = Double.NaN;
    private int perturbCandidateCount = 0;

    private Diagnostics copy() {
      Diagnostics c = new Diagnostics();
      c.detectorReached = detectorReached;
      c.finderPatternsFound = finderPatternsFound;
      c.candidateDimension = candidateDimension;
      c.sampledBitsDimension = sampledBitsDimension;
      c.selectedPointCount = selectedPointCount;
      c.detectedPointCount = detectedPointCount;
      c.alignmentSearchAttempted = alignmentSearchAttempted;
      c.originalAlignmentFound = originalAlignmentFound;
      c.alignmentX = alignmentX;
      c.alignmentY = alignmentY;
      c.searchPriorX = searchPriorX;
      c.searchPriorY = searchPriorY;
      c.residualDxPx = residualDxPx;
      c.residualDyPx = residualDyPx;
      c.residualDxModules = residualDxModules;
      c.residualDyModules = residualDyModules;
      c.residualNormModules = residualNormModules;
      c.projectivePriorX = projectivePriorX;
      c.projectivePriorY = projectivePriorY;
      c.projectiveScaleRatioX = projectiveScaleRatioX;
      c.projectiveScaleRatioY = projectiveScaleRatioY;
      c.selectedGeometry = selectedGeometry;
      c.gateDecision = gateDecision;
      c.gridSampled = gridSampled;
      c.perturbRetryOutcome = perturbRetryOutcome;
      c.projectivePriorNanos = projectivePriorNanos;
      c.perturbRetryNanos = perturbRetryNanos;
      c.perturbCandidateCount = perturbCandidateCount;
      return c;
    }

    public boolean isDetectorReached() { return detectorReached; }
    public boolean isFinderPatternsFound() { return finderPatternsFound; }
    public int getCandidateDimension() { return candidateDimension; }
    public int getSampledBitsDimension() { return sampledBitsDimension; }
    public int getSelectedPointCount() { return selectedPointCount; }
    public int getDetectedPointCount() { return detectedPointCount; }
    public boolean isAlignmentSearchAttempted() { return alignmentSearchAttempted; }
    public boolean isOriginalAlignmentFound() { return originalAlignmentFound; }
    public double getAlignmentX() { return alignmentX; }
    public double getAlignmentY() { return alignmentY; }
    public double getSearchPriorX() { return searchPriorX; }
    public double getSearchPriorY() { return searchPriorY; }
    public double getResidualDxPx() { return residualDxPx; }
    public double getResidualDyPx() { return residualDyPx; }
    public double getResidualDxModules() { return residualDxModules; }
    public double getResidualDyModules() { return residualDyModules; }
    public double getResidualNormModules() { return residualNormModules; }
    public double getProjectivePriorX() { return projectivePriorX; }
    public double getProjectivePriorY() { return projectivePriorY; }
    public double getProjectiveScaleRatioX() { return projectiveScaleRatioX; }
    public double getProjectiveScaleRatioY() { return projectiveScaleRatioY; }
    public String getSelectedGeometry() { return selectedGeometry; }
    public String getGateDecision() { return gateDecision; }
    public boolean isGridSampled() { return gridSampled; }
    public String getPerturbRetryOutcome() { return perturbRetryOutcome; }
    public double getProjectivePriorNanos() { return projectivePriorNanos; }
    public double getPerturbRetryNanos() { return perturbRetryNanos; }
    public int getPerturbCandidateCount() { return perturbCandidateCount; }
  }

  public Detector(BitMatrix image) {
    this.image = image;
  }

  protected final BitMatrix getImage() {
    return image;
  }

  protected final ResultPointCallback getResultPointCallback() {
    return resultPointCallback;
  }

  /**
   * <p>Detects a QR Code in an image.</p>
   *
   * @return {@link DetectorResult} encapsulating results of detecting a QR Code
   * @throws NotFoundException if QR Code cannot be found
   * @throws FormatException if a QR Code cannot be decoded
   */
  public DetectorResult detect() throws NotFoundException, FormatException {
    return detect(null);
  }

  /**
   * <p>Detects a QR Code in an image.</p>
   *
   * @param hints optional hints to detector
   * @return {@link DetectorResult} encapsulating results of detecting a QR Code
   * @throws NotFoundException if QR Code cannot be found
   * @throws FormatException if a QR Code cannot be decoded
   */
  public final DetectorResult detect(Map<DecodeHintType,?> hints) throws NotFoundException, FormatException {

    Diagnostics diagnostics = new Diagnostics();
    diagnostics.detectorReached = true;
    diagnostics.gateDecision = "FINDER_PATTERN_SEARCH";
    LAST_DIAGNOSTICS.set(diagnostics);

    resultPointCallback = hints == null ? null :
        (ResultPointCallback) hints.get(DecodeHintType.NEED_RESULT_POINT_CALLBACK);

    FinderPatternFinder finder = new FinderPatternFinder(image, resultPointCallback);
    FinderPatternInfo info;
    try {
      info = finder.find(hints);
    } catch (NotFoundException e) {
      diagnostics.gateDecision = "FINDER_PATTERN_NOT_FOUND";
      throw e;
    }
    diagnostics.finderPatternsFound = true;

    return processFinderPatternInfo(info);
  }

  protected final DetectorResult processFinderPatternInfo(FinderPatternInfo info)
      throws NotFoundException, FormatException {

    Diagnostics diagnostics = LAST_DIAGNOSTICS.get();
    FinderPattern topLeft = info.getTopLeft();
    FinderPattern topRight = info.getTopRight();
    FinderPattern bottomLeft = info.getBottomLeft();

    float moduleSize = calculateModuleSize(topLeft, topRight, bottomLeft);
    if (moduleSize < 1.0f) {
      diagnostics.gateDecision = "MODULE_SIZE_BELOW_ONE_PIXEL";
      throw NotFoundException.getNotFoundInstance();
    }
    int dimension = computeDimension(topLeft, topRight, bottomLeft, moduleSize);
    Version provisionalVersion = Version.getProvisionalVersionForDimension(dimension);
    int modulesBetweenFPCenters = provisionalVersion.getDimensionForVersion() - 7;
    diagnostics.candidateDimension = dimension;

    AlignmentPattern alignmentPattern = null;
    int estAlignmentX = -1;
    int estAlignmentY = -1;
    boolean alignmentSearchExpected = provisionalVersion.getAlignmentPatternCenters().length > 0;

    // E5 projective prior (alignment center, module dimension-6.5).
    ResultPoint projectivePrior = null;
    if (GEOMETRY_MODE == GeometryMode.E5_ADAPTIVE && alignmentSearchExpected) {
      long priorStartNanos = System.nanoTime();
      projectivePrior = computeProjectivePrior(topLeft, topRight, bottomLeft, dimension);
      diagnostics.projectivePriorNanos = (System.nanoTime() - priorStartNanos) / 1e6;
      if (projectivePrior != null) {
        diagnostics.projectivePriorX = projectivePrior.getX();
        diagnostics.projectivePriorY = projectivePrior.getY();
      }
    }

    // Anything above version 1 has an alignment pattern
    if (alignmentSearchExpected) {
      diagnostics.alignmentSearchAttempted = true;

      if (projectivePrior != null && ENABLE_PROJECTIVE_SEARCH) {
        // Perspective-aware search center.
        estAlignmentX = (int) projectivePrior.getX();
        estAlignmentY = (int) projectivePrior.getY();
      } else {
        // Guess where a "bottom right" finder pattern would have been
        float bottomRightX = topRight.getX() - topLeft.getX() + bottomLeft.getX();
        float bottomRightY = topRight.getY() - topLeft.getY() + bottomLeft.getY();

        // Estimate that alignment pattern is closer by 3 modules
        // from "bottom right" to known top left location
        float correctionToTopLeft = 1.0f - 3.0f / modulesBetweenFPCenters;
        estAlignmentX = (int) (topLeft.getX() + correctionToTopLeft * (bottomRightX - topLeft.getX()));
        estAlignmentY = (int) (topLeft.getY() + correctionToTopLeft * (bottomRightY - topLeft.getY()));
      }
      diagnostics.searchPriorX = estAlignmentX;
      diagnostics.searchPriorY = estAlignmentY;

      // Kind of arbitrary -- expand search radius before giving up
      for (int i = 4; i <= 16; i <<= 1) {
        try {
          alignmentPattern = findAlignmentInRegion(moduleSize,
              estAlignmentX,
              estAlignmentY,
              i);
          break;
        } catch (NotFoundException re) {
          // try next round
        }
      }
      // If we didn't find alignment pattern... well try anyway without it
    }

    diagnostics.originalAlignmentFound = alignmentPattern != null;
    if (alignmentPattern != null) {
      diagnostics.alignmentX = alignmentPattern.getX();
      diagnostics.alignmentY = alignmentPattern.getY();
      calculateP0Residual(diagnostics, topLeft, topRight, bottomLeft,
          alignmentPattern, dimension, estAlignmentX, estAlignmentY);
    }

    // Apply the selected E2/E5 geometry policy only after the unmodified ZXing
    // AlignmentPattern search has completed. The upstream transform/sampler is reused.
    if (GEOMETRY_MODE == GeometryMode.GT_ALIGN_ORACLE) {
      float[] gt = GT_ALIGNMENT.get();
      if (gt != null) {
        alignmentPattern = new AlignmentPattern(gt[0], gt[1], moduleSize);
        diagnostics.alignmentX = gt[0];
        diagnostics.alignmentY = gt[1];
        diagnostics.gateDecision = "GT_ALIGNMENT_INJECTED";
      } else {
        diagnostics.gateDecision = "GT_ALIGNMENT_MISSING";
      }
    } else if (GEOMETRY_MODE == GeometryMode.E5_ADAPTIVE) {
      if (projectivePrior != null) {
        // Perspective path: P0_proj gate + projective-prior fallback.
        double p0proj = (alignmentPattern != null)
            ? computeResidualModules(alignmentPattern, projectivePrior,
                topLeft, topRight, bottomLeft, dimension)
            : Double.NaN;
        diagnostics.residualNormModules = p0proj;
        if (alignmentPattern != null && ENABLE_P0PROJ_GATE && Double.isFinite(p0proj)
            && p0proj <= P0_THRESHOLD_MODULES) {
          diagnostics.gateDecision = "P0PROJ_WITHIN_THRESHOLD";
        } else if (alignmentPattern != null) {
          if (ENABLE_P0PROJ_GATE) {
            alignmentPattern = new AlignmentPattern(projectivePrior.getX(), projectivePrior.getY(), moduleSize);
            diagnostics.alignmentX = projectivePrior.getX();
            diagnostics.alignmentY = projectivePrior.getY();
            diagnostics.gateDecision = "P0PROJ_EXCEEDS_THRESHOLD";
          } else {
            diagnostics.gateDecision = "P0PROJ_GATE_DISABLED_KEEP";
          }
        } else {
          if (ENABLE_PROJECTIVE_FALLBACK) {
            alignmentPattern = new AlignmentPattern(projectivePrior.getX(), projectivePrior.getY(), moduleSize);
            diagnostics.alignmentX = projectivePrior.getX();
            diagnostics.alignmentY = projectivePrior.getY();
            diagnostics.gateDecision = "PROJECTIVE_FALLBACK_NO_ALIGNMENT";
          } else {
            diagnostics.gateDecision = "PROJECTIVE_FALLBACK_DISABLED";
          }
        }
      } else {
        // Near-affine path: affine P0 gate + affine three-point fallback (original P0_GATE).
        if (alignmentPattern != null) {
          if (Double.isFinite(diagnostics.residualNormModules)) {
            if (diagnostics.residualNormModules > P0_THRESHOLD_MODULES) {
              alignmentPattern = null;
              diagnostics.gateDecision = "P0_RESIDUAL_EXCEEDS_THRESHOLD";
            } else {
              diagnostics.gateDecision = "P0_RESIDUAL_WITHIN_THRESHOLD";
            }
          } else {
            diagnostics.gateDecision = "P0_RESIDUAL_UNAVAILABLE_KEEP_ALIGNMENT";
          }
        } else if (!alignmentSearchExpected) {
          diagnostics.gateDecision = "NO_ALIGNMENT_EXPECTED";
        } else {
          diagnostics.gateDecision = "NO_ALIGNMENT_FOUND";
        }
      }
    } else if (alignmentPattern != null) {
      switch (GEOMETRY_MODE) {
        case ORIGINAL:
          diagnostics.gateDecision = "ORIGINAL_KEEP_ALIGNMENT";
          break;
        case FORCE_THREE_POINT:
          alignmentPattern = null;
          diagnostics.gateDecision = "FORCE_THREE_POINT";
          break;
        case P0_GATE:
          if (Double.isFinite(diagnostics.residualNormModules)) {
            if (diagnostics.residualNormModules > P0_THRESHOLD_MODULES) {
              alignmentPattern = null;
              diagnostics.gateDecision = "P0_RESIDUAL_EXCEEDS_THRESHOLD";
            } else {
              diagnostics.gateDecision = "P0_RESIDUAL_WITHIN_THRESHOLD";
            }
          } else {
            diagnostics.gateDecision = "P0_RESIDUAL_UNAVAILABLE_KEEP_ALIGNMENT";
          }
          break;
        default:
          throw new IllegalStateException("Unhandled geometry mode: " + GEOMETRY_MODE);
      }
    } else if (!alignmentSearchExpected) {
      diagnostics.gateDecision = "NO_ALIGNMENT_EXPECTED";
    } else {
      diagnostics.gateDecision = "NO_ALIGNMENT_FOUND";
    }

    diagnostics.selectedGeometry = alignmentPattern == null ? "THREE_POINT" : "FOUR_POINT";
    diagnostics.selectedPointCount = alignmentPattern == null ? 3 : 4;

    PerspectiveTransform transform =
        createTransform(topLeft, topRight, bottomLeft, alignmentPattern, dimension);

    BitMatrix bits = sampleGrid(image, transform, dimension);
    diagnostics.gridSampled = true;
    // Match the public DetectorResult semantics: sampled dimension and point
    // count are available only after the full detector operation returns bits.
    diagnostics.sampledBitsDimension = dimension;

    ResultPoint[] points;
    if (alignmentPattern == null) {
      points = new ResultPoint[]{bottomLeft, topLeft, topRight};
    } else {
      points = new ResultPoint[]{bottomLeft, topLeft, topRight, alignmentPattern};
    }
    diagnostics.detectedPointCount = points.length;

    // Capture the four-point geometry for a possible checksum-triggered perturbation retry.
    if (GEOMETRY_MODE == GeometryMode.E5_ADAPTIVE && alignmentPattern != null) {
      LAST_SAMPLING_CONTEXT.set(new SamplingContext(image, topLeft, topRight, bottomLeft,
          alignmentPattern, dimension, moduleSize));
    } else {
      LAST_SAMPLING_CONTEXT.set(null);
    }

    return new DetectorResult(bits, points);
  }

  /**
   * Computes the E1-P0 residual with the same integer finder-only prior and
   * two-dimensional local QR basis used by AlignmentWhiteBoxProbe.
   *
   * The residual is diagnostic only in ORIGINAL mode; it does not influence
   * the upstream geometry decision unless P0_GATE is explicitly selected.
   */
  private static void calculateP0Residual(Diagnostics diagnostics,
                                          ResultPoint topLeft,
                                          ResultPoint topRight,
                                          ResultPoint bottomLeft,
                                          ResultPoint alignmentPattern,
                                          int dimension,
                                          int searchPriorX,
                                          int searchPriorY) {
    diagnostics.residualDxPx = alignmentPattern.getX() - searchPriorX;
    diagnostics.residualDyPx = alignmentPattern.getY() - searchPriorY;

    int modulesBetweenFinderCenters = dimension - 7;
    if (modulesBetweenFinderCenters <= 0) {
      return;
    }

    // Local image-space basis vectors, in pixels per QR module.
    double ux = (topRight.getX() - topLeft.getX()) / modulesBetweenFinderCenters;
    double uy = (topRight.getY() - topLeft.getY()) / modulesBetweenFinderCenters;
    double vx = (bottomLeft.getX() - topLeft.getX()) / modulesBetweenFinderCenters;
    double vy = (bottomLeft.getY() - topLeft.getY()) / modulesBetweenFinderCenters;
    double determinant = ux * vy - uy * vx;
    if (Math.abs(determinant) <= 1.0e-9) {
      return;
    }

    diagnostics.residualDxModules =
        (diagnostics.residualDxPx * vy - diagnostics.residualDyPx * vx) / determinant;
    diagnostics.residualDyModules =
        (ux * diagnostics.residualDyPx - uy * diagnostics.residualDxPx) / determinant;
    diagnostics.residualNormModules = Math.hypot(
        diagnostics.residualDxModules, diagnostics.residualDyModules);
  }

  /**
   * E5: reconstructs the projective position of the bottom-right alignment center
   * (module coordinate dimension-6.5) from the three finder centers and their local
   * directional module sizes, without extracting any corner points.
   *
   * The two vanishing points Vx, Vy are derived from the per-finder module-size ratios
   * along the top and left edges; the full homography is then closed-form interpolated.
   * When the scale ratios are ~1 (near-affine) or a scale is unavailable, this degenerates
   * to the original affine alignment-center prior.
   */
  private ResultPoint computeProjectivePrior(ResultPoint topLeft,
                                             ResultPoint topRight,
                                             ResultPoint bottomLeft,
                                             int dimension) {
    float mTLx = sizeOfBlackWhiteBlackRunBothWays(
        (int) topLeft.getX(), (int) topLeft.getY(),
        (int) topRight.getX(), (int) topRight.getY()) / 7.0f;
    float mTRx = sizeOfBlackWhiteBlackRunBothWays(
        (int) topRight.getX(), (int) topRight.getY(),
        (int) topLeft.getX(), (int) topLeft.getY()) / 7.0f;
    float mTLy = sizeOfBlackWhiteBlackRunBothWays(
        (int) topLeft.getX(), (int) topLeft.getY(),
        (int) bottomLeft.getX(), (int) bottomLeft.getY()) / 7.0f;
    float mBLy = sizeOfBlackWhiteBlackRunBothWays(
        (int) bottomLeft.getX(), (int) bottomLeft.getY(),
        (int) topLeft.getX(), (int) topLeft.getY()) / 7.0f;

    Diagnostics diagnostics = LAST_DIAGNOSTICS.get();
    float rX = (isValidScale(mTLx) && isValidScale(mTRx))
        ? (float) Math.sqrt(mTLx / mTRx) : Float.NaN;
    float rY = (isValidScale(mTLy) && isValidScale(mBLy))
        ? (float) Math.sqrt(mTLy / mBLy) : Float.NaN;
    diagnostics.projectiveScaleRatioX = Float.isNaN(rX) ? Double.NaN : rX;
    diagnostics.projectiveScaleRatioY = Float.isNaN(rY) ? Double.NaN : rY;

    // Near-affine detection: when both per-finder scale ratios are within epsilon of unity,
    // the projective correction is dominated by measurement noise while the affine prior is
    // exact. Signal the caller to use the affine path by returning null.
    if (Float.isNaN(rX) || Float.isNaN(rY)
        || (Math.abs(rX - 1.0f) < PERSPECTIVE_RATIO_EPSILON
            && Math.abs(rY - 1.0f) < PERSPECTIVE_RATIO_EPSILON)) {
      return null;
    }

    float[] vx = vanishingPoint(topLeft, topRight, mTLx, mTRx);
    float[] vy = vanishingPoint(topLeft, bottomLeft, mTLy, mBLy);
    if (vx == null || vy == null) {
      return null;
    }

    double ua = (dimension - 10.0) / (dimension - 7.0);
    double lx = (topRight.getX() - topLeft.getX()) / (vx[0] - topRight.getX());
    double ly = (bottomLeft.getY() - topLeft.getY()) / (vy[1] - bottomLeft.getY());

    double px = lx * ua * vx[0] + ly * ua * vy[0] + topLeft.getX();
    double py = lx * ua * vx[1] + ly * ua * vy[1] + topLeft.getY();
    double pw = lx * ua + ly * ua + 1.0;
    if (pw == 0.0 || !Double.isFinite(pw) || !Double.isFinite(px) || !Double.isFinite(py)) {
      return null;
    }
    return new ResultPoint((float) (px / pw), (float) (py / pw));
  }

  private static boolean isValidScale(float value) {
    return Float.isFinite(value) && value > 0.0f;
  }

  /** Returns the vanishing point of line (near, far) from the module-size ratio, or null if affine. */
  private static float[] vanishingPoint(ResultPoint near, ResultPoint far,
                                        float scaleNear, float scaleFar) {
    float r = (float) Math.sqrt(scaleNear / scaleFar);
    if (Math.abs(r - 1.0f) < 1.0e-6f) {
      return null;
    }
    float lambda = r / (r - 1.0f);
    return new float[]{
        near.getX() + lambda * (far.getX() - near.getX()),
        near.getY() + lambda * (far.getY() - near.getY())};
  }

  /** Module-space norm of the pixel residual (detected - reference) under the finder basis. */
  private static double computeResidualModules(ResultPoint detected, ResultPoint reference,
                                               ResultPoint topLeft, ResultPoint topRight,
                                               ResultPoint bottomLeft, int dimension) {
    int modulesBetweenFinderCenters = dimension - 7;
    if (modulesBetweenFinderCenters <= 0) {
      return Double.NaN;
    }
    double dx = detected.getX() - reference.getX();
    double dy = detected.getY() - reference.getY();
    double ux = (topRight.getX() - topLeft.getX()) / modulesBetweenFinderCenters;
    double uy = (topRight.getY() - topLeft.getY()) / modulesBetweenFinderCenters;
    double vx = (bottomLeft.getX() - topLeft.getX()) / modulesBetweenFinderCenters;
    double vy = (bottomLeft.getY() - topLeft.getY()) / modulesBetweenFinderCenters;
    double det = ux * vy - uy * vx;
    if (Math.abs(det) <= 1.0e-9) {
      return Double.NaN;
    }
    double dmx = (dx * vy - dy * vx) / det;
    double dmy = (ux * dy - uy * dx) / det;
    return Math.hypot(dmx, dmy);
  }

  private static PerspectiveTransform createTransform(ResultPoint topLeft,
                                                      ResultPoint topRight,
                                                      ResultPoint bottomLeft,
                                                      ResultPoint alignmentPattern,
                                                      int dimension) {
    float dimMinusThree = dimension - 3.5f;
    float bottomRightX;
    float bottomRightY;
    float sourceBottomRightX;
    float sourceBottomRightY;
    if (alignmentPattern != null) {
      bottomRightX = alignmentPattern.getX();
      bottomRightY = alignmentPattern.getY();
      sourceBottomRightX = dimMinusThree - 3.0f;
      sourceBottomRightY = sourceBottomRightX;
    } else {
      // Don't have an alignment pattern, just make up the bottom-right point
      bottomRightX = (topRight.getX() - topLeft.getX()) + bottomLeft.getX();
      bottomRightY = (topRight.getY() - topLeft.getY()) + bottomLeft.getY();
      sourceBottomRightX = dimMinusThree;
      sourceBottomRightY = dimMinusThree;
    }

    return PerspectiveTransform.quadrilateralToQuadrilateral(
        3.5f,
        3.5f,
        dimMinusThree,
        3.5f,
        sourceBottomRightX,
        sourceBottomRightY,
        3.5f,
        dimMinusThree,
        topLeft.getX(),
        topLeft.getY(),
        topRight.getX(),
        topRight.getY(),
        bottomRightX,
        bottomRightY,
        bottomLeft.getX(),
        bottomLeft.getY());
  }

  private static BitMatrix sampleGrid(BitMatrix image,
                                      PerspectiveTransform transform,
                                      int dimension) throws NotFoundException {

    GridSampler sampler = GridSampler.getInstance();
    return sampler.sampleGrid(image, dimension, dimension, transform);
  }

  /**
   * <p>Computes the dimension (number of modules on a size) of the QR Code based on the position
   * of the finder patterns and estimated module size.</p>
   */
  private static int computeDimension(ResultPoint topLeft,
                                      ResultPoint topRight,
                                      ResultPoint bottomLeft,
                                      float moduleSize) throws NotFoundException {
    int tltrCentersDimension = MathUtils.round(ResultPoint.distance(topLeft, topRight) / moduleSize);
    int tlblCentersDimension = MathUtils.round(ResultPoint.distance(topLeft, bottomLeft) / moduleSize);
    int dimension = ((tltrCentersDimension + tlblCentersDimension) / 2) + 7;
    switch (dimension & 0x03) { // mod 4
      case 0:
        dimension++;
        break;
        // 1? do nothing
      case 2:
        dimension--;
        break;
      case 3:
        dimension -= 2;
        break;
    }
    return dimension;
  }

  /**
   * <p>Computes an average estimated module size based on estimated derived from the positions
   * of the three finder patterns.</p>
   *
   * @param topLeft detected top-left finder pattern center
   * @param topRight detected top-right finder pattern center
   * @param bottomLeft detected bottom-left finder pattern center
   * @return estimated module size
   */
  protected final float calculateModuleSize(ResultPoint topLeft,
                                            ResultPoint topRight,
                                            ResultPoint bottomLeft) {
    // Take the average
    return (calculateModuleSizeOneWay(topLeft, topRight) +
        calculateModuleSizeOneWay(topLeft, bottomLeft)) / 2.0f;
  }

  /**
   * <p>Estimates module size based on two finder patterns -- it uses
   * {@link #sizeOfBlackWhiteBlackRunBothWays(int, int, int, int)} to figure the
   * width of each, measuring along the axis between their centers.</p>
   */
  private float calculateModuleSizeOneWay(ResultPoint pattern, ResultPoint otherPattern) {
    float moduleSizeEst1 = sizeOfBlackWhiteBlackRunBothWays((int) pattern.getX(),
        (int) pattern.getY(),
        (int) otherPattern.getX(),
        (int) otherPattern.getY());
    float moduleSizeEst2 = sizeOfBlackWhiteBlackRunBothWays((int) otherPattern.getX(),
        (int) otherPattern.getY(),
        (int) pattern.getX(),
        (int) pattern.getY());
    if (Float.isNaN(moduleSizeEst1)) {
      return moduleSizeEst2 / 7.0f;
    }
    if (Float.isNaN(moduleSizeEst2)) {
      return moduleSizeEst1 / 7.0f;
    }
    // Average them, and divide by 7 since we've counted the width of 3 black modules,
    // and 1 white and 1 black module on either side. Ergo, divide sum by 14.
    return (moduleSizeEst1 + moduleSizeEst2) / 14.0f;
  }

  /**
   * See {@link #sizeOfBlackWhiteBlackRun(int, int, int, int)}; computes the total width of
   * a finder pattern by looking for a black-white-black run from the center in the direction
   * of another point (another finder pattern center), and in the opposite direction too.
   */
  private float sizeOfBlackWhiteBlackRunBothWays(int fromX, int fromY, int toX, int toY) {

    float result = sizeOfBlackWhiteBlackRun(fromX, fromY, toX, toY);

    // Now count other way -- don't run off image though of course
    float scale = 1.0f;
    int otherToX = fromX - (toX - fromX);
    if (otherToX < 0) {
      scale = fromX / (float) (fromX - otherToX);
      otherToX = 0;
    } else if (otherToX >= image.getWidth()) {
      scale = (image.getWidth() - 1 - fromX) / (float) (otherToX - fromX);
      otherToX = image.getWidth() - 1;
    }
    int otherToY = (int) (fromY - (toY - fromY) * scale);

    scale = 1.0f;
    if (otherToY < 0) {
      scale = fromY / (float) (fromY - otherToY);
      otherToY = 0;
    } else if (otherToY >= image.getHeight()) {
      scale = (image.getHeight() - 1 - fromY) / (float) (otherToY - fromY);
      otherToY = image.getHeight() - 1;
    }
    otherToX = (int) (fromX + (otherToX - fromX) * scale);

    result += sizeOfBlackWhiteBlackRun(fromX, fromY, otherToX, otherToY);

    // Middle pixel is double-counted this way; subtract 1
    return result - 1.0f;
  }

  /**
   * <p>This method traces a line from a point in the image, in the direction towards another point.
   * It begins in a black region, and keeps going until it finds white, then black, then white again.
   * It reports the distance from the start to this point.</p>
   *
   * <p>This is used when figuring out how wide a finder pattern is, when the finder pattern
   * may be skewed or rotated.</p>
   */
  private float sizeOfBlackWhiteBlackRun(int fromX, int fromY, int toX, int toY) {
    // Mild variant of Bresenham's algorithm;
    // see http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
    boolean steep = Math.abs(toY - fromY) > Math.abs(toX - fromX);
    if (steep) {
      int temp = fromX;
      fromX = fromY;
      fromY = temp;
      temp = toX;
      toX = toY;
      toY = temp;
    }

    int dx = Math.abs(toX - fromX);
    int dy = Math.abs(toY - fromY);
    int error = -dx / 2;
    int xstep = fromX < toX ? 1 : -1;
    int ystep = fromY < toY ? 1 : -1;

    // In black pixels, looking for white, first or second time.
    int state = 0;
    // Loop up until x == toX, but not beyond
    int xLimit = toX + xstep;
    for (int x = fromX, y = fromY; x != xLimit; x += xstep) {
      int realX = steep ? y : x;
      int realY = steep ? x : y;

      // Does current pixel mean we have moved white to black or vice versa?
      // Scanning black in state 0,2 and white in state 1, so if we find the wrong
      // color, advance to next state or end if we are in state 2 already
      if ((state == 1) == image.get(realX, realY)) {
        if (state == 2) {
          return MathUtils.distance(x, y, fromX, fromY);
        }
        state++;
      }

      error += dy;
      if (error > 0) {
        if (y == toY) {
          break;
        }
        y += ystep;
        error -= dx;
      }
    }
    // Found black-white-black; give the benefit of the doubt that the next pixel outside the image
    // is "white" so this last point at (toX+xStep,toY) is the right ending. This is really a
    // small approximation; (toX+xStep,toY+yStep) might be really correct. Ignore this.
    if (state == 2) {
      return MathUtils.distance(toX + xstep, toY, fromX, fromY);
    }
    // else we didn't find even black-white-black; no estimate is really possible
    return Float.NaN;
  }

  /**
   * E5 perturbation retry: retains the last sampled four-point geometry and re-samples the grid
   * with the fourth point displaced by small module offsets along the finder basis (u, v). This is
   * the checksum-triggered "local micro-grid correction" used to absorb the sub-module residual
   * error of {@code p_proj} under heavy blur, without paying any cost on images that already
   * decode.
   */
  private static final class SamplingContext {
    private final BitMatrix image;
    private final ResultPoint topLeft;
    private final ResultPoint topRight;
    private final ResultPoint bottomLeft;
    private final ResultPoint fourthPoint;
    private final int dimension;
    private final float moduleSize;

    /** Perturbation step sizes, in modules. */
    private static final float[] PERTURB_STEPS = {0.25f, 0.5f, 1.0f};
    /** Unit offset directions in (u, v) module basis: 4 axis + 4 diagonal. */
    private static final int[][] PERTURB_DIRS = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };

    private SamplingContext(BitMatrix image, ResultPoint topLeft, ResultPoint topRight,
                            ResultPoint bottomLeft, ResultPoint fourthPoint,
                            int dimension, float moduleSize) {
      this.image = image;
      this.topLeft = topLeft;
      this.topRight = topRight;
      this.bottomLeft = bottomLeft;
      this.fourthPoint = fourthPoint;
      this.dimension = dimension;
      this.moduleSize = moduleSize;
    }

    private List<DetectorResult> perturbedDetections() {
      int modulesBetweenFPCenters = dimension - 7;
      if (modulesBetweenFPCenters <= 0) {
        return Collections.emptyList();
      }
      double ux = (topRight.getX() - topLeft.getX()) / modulesBetweenFPCenters;
      double uy = (topRight.getY() - topLeft.getY()) / modulesBetweenFPCenters;
      double vx = (bottomLeft.getX() - topLeft.getX()) / modulesBetweenFPCenters;
      double vy = (bottomLeft.getY() - topLeft.getY()) / modulesBetweenFPCenters;

      List<DetectorResult> out = new ArrayList<>();
      for (float step : PERTURB_STEPS) {
        for (int[] dir : PERTURB_DIRS) {
          addPerturbation(out, step * dir[0], step * dir[1], ux, uy, vx, vy);
        }
      }
      return out;
    }

    private void addPerturbation(List<DetectorResult> out, double du, double dv,
                                 double ux, double uy, double vx, double vy) {
      float nx = (float) (fourthPoint.getX() + du * ux + dv * vx);
      float ny = (float) (fourthPoint.getY() + du * uy + dv * vy);
      if (nx < 0.0f || ny < 0.0f || nx >= image.getWidth() || ny >= image.getHeight()) {
        return;
      }
      AlignmentPattern newFourth = new AlignmentPattern(nx, ny, moduleSize);
      PerspectiveTransform transform =
          createTransform(topLeft, topRight, bottomLeft, newFourth, dimension);
      try {
        BitMatrix bits = sampleGrid(image, transform, dimension);
        ResultPoint[] points = new ResultPoint[]{bottomLeft, topLeft, topRight, newFourth};
        out.add(new DetectorResult(bits, points));
      } catch (NotFoundException ignored) {
        // This offset failed to sample; skip it.
      }
    }
  }

  /**
   * <p>Attempts to locate an alignment pattern in a limited region of the image, which is
   * guessed to contain it. This method uses {@link AlignmentPattern}.</p>
   *
   * @param overallEstModuleSize estimated module size so far
   * @param estAlignmentX x coordinate of center of area probably containing alignment pattern
   * @param estAlignmentY y coordinate of above
   * @param allowanceFactor number of pixels in all directions to search from the center
   * @return {@link AlignmentPattern} if found, or null otherwise
   * @throws NotFoundException if an unexpected error occurs during detection
   */
  protected final AlignmentPattern findAlignmentInRegion(float overallEstModuleSize,
                                                         int estAlignmentX,
                                                         int estAlignmentY,
                                                         float allowanceFactor)
      throws NotFoundException {
    // Look for an alignment pattern (3 modules in size) around where it
    // should be
    int allowance = (int) (allowanceFactor * overallEstModuleSize);
    int alignmentAreaLeftX = Math.max(0, estAlignmentX - allowance);
    int alignmentAreaRightX = Math.min(image.getWidth() - 1, estAlignmentX + allowance);
    if (alignmentAreaRightX - alignmentAreaLeftX < overallEstModuleSize * 3) {
      throw NotFoundException.getNotFoundInstance();
    }

    int alignmentAreaTopY = Math.max(0, estAlignmentY - allowance);
    int alignmentAreaBottomY = Math.min(image.getHeight() - 1, estAlignmentY + allowance);
    if (alignmentAreaBottomY - alignmentAreaTopY < overallEstModuleSize * 3) {
      throw NotFoundException.getNotFoundInstance();
    }

    AlignmentPatternFinder alignmentFinder =
        new AlignmentPatternFinder(
            image,
            alignmentAreaLeftX,
            alignmentAreaTopY,
            alignmentAreaRightX - alignmentAreaLeftX,
            alignmentAreaBottomY - alignmentAreaTopY,
            overallEstModuleSize,
            resultPointCallback);
    return alignmentFinder.find();
  }

}
