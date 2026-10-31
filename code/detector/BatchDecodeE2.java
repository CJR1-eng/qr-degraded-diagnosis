package edu.wuyi.qr;

import com.google.zxing.BinaryBitmap;
import com.google.zxing.ChecksumException;
import com.google.zxing.DecodeHintType;
import com.google.zxing.FormatException;
import com.google.zxing.NotFoundException;
import com.google.zxing.RGBLuminanceSource;
import com.google.zxing.Result;
import com.google.zxing.common.HybridBinarizer;
import com.google.zxing.qrcode.QRCodeReader;
import com.google.zxing.qrcode.detector.Detector;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * E2 batch decoder. It intentionally mirrors the input conversion, QRCodeReader,
 * warm-up, timing boundary, and exception classification of the original
 * BatchDecode.java, while adding per-image Detector geometry telemetry.
 *
 * Usage from project root:
 *   java -Dzxing.e2.geometryMode=ORIGINAL \
 *        -Dzxing.e2.thresholdModules=2.0 \
 *        -cp .\zxing-core-e2\target\classes \
 *        edu.wuyi.qr.BatchDecodeE2 <input-image-dir> <output-csv>
 *
 * Supported geometry modes: ORIGINAL, FORCE_THREE_POINT, P0_GATE.
 * Output files are never overwritten; data is first written to a temporary file.
 */
public final class BatchDecodeE2 {

    private static final int WARMUP_LIMIT = 10;
    private static final List<String> IMAGE_EXTENSIONS =
            List.of(".png", ".jpg", ".jpeg", ".bmp");

    private BatchDecodeE2() {
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: java [system properties] -cp <classes> "
                    + "edu.wuyi.qr.BatchDecodeE2 <input-image-dir> <output.csv>");
            System.exit(2);
        }

        Path inputDir = Paths.get(args[0]).toAbsolutePath().normalize();
        Path outputCsv = Paths.get(args[1]).toAbsolutePath().normalize();
        try {
            run(inputDir, outputCsv);
        } catch (Exception exception) {
            System.err.println("E2 ERROR: " + exception.getMessage());
            exception.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static void run(Path inputDir, Path outputCsv) throws IOException {
        if (!Files.isDirectory(inputDir)) {
            throw new IOException("Input directory does not exist: " + inputDir);
        }
        if (Files.exists(outputCsv)) {
            throw new IOException("Output already exists; refusing to overwrite: " + outputCsv);
        }

        List<Path> imageFiles = new ArrayList<>();
        try (var paths = Files.list(inputDir)) {
            paths.filter(Files::isRegularFile)
                    .filter(BatchDecodeE2::isSupportedImage)
                    .sorted(Comparator.comparing(path ->
                            path.getFileName().toString().toLowerCase(Locale.ROOT)))
                    .forEach(imageFiles::add);
        }
        if (imageFiles.isEmpty()) {
            throw new IOException("No supported image files found in: " + inputDir);
        }

        Map<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
        hints.put(DecodeHintType.TRY_HARDER, Boolean.TRUE);
        hints.put(DecodeHintType.CHARACTER_SET, StandardCharsets.UTF_8.name());
        QRCodeReader reader = new QRCodeReader();

        int warmupCount = Math.min(WARMUP_LIMIT, imageFiles.size());
        for (int i = 0; i < warmupCount; i++) {
            Detector.resetLastDiagnostics();
            decodeOne(imageFiles.get(i), reader, hints);
        }

        Path parent = outputCsv.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path tempCsv = outputCsv.resolveSibling(outputCsv.getFileName() + ".tmp");
        if (Files.exists(tempCsv)) {
            throw new IOException("Temporary output already exists; inspect before retrying: " + tempCsv);
        }

        int decodedCount = 0;
        int fourPointCount = 0;
        int threePointCount = 0;
        int p0RejectedCount = 0;
        int forcedThreePointCount = 0;

        try (BufferedWriter writer = Files.newBufferedWriter(tempCsv, StandardCharsets.UTF_8)) {
            writer.write("\uFEFFfilename,decoder_status,error_type,decoded_payload,elapsed_ms,width,height,"
                    + "detector_reached,finder_patterns_found,candidate_dimension,sampled_bits_dimension,"
                    + "selected_point_count,detected_point_count,"
                    + "alignment_search_attempted,original_alignment_found,alignment_x,alignment_y,"
                    + "search_prior_x,search_prior_y,residual_dx_px,residual_dy_px,"
                    + "residual_dx_modules,residual_dy_modules,residual_norm_modules,"
                    + "selected_geometry,gate_decision,geometry_mode,threshold_modules,grid_sampled,"
                    + "projective_prior_x,projective_prior_y,projective_scale_ratio_x,projective_scale_ratio_y,"
                    + "perturb_retry_outcome,projective_prior_ms,perturb_retry_ms,perturb_candidate_count");
            writer.newLine();

            for (int i = 0; i < imageFiles.size(); i++) {
                Path imageFile = imageFiles.get(i);
                Detector.resetLastDiagnostics();
                long startNanos = System.nanoTime();
                DecodeOutcome outcome = decodeOne(imageFile, reader, hints);
                long elapsedNanos = System.nanoTime() - startNanos;
                double elapsedMs = elapsedNanos / 1_000_000.0;
                Detector.Diagnostics d = Detector.getLastDiagnostics();

                if ("DECODED".equals(outcome.status)) {
                    decodedCount++;
                }
                if ("FOUR_POINT".equals(d.getSelectedGeometry())) {
                    fourPointCount++;
                } else if ("THREE_POINT".equals(d.getSelectedGeometry())) {
                    threePointCount++;
                }
                if ("P0_RESIDUAL_EXCEEDS_THRESHOLD".equals(d.getGateDecision())) {
                    p0RejectedCount++;
                }
                if ("FORCE_THREE_POINT".equals(d.getGateDecision())) {
                    forcedThreePointCount++;
                }

                writeCsvRow(writer,
                        imageFile.getFileName().toString(),
                        outcome.status,
                        outcome.errorType,
                        outcome.payload,
                        String.format(Locale.ROOT, "%.4f", elapsedMs),
                        Integer.toString(outcome.width),
                        Integer.toString(outcome.height),
                        Boolean.toString(d.isDetectorReached()),
                        Boolean.toString(d.isFinderPatternsFound()),
                        Integer.toString(d.getCandidateDimension()),
                        Integer.toString(d.getSampledBitsDimension()),
                        Integer.toString(d.getSelectedPointCount()),
                        Integer.toString(d.getDetectedPointCount()),
                        Boolean.toString(d.isAlignmentSearchAttempted()),
                        Boolean.toString(d.isOriginalAlignmentFound()),
                        fmt(d.getAlignmentX()),
                        fmt(d.getAlignmentY()),
                        fmt(d.getSearchPriorX()),
                        fmt(d.getSearchPriorY()),
                        fmt(d.getResidualDxPx()),
                        fmt(d.getResidualDyPx()),
                        fmt(d.getResidualDxModules()),
                        fmt(d.getResidualDyModules()),
                        fmt(d.getResidualNormModules()),
                        d.getSelectedGeometry(),
                        d.getGateDecision(),
                        Detector.getConfiguredGeometryMode(),
                        String.format(Locale.ROOT, "%.6f", Detector.getConfiguredThresholdModules()),
                        Boolean.toString(d.isGridSampled()),
                        fmt(d.getProjectivePriorX()),
                        fmt(d.getProjectivePriorY()),
                        fmt(d.getProjectiveScaleRatioX()),
                        fmt(d.getProjectiveScaleRatioY()),
                        d.getPerturbRetryOutcome(),
                        fmt(d.getProjectivePriorNanos()),
                        fmt(d.getPerturbRetryNanos()),
                        Integer.toString(d.getPerturbCandidateCount()));

                if ((i + 1) % 100 == 0 || i + 1 == imageFiles.size()) {
                    System.out.println("Processed: " + (i + 1) + "/" + imageFiles.size());
                }
            }
        } catch (Exception e) {
            Files.deleteIfExists(tempCsv);
            throw e;
        }

        try {
            Files.move(tempCsv, outputCsv, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(tempCsv, outputCsv);
        }

        System.out.println("\nE2 batch decoding completed.");
        System.out.println("Input images: " + imageFiles.size());
        System.out.println("Warm-up images (not recorded): " + warmupCount);
        System.out.println("ZXing returned decoded text: " + decodedCount);
        System.out.println("Geometry mode: " + Detector.getConfiguredGeometryMode());
        System.out.println("P0 threshold (modules): " + Detector.getConfiguredThresholdModules());
        System.out.println("Selected FOUR_POINT: " + fourPointCount);
        System.out.println("Selected THREE_POINT: " + threePointCount);
        System.out.println("P0 rejected Alignment candidates: " + p0RejectedCount);
        System.out.println("Forced three-point candidates: " + forcedThreePointCount);
        System.out.println("CSV: " + outputCsv);
        System.out.println("Payload correctness must be checked separately against the manifest.");
    }

    private static DecodeOutcome decodeOne(
            Path imagePath,
            QRCodeReader reader,
            Map<DecodeHintType, ?> hints) {
        int width = 0;
        int height = 0;
        try {
            BufferedImage loaded = ImageIO.read(imagePath.toFile());
            if (loaded == null) {
                throw new IOException("Unsupported or unreadable image format");
            }
            width = loaded.getWidth();
            height = loaded.getHeight();
            int[] argbPixels = loaded.getRGB(0, 0, width, height, null, 0, width);
            RGBLuminanceSource source = new RGBLuminanceSource(width, height, argbPixels);
            BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(source));
            Result result = reader.decode(bitmap, hints);
            return new DecodeOutcome("DECODED", "", result.getText(), width, height);
        } catch (NotFoundException exception) {
            return new DecodeOutcome("FAIL", "NotFoundException", "", width, height);
        } catch (ChecksumException exception) {
            return new DecodeOutcome("FAIL", "ChecksumException", "", width, height);
        } catch (FormatException exception) {
            return new DecodeOutcome("FAIL", "FormatException", "", width, height);
        } catch (IOException exception) {
            return new DecodeOutcome("FAIL", "IOException", "", width, height);
        } catch (RuntimeException exception) {
            return new DecodeOutcome("FAIL", exception.getClass().getSimpleName(), "", width, height);
        } finally {
            reader.reset();
        }
    }

    private static boolean isSupportedImage(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return IMAGE_EXTENSIONS.stream().anyMatch(name::endsWith);
    }

    private static String fmt(double value) {
        if (!Double.isFinite(value)) {
            return "";
        }
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static void writeCsvRow(BufferedWriter writer, String... values) throws IOException {
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                writer.write(',');
            }
            writer.write(csvEscape(values[i]));
        }
        writer.newLine();
    }

    private static String csvEscape(String value) {
        String safe = value == null ? "" : value;
        if (safe.contains(",") || safe.contains("\"") || safe.contains("\n") || safe.contains("\r")) {
            return "\"" + safe.replace("\"", "\"\"") + "\"";
        }
        return safe;
    }

    private static final class DecodeOutcome {
        private final String status;
        private final String errorType;
        private final String payload;
        private final int width;
        private final int height;

        private DecodeOutcome(String status, String errorType, String payload, int width, int height) {
            this.status = status;
            this.errorType = errorType;
            this.payload = payload;
            this.width = width;
            this.height = height;
        }
    }
}
