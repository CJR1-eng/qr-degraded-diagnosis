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
import java.io.BufferedReader;
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
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * E5 Step 1: GT_ALIGN_ORACLE harness. For every image where the three Finder
 * patterns are found, it replaces the detected (or missing) Alignment pattern
 * with the ground-truth alignment center read from the E4 manifest, isolating
 * the pure four-point geometry upper bound.
 *
 * Usage from project root:
 *   java -Dzxing.e2.geometryMode=GT_ALIGN_ORACLE \
 *        -cp .\zxing-core-e2\target\classes \
 *        edu.wuyi.qr.GtAlignOracle <manifest.csv> <input-image-dir> <output.csv>
 *
 * Output columns mirror BatchDecodeE2 so the result can be merged downstream.
 */
public final class GtAlignOracle {

    private static final int WARMUP_LIMIT = 10;
    private static final List<String> IMAGE_EXTENSIONS =
            List.of(".png", ".jpg", ".jpeg", ".bmp");

    private GtAlignOracle() {
    }

    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: java -Dzxing.e2.geometryMode=GT_ALIGN_ORACLE "
                    + "-cp <classes> edu.wuyi.qr.GtAlignOracle "
                    + "<manifest.csv> <input-image-dir> <output.csv>");
            System.exit(2);
        }
        Path manifestCsv = Paths.get(args[0]).toAbsolutePath().normalize();
        Path inputDir = Paths.get(args[1]).toAbsolutePath().normalize();
        Path outputCsv = Paths.get(args[2]).toAbsolutePath().normalize();
        try {
            run(manifestCsv, inputDir, outputCsv);
        } catch (Exception exception) {
            System.err.println("GT_ALIGN_ORACLE ERROR: " + exception.getMessage());
            exception.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static void run(Path manifestCsv, Path inputDir, Path outputCsv) throws IOException {
        if (!Files.isRegularFile(manifestCsv)) {
            throw new IOException("Manifest does not exist: " + manifestCsv);
        }
        if (!Files.isDirectory(inputDir)) {
            throw new IOException("Input directory does not exist: " + inputDir);
        }
        if (Files.exists(outputCsv)) {
            throw new IOException("Output already exists; refusing to overwrite: " + outputCsv);
        }

        Map<String, float[]> gtAlignment = loadGtAlignment(manifestCsv);

        List<Path> imageFiles = new ArrayList<>();
        try (var paths = Files.list(inputDir)) {
            paths.filter(Files::isRegularFile)
                    .filter(GtAlignOracle::isSupportedImage)
                    .sorted(Comparator.comparing(path ->
                            path.getFileName().toString().toLowerCase(Locale.ROOT)))
                    .forEach(imageFiles::add);
        }
        if (imageFiles.isEmpty()) {
            throw new IOException("No supported image files found in: " + inputDir);
        }
        int missingGt = 0;
        for (Path imageFile : imageFiles) {
            if (!gtAlignment.containsKey(imageFile.getFileName().toString())) {
                missingGt++;
            }
        }
        if (missingGt > 0) {
            throw new IOException("Missing ground-truth alignment for " + missingGt + " images");
        }

        Map<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
        hints.put(DecodeHintType.TRY_HARDER, Boolean.TRUE);
        hints.put(DecodeHintType.CHARACTER_SET, StandardCharsets.UTF_8.name());
        QRCodeReader reader = new QRCodeReader();

        int warmupCount = Math.min(WARMUP_LIMIT, imageFiles.size());
        for (int i = 0; i < warmupCount; i++) {
            float[] gt = gtAlignment.get(imageFiles.get(i).getFileName().toString());
            Detector.resetLastDiagnostics();
            Detector.setGtAlignment(gt[0], gt[1]);
            decodeOne(imageFiles.get(i), reader, hints);
            Detector.clearGtAlignment();
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
        int injectedCount = 0;

        try (BufferedWriter writer = Files.newBufferedWriter(tempCsv, StandardCharsets.UTF_8)) {
            writer.write("\uFEFFfilename,decoder_status,error_type,decoded_payload,elapsed_ms,width,height,"
                    + "detector_reached,finder_patterns_found,candidate_dimension,sampled_bits_dimension,"
                    + "selected_point_count,detected_point_count,"
                    + "alignment_search_attempted,original_alignment_found,alignment_x,alignment_y,"
                    + "search_prior_x,search_prior_y,residual_dx_px,residual_dy_px,"
                    + "residual_dx_modules,residual_dy_modules,residual_norm_modules,"
                    + "selected_geometry,gate_decision,geometry_mode,threshold_modules,grid_sampled,"
                    + "gt_x,gt_y");
            writer.newLine();

            for (int i = 0; i < imageFiles.size(); i++) {
                Path imageFile = imageFiles.get(i);
                float[] gt = gtAlignment.get(imageFile.getFileName().toString());
                Detector.resetLastDiagnostics();
                Detector.setGtAlignment(gt[0], gt[1]);
                long startNanos = System.nanoTime();
                DecodeOutcome outcome = decodeOne(imageFile, reader, hints);
                long elapsedNanos = System.nanoTime() - startNanos;
                Detector.clearGtAlignment();
                double elapsedMs = elapsedNanos / 1_000_000.0;
                Detector.Diagnostics d = Detector.getLastDiagnostics();

                if ("DECODED".equals(outcome.status)) {
                    decodedCount++;
                }
                if ("GT_ALIGNMENT_INJECTED".equals(d.getGateDecision())) {
                    injectedCount++;
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
                        fmt(gt[0]),
                        fmt(gt[1]));

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

        System.out.println("\nGT_ALIGN_ORACLE batch completed.");
        System.out.println("Input images: " + imageFiles.size());
        System.out.println("GT alignment injected: " + injectedCount);
        System.out.println("ZXing returned decoded text: " + decodedCount);
        System.out.println("CSV: " + outputCsv);
        System.out.println("Payload correctness must be checked separately against the manifest.");
    }

    private static Map<String, float[]> loadGtAlignment(Path manifestCsv) throws IOException {
        Map<String, float[]> map = new HashMap<>();
        try (BufferedReader reader = Files.newBufferedReader(manifestCsv, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null) {
                throw new IOException("Empty manifest: " + manifestCsv);
            }
            header = header.replace("\uFEFF", "");
            String[] columns = header.split(",");
            int filenameIdx = -1;
            int gtXIdx = -1;
            int gtYIdx = -1;
            for (int i = 0; i < columns.length; i++) {
                String col = columns[i].trim();
                if (col.equals("image_path")) {
                    filenameIdx = i;
                } else if (col.equals("gt_alignment_dst_x")) {
                    gtXIdx = i;
                } else if (col.equals("gt_alignment_dst_y")) {
                    gtYIdx = i;
                }
            }
            if (filenameIdx < 0 || gtXIdx < 0 || gtYIdx < 0) {
                throw new IOException("Manifest missing required columns: image_path, "
                        + "gt_alignment_dst_x, gt_alignment_dst_y");
            }
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                String[] fields = splitCsv(line);
                String imagePath = fields[filenameIdx].replace("\\", "/");
                String filename = imagePath.substring(imagePath.lastIndexOf('/') + 1);
                float x = Float.parseFloat(fields[gtXIdx]);
                float y = Float.parseFloat(fields[gtYIdx]);
                map.put(filename, new float[]{x, y});
            }
        }
        return map;
    }

    private static String[] splitCsv(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        current.append('"');
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    current.append(c);
                }
            } else if (c == '"') {
                inQuotes = true;
            } else if (c == ',') {
                fields.add(current.toString());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        fields.add(current.toString());
        return fields.toArray(new String[0]);
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
