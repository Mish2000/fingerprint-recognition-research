package org.fingerprintresearch.sourceafis.v2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.cbor.CBORFactory;
import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class SourceAfisV2Engine {
    static final String METHOD = "sourceafis";
    static final String TEMPLATE_FORMAT = "sourceafis";
    static final String METHOD_INTERNAL_TIMING_UNIT = "milliseconds";
    static final String EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE =
        "FingerprintImageOptions construction, FingerprintImage construction, FingerprintTemplate extraction, " +
        "and FingerprintTemplate.toByteArray serialization; excludes HTTP, JSON, request Base64 decoding, " +
        "and response Base64 encoding.";
    static final String EXTRACT_RAW_TEMPLATE_INTERNAL_TIMING_SCOPE =
        "FingerprintImageOptions construction, raw FingerprintImage construction, FingerprintTemplate extraction, " +
        "FingerprintTemplate.toByteArray serialization, and template SHA-256; excludes HTTP, JSON, request Base64 " +
        "decoding, response Base64 encoding, and response model/JSON serialization.";
    static final String EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE =
        "FingerprintImageOptions construction, raw FingerprintImage construction, FingerprintTemplate extraction, " +
        "FingerprintTemplate.toByteArray serialization, documented native-template CBOR parsing, and response " +
        "model construction; excludes HTTP, JSON, request Base64 decoding, and response JSON serialization.";
    static final String VERIFY_INTERNAL_TIMING_SCOPE =
        "FingerprintTemplate deserialization for both templates, FingerprintMatcher construction, and " +
        "FingerprintMatcher.match; excludes HTTP, JSON, and request Base64 decoding.";
    static final double MIN_DPI = 100.0;
    static final double MAX_DPI = 4000.0;
    static final String RAW_TEMPLATE_ENDPOINT = "/extract-template-raw";
    static final String RAW_TEMPLATE_INPUT = "raw_uint8_grayscale_row_major";
    static final String FINAL_MINUTIAE_ENDPOINT = "/extract-final-minutiae";
    static final String FINAL_MINUTIAE_INPUT = "raw_uint8_grayscale_row_major";
    static final String FINAL_MINUTIAE_COORDINATE_SPACE = "sourceafis_500_dpi_scaled_image";
    static final String FINAL_MINUTIAE_STAGE = "final_template_minutiae";
    static final String FINAL_MINUTIAE_SELECTION_SEMANTICS = "sourceafis_final_selected_minutia_set";
    static final String FINAL_MINUTIAE_ORDER_SEMANTICS =
        "deterministic_sourceafis_template_order_not_quality_ranking";
    private static final ObjectMapper CBOR = new ObjectMapper(new CBORFactory());
    private static final Set<String> NATIVE_TEMPLATE_FIELDS = Set.of(
        "version", "width", "height", "positionsX", "positionsY", "directions", "types"
    );

    private final BuildInfo buildInfo;

    SourceAfisV2Engine(BuildInfo buildInfo) {
        this.buildInfo = buildInfo;
    }

    Map<String, Object> health(String bindHost, int port) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status", "ok");
        response.put("method", METHOD);
        response.put("official_implementation_family", "Java");
        response.put("engine", "SourceAFIS");
        response.put("sourceafis_version", buildInfo.sourceAfisVersion());
        response.put("method_version", buildInfo.sourceAfisVersion());
        response.put("maven_coordinates", buildInfo.sourceAfisMavenCoordinates());
        response.put("template_format", TEMPLATE_FORMAT);
        response.put("template_version", buildInfo.sourceAfisVersion());
        response.put("contract_version", buildInfo.contractVersion());
        response.put("sidecar_implementation_version", buildInfo.implementationVersion());
        response.put("java_runtime_version", System.getProperty("java.runtime.version"));
        response.put("java_runtime_vendor", System.getProperty("java.vendor"));
        response.put("transport", "localhost_http");
        response.put("bind_host", bindHost);
        response.put("port", port);
        response.put("dpi_policy", dpiPolicy());
        response.put("external_preprocessing", "none");
        response.put("template_cache", false);
        response.put("supports_template_extraction", true);
        response.put("supports_raw_template_extraction", true);
        response.put("supports_final_minutiae_extraction", true);
        response.put("supports_pairwise_verification", true);
        response.put("supports_identification", false);
        response.put("method_internal_timing_unit", METHOD_INTERNAL_TIMING_UNIT);
        response.put("extract_template_internal_timing_scope", EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE);
        response.put("extract_raw_template_internal_timing_scope", EXTRACT_RAW_TEMPLATE_INTERNAL_TIMING_SCOPE);
        response.put(
            "extract_final_minutiae_internal_timing_scope",
            EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE
        );
        response.put("verify_internal_timing_scope", VERIFY_INTERNAL_TIMING_SCOPE);
        response.put("raw_template_endpoint", RAW_TEMPLATE_ENDPOINT);
        response.put("raw_template_input", RAW_TEMPLATE_INPUT);
        response.put("final_minutiae_endpoint", FINAL_MINUTIAE_ENDPOINT);
        response.put("final_minutiae_input", FINAL_MINUTIAE_INPUT);
        response.put("final_minutiae_coordinate_space", FINAL_MINUTIAE_COORDINATE_SPACE);
        response.put("final_minutiae_stage", FINAL_MINUTIAE_STAGE);
        return response;
    }

    Map<String, Object> extractTemplate(Map<String, Object> request) {
        byte[] imageBytes = decodeRequiredBase64(stringValue(request.get("image_base64")), "image_base64");
        double dpi = requiredDpi(request.get("dpi"));
        try {
            long started = System.nanoTime();
            FingerprintImageOptions options = new FingerprintImageOptions().dpi(dpi);
            FingerprintTemplate template = new FingerprintTemplate(new FingerprintImage(imageBytes, options));
            byte[] serializedTemplate = template.toByteArray();
            double methodInternalMs = elapsedMilliseconds(started);
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("template_base64", Base64.getEncoder().encodeToString(serializedTemplate));
            response.put("template_format", TEMPLATE_FORMAT);
            response.put("template_version", buildInfo.sourceAfisVersion());
            response.put("sourceafis_version", buildInfo.sourceAfisVersion());
            response.put("effective_dpi", dpi);
            response.put("dpi_source", "request");
            response.put("method_internal_ms", methodInternalMs);
            return response;
        } catch (IllegalArgumentException e) {
            throw new ApiException(422, "invalid_image", "Image bytes are not a valid supported fingerprint image.");
        } catch (RuntimeException e) {
            throw new ApiException(500, "template_extraction_failure", "SourceAFIS template extraction failed.");
        }
    }

    Map<String, Object> verify(Map<String, Object> request) {
        byte[] serializedA = decodeRequiredBase64(
            stringValue(request.get("template_a_base64")),
            "template_a_base64"
        );
        byte[] serializedB = decodeRequiredBase64(
            stringValue(request.get("template_b_base64")),
            "template_b_base64"
        );

        long started = System.nanoTime();
        FingerprintTemplate templateA = templateFromSerialized(serializedA);
        FingerprintTemplate templateB = templateFromSerialized(serializedB);

        try {
            double score = new FingerprintMatcher(templateA).match(templateB);
            double methodInternalMs = elapsedMilliseconds(started);
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("raw_score", score);
            response.put("sourceafis_version", buildInfo.sourceAfisVersion());
            response.put("method_internal_ms", methodInternalMs);
            return response;
        } catch (RuntimeException e) {
            throw new ApiException(500, "comparison_failure", "SourceAFIS comparison failed.");
        }
    }

    Map<String, Object> extractTemplateRaw(Map<String, Object> request) {
        RawImageInput input = rawImageInput(request);
        try {
            long started = System.nanoTime();
            FingerprintImageOptions options = new FingerprintImageOptions().dpi(input.dpi);
            FingerprintImage image = new FingerprintImage(input.width, input.height, input.pixels, options);
            FingerprintTemplate template = new FingerprintTemplate(image);
            byte[] serializedTemplate = template.toByteArray();
            String templateSha256 = sha256(serializedTemplate);
            double methodInternalMs = elapsedMilliseconds(started);
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("template_base64", Base64.getEncoder().encodeToString(serializedTemplate));
            response.put("template_sha256", templateSha256);
            response.put("template_format", TEMPLATE_FORMAT);
            response.put("template_version", buildInfo.sourceAfisVersion());
            response.put("sourceafis_version", buildInfo.sourceAfisVersion());
            response.put("effective_dpi", input.dpi);
            response.put("native_width", input.width);
            response.put("native_height", input.height);
            response.put("method_internal_ms", methodInternalMs);
            return response;
        } catch (IllegalArgumentException e) {
            throw new ApiException(422, "invalid_raw_image", "Raw grayscale dimensions or pixels are invalid.");
        } catch (RuntimeException e) {
            throw new ApiException(500, "raw_template_extraction_failure", "SourceAFIS raw-template extraction failed.");
        }
    }

    Map<String, Object> extractFinalMinutiae(Map<String, Object> request) {
        RawImageInput input = rawImageInput(request);

        try {
            long started = System.nanoTime();
            FingerprintImageOptions options = new FingerprintImageOptions().dpi(input.dpi);
            FingerprintImage image = new FingerprintImage(input.width, input.height, input.pixels, options);
            FingerprintTemplate template = new FingerprintTemplate(image);
            byte[] serializedTemplate = template.toByteArray();
            NativeTemplate parsed = parseNativeTemplate(serializedTemplate);
            Map<String, Object> response = finalMinutiaeResponse(
                parsed,
                serializedTemplate,
                input.width,
                input.height,
                input.dpi
            );
            response.put("method_internal_ms", elapsedMilliseconds(started));
            return response;
        } catch (ApiException e) {
            throw e;
        } catch (IllegalArgumentException e) {
            throw new ApiException(422, "invalid_raw_image", "Raw grayscale dimensions or pixels are invalid.");
        } catch (RuntimeException e) {
            throw new ApiException(500, "final_minutiae_extraction_failure", "SourceAFIS final-minutiae extraction failed.");
        }
    }

    NativeTemplate parseNativeTemplate(byte[] serializedTemplate) {
        final JsonNode root;
        try {
            root = CBOR.readTree(serializedTemplate);
        } catch (IOException | RuntimeException e) {
            throw new ApiException(500, "native_template_cbor_parse_failure", "Cannot parse SourceAFIS native template CBOR.");
        }
        if (root == null || !root.isObject()) {
            throw nativeSchema("Native template CBOR root must be an object.");
        }
        Set<String> actualFields = new HashSet<>();
        root.fieldNames().forEachRemaining(actualFields::add);
        if (!actualFields.equals(NATIVE_TEMPLATE_FIELDS)) {
            throw nativeSchema("Native template CBOR fields do not match the documented 3.18.1 schema.");
        }
        String version = requiredText(root, "version");
        if (!version.startsWith(buildInfo.sourceAfisVersion() + "-")) {
            throw nativeSchema("Native template version is not in the pinned SourceAFIS 3.18.1 family.");
        }
        int width = requiredPositiveCborInteger(root, "width");
        int height = requiredPositiveCborInteger(root, "height");
        JsonNode positionsX = requiredArray(root, "positionsX");
        JsonNode positionsY = requiredArray(root, "positionsY");
        JsonNode directions = requiredArray(root, "directions");
        String types = requiredText(root, "types");
        int count = positionsX.size();
        if (positionsY.size() != count || directions.size() != count || types.length() != count) {
            throw nativeSchema("Native template minutia fields have inconsistent lengths.");
        }
        List<NativeMinutia> minutiae = new ArrayList<>(count);
        for (int index = 0; index < count; ++index) {
            int x = arrayCoordinate(positionsX, index, "positionsX");
            int y = arrayCoordinate(positionsY, index, "positionsY");
            if (x < 0 || x >= width || y < 0 || y >= height) {
                throw nativeSchema("Native template minutia coordinate is outside scaled image bounds.");
            }
            JsonNode directionNode = directions.get(index);
            if (directionNode == null || !directionNode.isNumber()) {
                throw nativeSchema("Native template direction must be numeric.");
            }
            double direction = directionNode.doubleValue();
            if (!Double.isFinite(direction)) {
                throw nativeSchema("Native template direction must be finite.");
            }
            char type = types.charAt(index);
            if (type != 'E' && type != 'B') {
                throw nativeSchema("Native template types may contain only E and B.");
            }
            minutiae.add(new NativeMinutia(index, x, y, (float) direction, type));
        }
        return new NativeTemplate(version, width, height, minutiae);
    }

    private Map<String, Object> finalMinutiaeResponse(
        NativeTemplate template,
        byte[] serializedTemplate,
        int nativeWidth,
        int nativeHeight,
        double dpi
    ) {
        List<Map<String, Object>> minutiae = new ArrayList<>(template.minutiae.size());
        for (NativeMinutia minutia : template.minutiae) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("source_index", minutia.sourceIndex);
            item.put("x_scaled", minutia.x);
            item.put("y_scaled", minutia.y);
            item.put("direction_radians", minutia.direction);
            item.put("type", minutia.type == 'E' ? "ENDING" : "BIFURCATION");
            minutiae.add(item);
        }
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("sourceafis_version", buildInfo.sourceAfisVersion());
        response.put("template_version", template.version);
        response.put("effective_dpi", dpi);
        response.put("native_width", nativeWidth);
        response.put("native_height", nativeHeight);
        response.put("scaled_width", template.width);
        response.put("scaled_height", template.height);
        response.put("coordinate_space", FINAL_MINUTIAE_COORDINATE_SPACE);
        response.put("selection_stage", "sourceafis_final_template_minutiae");
        response.put("selection_semantics", FINAL_MINUTIAE_SELECTION_SEMANTICS);
        response.put("source_order_semantics", FINAL_MINUTIAE_ORDER_SEMANTICS);
        response.put("template_sha256", sha256(serializedTemplate));
        response.put("minutia_count", minutiae.size());
        response.put("minutiae", minutiae);
        return response;
    }

    private FingerprintTemplate templateFromSerialized(byte[] serialized) {
        try {
            return new FingerprintTemplate(serialized);
        } catch (RuntimeException e) {
            throw new ApiException(422, "invalid_serialized_template", "Template bytes are not a valid SourceAFIS template.");
        }
    }

    private double elapsedMilliseconds(long started) {
        long elapsedNanos = System.nanoTime() - started;
        return Math.max(0L, elapsedNanos) / 1_000_000.0;
    }

    private RawImageInput rawImageInput(Map<String, Object> request) {
        int width = requiredPositiveInteger(request.get("width"), "width");
        int height = requiredPositiveInteger(request.get("height"), "height");
        byte[] pixels = decodeRequiredBase64(stringValue(request.get("pixels_base64")), "pixels_base64");
        long expectedPixels = (long) width * (long) height;
        if (pixels.length != expectedPixels) {
            throw new ApiException(
                422,
                "pixel_length_mismatch",
                "Decoded pixels_base64 length must equal width * height exactly."
            );
        }
        return new RawImageInput(width, height, pixels, requiredDpi(request.get("dpi")));
    }

    private int requiredPositiveInteger(Object value, String fieldName) {
        if (value == null) {
            throw new ApiException(400, "missing_field", "Request field " + fieldName + " is required.");
        }
        if (value instanceof Boolean || !(value instanceof Byte || value instanceof Short || value instanceof Integer || value instanceof Long)) {
            throw new ApiException(422, "invalid_dimensions", "Request field " + fieldName + " must be an integer.");
        }
        long parsed = ((Number) value).longValue();
        if (parsed <= 0 || parsed > Integer.MAX_VALUE) {
            throw new ApiException(422, "invalid_dimensions", "Request field " + fieldName + " must be a positive integer.");
        }
        return (int) parsed;
    }

    private String requiredText(JsonNode root, String fieldName) {
        JsonNode value = root.get(fieldName);
        if (value == null || !value.isTextual() || value.textValue().isEmpty()) {
            throw nativeSchema("Native template field " + fieldName + " must be non-empty text.");
        }
        return value.textValue();
    }

    private int requiredPositiveCborInteger(JsonNode root, String fieldName) {
        JsonNode value = root.get(fieldName);
        if (value == null || !value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() <= 0) {
            throw nativeSchema("Native template field " + fieldName + " must be a positive integer.");
        }
        return value.intValue();
    }

    private JsonNode requiredArray(JsonNode root, String fieldName) {
        JsonNode value = root.get(fieldName);
        if (value == null || !value.isArray()) {
            throw nativeSchema("Native template field " + fieldName + " must be an array.");
        }
        return value;
    }

    private int arrayCoordinate(JsonNode array, int index, String fieldName) {
        JsonNode value = array.get(index);
        if (value == null || !value.isIntegralNumber() || !value.canConvertToInt()) {
            throw nativeSchema("Native template field " + fieldName + " must contain integer coordinates.");
        }
        return value.intValue();
    }

    private ApiException nativeSchema(String message) {
        return new ApiException(500, "native_template_schema_mismatch", message);
    }

    private String sha256(byte[] bytes) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
            StringBuilder output = new StringBuilder(digest.length * 2);
            for (byte item : digest) {
                output.append(String.format("%02x", item & 0xff));
            }
            return output.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable in this Java runtime.", e);
        }
    }

    private byte[] decodeRequiredBase64(String encoded, String fieldName) {
        if (encoded == null) {
            throw new ApiException(400, "missing_field", "Request field " + fieldName + " is required.");
        }
        try {
            return Base64.getDecoder().decode(encoded);
        } catch (IllegalArgumentException e) {
            throw new ApiException(400, "invalid_base64", "Request field " + fieldName + " must be valid base64.");
        }
    }

    private double requiredDpi(Object value) {
        if (value == null || stringValue(value) == null) {
            throw new ApiException(400, "missing_dpi", "Request field dpi is required.");
        }
        double dpi;
        if (value instanceof Number) {
            dpi = ((Number) value).doubleValue();
        } else {
            try {
                dpi = Double.parseDouble(String.valueOf(value));
            } catch (NumberFormatException e) {
                throw new ApiException(422, "invalid_dpi", "Request field dpi must be numeric.");
            }
        }
        if (!Double.isFinite(dpi) || dpi < MIN_DPI || dpi > MAX_DPI) {
            throw new ApiException(
                422,
                "invalid_dpi",
                "Request field dpi must be finite and within the documented SourceAFIS adapter policy range."
            );
        }
        return dpi;
    }

    private Map<String, Object> dpiPolicy() {
        Map<String, Object> policy = new LinkedHashMap<>();
        policy.put("required", true);
        policy.put("source", "manifest_or_prepare_metadata");
        policy.put("min_dpi", MIN_DPI);
        policy.put("max_dpi", MAX_DPI);
        policy.put("silent_default", false);
        return policy;
    }

    private String stringValue(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }

    static final class NativeTemplate {
        final String version;
        final int width;
        final int height;
        final List<NativeMinutia> minutiae;

        NativeTemplate(String version, int width, int height, List<NativeMinutia> minutiae) {
            this.version = version;
            this.width = width;
            this.height = height;
            this.minutiae = List.copyOf(minutiae);
        }
    }

    private static final class RawImageInput {
        final int width;
        final int height;
        final byte[] pixels;
        final double dpi;

        RawImageInput(int width, int height, byte[] pixels, double dpi) {
            this.width = width;
            this.height = height;
            this.pixels = pixels;
            this.dpi = dpi;
        }
    }

    static final class NativeMinutia {
        final int sourceIndex;
        final int x;
        final int y;
        final float direction;
        final char type;

        NativeMinutia(int sourceIndex, int x, int y, float direction, char type) {
            this.sourceIndex = sourceIndex;
            this.x = x;
            this.y = y;
            this.direction = direction;
            this.type = type;
        }
    }
}
