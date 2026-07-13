package org.fingerprintresearch.sourceafis.v2;

import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Map;

final class SourceAfisV2Engine {
    static final String METHOD = "sourceafis";
    static final String TEMPLATE_FORMAT = "sourceafis";
    static final String METHOD_INTERNAL_TIMING_UNIT = "milliseconds";
    static final String EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE =
        "FingerprintImageOptions construction, FingerprintImage construction, FingerprintTemplate extraction, " +
        "and FingerprintTemplate.toByteArray serialization; excludes HTTP, JSON, request Base64 decoding, " +
        "and response Base64 encoding.";
    static final String VERIFY_INTERNAL_TIMING_SCOPE =
        "FingerprintTemplate deserialization for both templates, FingerprintMatcher construction, and " +
        "FingerprintMatcher.match; excludes HTTP, JSON, and request Base64 decoding.";
    static final double MIN_DPI = 100.0;
    static final double MAX_DPI = 4000.0;

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
        response.put("supports_pairwise_verification", true);
        response.put("supports_identification", false);
        response.put("method_internal_timing_unit", METHOD_INTERNAL_TIMING_UNIT);
        response.put("extract_template_internal_timing_scope", EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE);
        response.put("verify_internal_timing_scope", VERIFY_INTERNAL_TIMING_SCOPE);
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
}
