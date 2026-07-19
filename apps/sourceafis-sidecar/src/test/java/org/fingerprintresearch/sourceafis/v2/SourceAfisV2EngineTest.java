package org.fingerprintresearch.sourceafis.v2;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.cbor.CBORFactory;
import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintTemplate;
import com.machinezoo.sourceafis.FingerprintTransparency;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SourceAfisV2EngineTest {
    private static final ObjectMapper CBOR = new ObjectMapper(new CBORFactory());
    private final SourceAfisV2Engine engine = new SourceAfisV2Engine(BuildInfo.load());

    @Test
    void healthReportsOfficialJavaRuntimeAndPinnedBuildVersion() {
        Map<String, Object> health = engine.health("127.0.0.1", 8765);

        assertEquals("ok", health.get("status"));
        assertEquals("sourceafis", health.get("method"));
        assertEquals("Java", health.get("official_implementation_family"));
        assertEquals("3.18.1", health.get("sourceafis_version"));
        assertEquals("3.18.1", health.get("method_version"));
        assertEquals("com.machinezoo.sourceafis:sourceafis:3.18.1", health.get("maven_coordinates"));
        assertEquals("sourceafis-sidecar-v2.3", health.get("contract_version"));
        assertEquals("0.4.0", health.get("sidecar_implementation_version"));
        assertEquals("sourceafis", health.get("template_format"));
        assertEquals("3.18.1", health.get("template_version"));
        assertEquals("localhost_http", health.get("transport"));
        assertEquals("none", health.get("external_preprocessing"));
        assertEquals(false, health.get("template_cache"));
        assertEquals(true, health.get("supports_template_extraction"));
        assertEquals(true, health.get("supports_raw_template_extraction"));
        assertEquals(true, health.get("supports_final_minutiae_extraction"));
        assertEquals(true, health.get("supports_pairwise_verification"));
        assertEquals(false, health.get("supports_identification"));
        assertEquals("milliseconds", health.get("method_internal_timing_unit"));
        assertEquals(
            SourceAfisV2Engine.EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE,
            health.get("extract_template_internal_timing_scope")
        );
        assertEquals(
            SourceAfisV2Engine.EXTRACT_RAW_TEMPLATE_INTERNAL_TIMING_SCOPE,
            health.get("extract_raw_template_internal_timing_scope")
        );
        assertEquals(
            SourceAfisV2Engine.EXTRACT_FINAL_MINUTIAE_INTERNAL_TIMING_SCOPE,
            health.get("extract_final_minutiae_internal_timing_scope")
        );
        assertEquals(SourceAfisV2Engine.VERIFY_INTERNAL_TIMING_SCOPE, health.get("verify_internal_timing_scope"));
        assertEquals("/extract-template-raw", health.get("raw_template_endpoint"));
        assertEquals("raw_uint8_grayscale_row_major", health.get("raw_template_input"));
        assertEquals("/extract-final-minutiae", health.get("final_minutiae_endpoint"));
        assertEquals("raw_uint8_grayscale_row_major", health.get("final_minutiae_input"));
        assertEquals("sourceafis_500_dpi_scaled_image", health.get("final_minutiae_coordinate_space"));
        assertEquals("final_template_minutiae", health.get("final_minutiae_stage"));
        assertTrue(String.valueOf(health.get("java_runtime_version")).length() > 0);
        assertTrue(String.valueOf(health.get("java_runtime_vendor")).length() > 0);
    }

    @Test
    void extractsTemplatesAndReturnsRawPairwiseScoreOnly() throws IOException {
        String template = extractTemplateBase64(syntheticFingerprintPng(0), 1000);

        Map<String, Object> verification = engine.verify(Map.of(
            "template_a_base64", template,
            "template_b_base64", template
        ));

        assertInstanceOf(Number.class, verification.get("raw_score"));
        assertTrue(Double.isFinite(((Number) verification.get("raw_score")).doubleValue()));
        assertNonnegativeFiniteTiming(verification);
        assertFalse(verification.containsKey("normalized_score"));
        assertFalse(verification.containsKey("threshold"));
        assertFalse(verification.containsKey("decision"));
    }

    @Test
    void dpiIsRequiredAndAffectsExtraction() throws IOException {
        byte[] image = syntheticFingerprintPng(1);

        Map<String, Object> dpi1000 = extractTemplateResponse(image, 1000);
        Map<String, Object> dpi2000 = extractTemplateResponse(image, 2000);

        assertEquals(2000.0, ((Number) dpi2000.get("effective_dpi")).doubleValue());
        assertEquals("request", dpi2000.get("dpi_source"));
        assertNonnegativeFiniteTiming(dpi1000);
        assertNonnegativeFiniteTiming(dpi2000);
        assertNotEquals(dpi1000.get("template_base64"), dpi2000.get("template_base64"));

        ApiException missing = assertThrows(ApiException.class, () -> engine.extractTemplate(Map.of(
            "image_base64", Base64.getEncoder().encodeToString(image)
        )));
        assertEquals("missing_dpi", missing.code);
    }

    @Test
    void invalidDpiAndInvalidTemplatesFailExplicitly() {
        ApiException invalidDpi = assertThrows(ApiException.class, () -> engine.extractTemplate(Map.of(
            "image_base64", Base64.getEncoder().encodeToString("not an image".getBytes(java.nio.charset.StandardCharsets.UTF_8)),
            "dpi", -1
        )));
        assertEquals(422, invalidDpi.statusCode);
        assertEquals("invalid_dpi", invalidDpi.code);

        String bogusTemplate = Base64.getEncoder().encodeToString("not a template".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        ApiException invalidTemplate = assertThrows(ApiException.class, () -> engine.verify(Map.of(
            "template_a_base64", bogusTemplate,
            "template_b_base64", bogusTemplate
        )));
        assertEquals(422, invalidTemplate.statusCode);
        assertEquals("invalid_serialized_template", invalidTemplate.code);
    }

    @Test
    void invalidBase64ReturnsStructuredError() {
        ApiException error = assertThrows(ApiException.class, () -> engine.extractTemplate(Map.of(
            "image_base64", "not base64",
            "dpi", 1000
        )));

        assertEquals(400, error.statusCode);
        assertEquals("invalid_base64", error.code);
    }

    @Test
    void extractsFinalMinutiaeFromExactRawGrayscaleDeterministically() {
        BufferedImage image = syntheticFingerprintImage(2);
        byte[] pixels = rawGrayscalePixels(image);
        Map<String, Object> request = rawRequest(image, pixels, 1000);

        Map<String, Object> first = engine.extractFinalMinutiae(request);
        Map<String, Object> second = engine.extractFinalMinutiae(request);

        assertEquals("3.18.1", first.get("sourceafis_version"));
        assertEquals("3.18.1-java", first.get("template_version"));
        assertEquals(image.getWidth(), first.get("native_width"));
        assertEquals(image.getHeight(), first.get("native_height"));
        assertTrue(((Number) first.get("scaled_width")).intValue() > 0);
        assertTrue(((Number) first.get("scaled_height")).intValue() > 0);
        assertEquals("sourceafis_500_dpi_scaled_image", first.get("coordinate_space"));
        assertEquals("sourceafis_final_template_minutiae", first.get("selection_stage"));
        assertEquals("sourceafis_final_selected_minutia_set", first.get("selection_semantics"));
        assertEquals(
            "deterministic_sourceafis_template_order_not_quality_ranking",
            first.get("source_order_semantics")
        );
        assertTrue(String.valueOf(first.get("template_sha256")).matches("[0-9a-f]{64}"));
        assertEquals(first.get("template_sha256"), second.get("template_sha256"));
        assertEquals(first.get("minutiae"), second.get("minutiae"));
        assertEquals(first.get("minutia_count"), ((List<?>) first.get("minutiae")).size());
        assertNonnegativeFiniteTiming(first);
        assertFalse(first.containsKey("template_base64"));
        assertFalse(first.containsKey("threshold"));
        assertFalse(first.containsKey("decision"));
    }

    @Test
    void rawTemplateMatchesFinalMinutiaeTemplateForTheSamePixelsAndDpi() {
        BufferedImage image = syntheticFingerprintImage(2);
        byte[] pixels = rawGrayscalePixels(image);
        Map<String, Object> request = rawRequest(image, pixels, 1000);

        Map<String, Object> first = engine.extractTemplateRaw(request);
        Map<String, Object> second = engine.extractTemplateRaw(request);
        Map<String, Object> finalMinutiae = engine.extractFinalMinutiae(request);

        assertEquals("sourceafis", first.get("template_format"));
        assertEquals("3.18.1", first.get("template_version"));
        assertEquals("3.18.1", first.get("sourceafis_version"));
        assertEquals(1000.0, ((Number) first.get("effective_dpi")).doubleValue());
        assertEquals(image.getWidth(), first.get("native_width"));
        assertEquals(image.getHeight(), first.get("native_height"));
        assertTrue(String.valueOf(first.get("template_sha256")).matches("[0-9a-f]{64}"));
        assertEquals(first.get("template_base64"), second.get("template_base64"));
        assertEquals(first.get("template_sha256"), second.get("template_sha256"));
        assertEquals(first.get("template_sha256"), finalMinutiae.get("template_sha256"));
        assertNonnegativeFiniteTiming(first);
        assertFalse(first.containsKey("threshold"));
        assertFalse(first.containsKey("decision"));
    }

    @Test
    void rawFinalMinutiaeValidationFailsWithSpecificCodes() {
        BufferedImage image = syntheticFingerprintImage(3);
        byte[] pixels = rawGrayscalePixels(image);

        ApiException invalidWidth = assertThrows(ApiException.class, () -> engine.extractFinalMinutiae(Map.of(
            "width", 0,
            "height", image.getHeight(),
            "pixels_base64", Base64.getEncoder().encodeToString(pixels),
            "dpi", 1000
        )));
        assertEquals("invalid_dimensions", invalidWidth.code);

        ApiException nonIntegerWidth = assertThrows(ApiException.class, () -> engine.extractFinalMinutiae(Map.of(
            "width", 360.0,
            "height", image.getHeight(),
            "pixels_base64", Base64.getEncoder().encodeToString(pixels),
            "dpi", 1000
        )));
        assertEquals("invalid_dimensions", nonIntegerWidth.code);

        ApiException lengthMismatch = assertThrows(ApiException.class, () -> engine.extractFinalMinutiae(Map.of(
            "width", image.getWidth(),
            "height", image.getHeight(),
            "pixels_base64", Base64.getEncoder().encodeToString(new byte[pixels.length - 1]),
            "dpi", 1000
        )));
        assertEquals("pixel_length_mismatch", lengthMismatch.code);

        ApiException invalidBase64 = assertThrows(ApiException.class, () -> engine.extractFinalMinutiae(Map.of(
            "width", image.getWidth(),
            "height", image.getHeight(),
            "pixels_base64", "not base64",
            "dpi", 1000
        )));
        assertEquals("invalid_base64", invalidBase64.code);

        ApiException invalidDpi = assertThrows(ApiException.class, () -> engine.extractFinalMinutiae(Map.of(
            "width", image.getWidth(),
            "height", image.getHeight(),
            "pixels_base64", Base64.getEncoder().encodeToString(pixels),
            "dpi", Double.NaN
        )));
        assertEquals("invalid_dpi", invalidDpi.code);
    }

    @Test
    void nativeTemplateParserRejectsMalformedStructures() throws IOException {
        List<Object> malformed = List.of(
            List.of(1, 2, 3),
            Map.of("version", "3.18.1-java"),
            nativeTemplateFixture(Map.of("positionsY", List.of(1, 2))),
            nativeTemplateFixture(Map.of("types", "X")),
            nativeTemplateFixture(Map.of("positionsX", List.of(999)))
        );
        for (Object payload : malformed) {
            ApiException error = assertThrows(
                ApiException.class,
                () -> engine.parseNativeTemplate(CBOR.writeValueAsBytes(payload))
            );
            assertTrue(error.code.startsWith("native_template_"));
        }

        ApiException invalidCbor = assertThrows(
            ApiException.class,
            () -> engine.parseNativeTemplate(new byte[] {(byte) 0xff, (byte) 0xff})
        );
        assertEquals("native_template_cbor_parse_failure", invalidCbor.code);
    }

    @Test
    void topMinutiaeSetEqualsFinalNativeTemplateSet() throws IOException {
        for (int variant : List.of(0, 1, 2)) {
            BufferedImage image = syntheticFingerprintImage(variant);
            byte[] pixels = rawGrayscalePixels(image);
            TopMinutiaeCapture capture = new TopMinutiaeCapture();
            FingerprintTemplate template;
            try (TopMinutiaeCapture ignored = capture) {
                template = new FingerprintTemplate(new FingerprintImage(
                    image.getWidth(),
                    image.getHeight(),
                    pixels,
                    new FingerprintImageOptions().dpi(1000)
                ));
            }
            assertTrue(capture.topMinutiae != null && capture.topMinutiae.length > 0);
            Set<String> topSet = canonicalTopSet(capture.topMinutiae);
            Set<String> finalSet = canonicalNativeSet(engine.parseNativeTemplate(template.toByteArray()));
            assertEquals(topSet, finalSet);
        }
    }

    @Test
    void sidecarBindingIsRestrictedToExplicitLoopbackHosts() {
        assertDoesNotThrow(() -> SourceAfisSidecarService.validateBindHost("localhost"));
        assertDoesNotThrow(() -> SourceAfisSidecarService.validateBindHost("127.0.0.1"));
        assertDoesNotThrow(() -> SourceAfisSidecarService.validateBindHost("::1"));
        assertThrows(IllegalArgumentException.class, () -> SourceAfisSidecarService.validateBindHost("0.0.0.0"));
        assertThrows(IllegalArgumentException.class, () -> SourceAfisSidecarService.validateBindHost("192.168.1.5"));
    }

    private void assertNonnegativeFiniteTiming(Map<String, Object> response) {
        assertInstanceOf(Number.class, response.get("method_internal_ms"));
        double timing = ((Number) response.get("method_internal_ms")).doubleValue();
        assertTrue(Double.isFinite(timing));
        assertTrue(timing >= 0.0);
    }

    private String extractTemplateBase64(byte[] image, int dpi) {
        Map<String, Object> response = extractTemplateResponse(image, dpi);
        String template = (String) response.get("template_base64");
        assertTrue(template.length() > 10);
        return template;
    }

    private Map<String, Object> extractTemplateResponse(byte[] image, int dpi) {
        return engine.extractTemplate(Map.of(
            "image_base64", Base64.getEncoder().encodeToString(image),
            "dpi", dpi
        ));
    }

    private byte[] syntheticFingerprintPng(int variant) throws IOException {
        BufferedImage image = syntheticFingerprintImage(variant);
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ImageIO.write(image, "png", output);
        return output.toByteArray();
    }

    private BufferedImage syntheticFingerprintImage(int variant) {
        int width = 360;
        int height = 460;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics = image.createGraphics();
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, width, height);
        graphics.setColor(Color.BLACK);
        graphics.setStroke(new BasicStroke(3.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        for (int ridge = 0; ridge < 34; ridge++) {
            double y = 55 + ridge * 10.5;
            double phase = variant * 0.35 + ridge * 0.18;
            Path2D path = new Path2D.Double();
            path.moveTo(42, y + Math.sin(phase) * 10);
            for (int x = 62; x <= width - 42; x += 28) {
                double curveY = y + Math.sin(x * 0.035 + phase) * 18;
                path.lineTo(x, curveY);
            }
            graphics.draw(path);
        }
        graphics.dispose();
        return image;
    }

    private byte[] rawGrayscalePixels(BufferedImage image) {
        return ((DataBufferByte) image.getRaster().getDataBuffer()).getData().clone();
    }

    private Map<String, Object> rawRequest(BufferedImage image, byte[] pixels, int dpi) {
        return Map.of(
            "width", image.getWidth(),
            "height", image.getHeight(),
            "pixels_base64", Base64.getEncoder().encodeToString(pixels),
            "dpi", dpi
        );
    }

    private Map<String, Object> nativeTemplateFixture(Map<String, Object> replacements) {
        Map<String, Object> payload = new java.util.LinkedHashMap<>();
        payload.put("version", "3.18.1-java");
        payload.put("width", 10);
        payload.put("height", 10);
        payload.put("positionsX", List.of(1));
        payload.put("positionsY", List.of(1));
        payload.put("directions", List.of(1.25f));
        payload.put("types", "E");
        payload.putAll(replacements);
        return payload;
    }

    private Set<String> canonicalNativeSet(SourceAfisV2Engine.NativeTemplate template) {
        Set<String> result = new HashSet<>();
        for (SourceAfisV2Engine.NativeMinutia minutia : template.minutiae) {
            result.add(canonicalMinutia(minutia.x, minutia.y, minutia.type == 'E' ? "ENDING" : "BIFURCATION", minutia.direction));
        }
        return result;
    }

    private Set<String> canonicalTopSet(byte[] topMinutiae) throws IOException {
        JsonNode root = CBOR.readTree(topMinutiae);
        assertTrue(root.isObject());
        JsonNode items = root.get("minutiae");
        assertTrue(items.isArray());
        Set<String> result = new HashSet<>();
        for (JsonNode item : items) {
            JsonNode position = item.get("position");
            result.add(canonicalMinutia(
                position.get("x").intValue(),
                position.get("y").intValue(),
                item.get("type").textValue(),
                item.get("direction").floatValue()
            ));
        }
        return result;
    }

    private String canonicalMinutia(int x, int y, String type, float direction) {
        return String.format(
            Locale.ROOT,
            "%d:%d:%s:%08x",
            x,
            y,
            type,
            Float.floatToIntBits(direction)
        );
    }

    private static final class TopMinutiaeCapture extends FingerprintTransparency {
        private byte[] topMinutiae;

        @Override
        public boolean accepts(String key) {
            return "top-minutiae".equals(key);
        }

        @Override
        public synchronized void take(String key, String mime, byte[] data) {
            if (!accepts(key)) {
                throw new AssertionError("Unexpected transparency key: " + key);
            }
            if (!"application/cbor".equals(mime)) {
                throw new AssertionError("Unexpected top-minutiae MIME type: " + mime);
            }
            if (topMinutiae != null) {
                throw new AssertionError("top-minutiae was emitted more than once.");
            }
            topMinutiae = data.clone();
        }
    }
}
