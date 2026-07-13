package org.fingerprintresearch.sourceafis.v2;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SourceAfisV2EngineTest {
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
        assertEquals("sourceafis-sidecar-v2.1", health.get("contract_version"));
        assertEquals("0.2.0", health.get("sidecar_implementation_version"));
        assertEquals("sourceafis", health.get("template_format"));
        assertEquals("3.18.1", health.get("template_version"));
        assertEquals("localhost_http", health.get("transport"));
        assertEquals("none", health.get("external_preprocessing"));
        assertEquals(false, health.get("template_cache"));
        assertEquals(true, health.get("supports_template_extraction"));
        assertEquals(true, health.get("supports_pairwise_verification"));
        assertEquals(false, health.get("supports_identification"));
        assertEquals("milliseconds", health.get("method_internal_timing_unit"));
        assertEquals(
            SourceAfisV2Engine.EXTRACT_TEMPLATE_INTERNAL_TIMING_SCOPE,
            health.get("extract_template_internal_timing_scope")
        );
        assertEquals(SourceAfisV2Engine.VERIFY_INTERNAL_TIMING_SCOPE, health.get("verify_internal_timing_scope"));
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

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ImageIO.write(image, "png", output);
        return output.toByteArray();
    }
}
