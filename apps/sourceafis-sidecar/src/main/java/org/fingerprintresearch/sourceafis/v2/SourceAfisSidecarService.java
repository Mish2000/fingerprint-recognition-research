package org.fingerprintresearch.sourceafis.v2;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class SourceAfisSidecarService {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final TypeReference<Map<String, Object>> JSON_OBJECT = new TypeReference<>() {};
    private static final int DEFAULT_PORT = 8765;
    private static final int MAX_REQUEST_BYTES = 64 * 1024 * 1024;

    private final SourceAfisV2Engine engine;
    private final String bindHost;
    private final int port;

    SourceAfisSidecarService(SourceAfisV2Engine engine, String bindHost, int port) {
        this.engine = engine;
        this.bindHost = bindHost;
        this.port = port;
    }

    public static void main(String[] args) throws IOException {
        String host = env("SOURCEAFIS_HOST", "127.0.0.1");
        validateBindHost(host);
        int port = parsePort(env("SOURCEAFIS_PORT", String.valueOf(DEFAULT_PORT)));
        SourceAfisSidecarService service = new SourceAfisSidecarService(
            new SourceAfisV2Engine(BuildInfo.load()),
            host,
            port
        );
        HttpServer server = HttpServer.create(new InetSocketAddress(host, port), 0);
        service.register(server);
        ExecutorService executor = Executors.newFixedThreadPool(Math.max(2, Runtime.getRuntime().availableProcessors()));
        server.setExecutor(executor);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.stop(0);
            executor.shutdownNow();
        }, "sourceafis-sidecar-shutdown"));
        server.start();
        System.out.printf("SourceAFIS sidecar v2.1 listening on http://%s:%d%n", host, port);
    }

    void register(HttpServer server) {
        server.createContext("/health", exchange -> handle(exchange, "/health", "GET", request -> engine.health(bindHost, port)));
        server.createContext("/extract-template", exchange -> handle(exchange, "/extract-template", "POST", engine::extractTemplate));
        server.createContext("/verify", exchange -> handle(exchange, "/verify", "POST", engine::verify));
    }

    private void handle(HttpExchange exchange, String expectedPath, String expectedMethod, Route route) throws IOException {
        try {
            if (!expectedPath.equals(exchange.getRequestURI().getPath())) {
                writeJson(exchange, 404, error("not_found", "Endpoint not found."));
                return;
            }
            if (!expectedMethod.equalsIgnoreCase(exchange.getRequestMethod())) {
                writeJson(exchange, 405, error("method_not_allowed", "HTTP method is not allowed for this endpoint."));
                return;
            }
            Map<String, Object> request = "GET".equalsIgnoreCase(expectedMethod) ? Map.of() : readJsonObject(exchange);
            writeJson(exchange, 200, route.handle(request));
        } catch (ApiException e) {
            writeJson(exchange, e.statusCode, error(e.code, e.getMessage()));
        } catch (JsonProcessingException e) {
            writeJson(exchange, 400, error("invalid_json", "Request body must be a JSON object."));
        } catch (IOException e) {
            throw e;
        } catch (RuntimeException e) {
            System.err.printf("SourceAFIS v2 sidecar internal error: %s%n", e.getClass().getSimpleName());
            writeJson(exchange, 500, error("internal_error", "SourceAFIS sidecar failed to process the request."));
        } finally {
            exchange.close();
        }
    }

    private Map<String, Object> readJsonObject(HttpExchange exchange) throws IOException {
        byte[] body = readLimited(exchange.getRequestBody(), MAX_REQUEST_BYTES);
        if (body.length == 0) {
            throw new ApiException(400, "invalid_json", "Request body must be a JSON object.");
        }
        Object payload = JSON.readValue(body, Object.class);
        if (!(payload instanceof Map<?, ?>)) {
            throw new ApiException(400, "invalid_json", "Request body must be a JSON object.");
        }
        return JSON.readValue(body, JSON_OBJECT);
    }

    private byte[] readLimited(InputStream input, int maxBytes) throws IOException {
        byte[] buffer = new byte[8192];
        int total = 0;
        try (ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            while (true) {
                int count = input.read(buffer);
                if (count < 0) {
                    return output.toByteArray();
                }
                total += count;
                if (total > maxBytes) {
                    throw new ApiException(413, "request_too_large", "Request body is too large.");
                }
                output.write(buffer, 0, count);
            }
        }
    }

    private void writeJson(HttpExchange exchange, int statusCode, Map<String, Object> payload) throws IOException {
        byte[] response = JSON.writeValueAsBytes(payload);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Cache-Control", "no-store");
        exchange.sendResponseHeaders(statusCode, response.length);
        try (OutputStream output = exchange.getResponseBody()) {
            output.write(response);
        }
    }

    private static Map<String, Object> error(String code, String message) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("error_code", code);
        payload.put("error_message", message);
        return payload;
    }

    private static String env(String name, String fallback) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? fallback : value.trim();
    }

    private static int parsePort(String value) {
        try {
            int port = Integer.parseInt(value);
            if (port > 0 && port <= 65535) {
                return port;
            }
        } catch (NumberFormatException ignored) {
            // Use default below.
        }
        return DEFAULT_PORT;
    }

    static void validateBindHost(String host) {
        String normalized = host == null ? "" : host.trim().toLowerCase(Locale.ROOT);
        if (!normalized.equals("localhost") && !normalized.equals("127.0.0.1") && !normalized.equals("::1")) {
            throw new IllegalArgumentException(
                "SOURCEAFIS_HOST must be localhost, 127.0.0.1, or ::1; remote plain-HTTP binding is forbidden."
            );
        }
    }

    @FunctionalInterface
    private interface Route {
        Map<String, Object> handle(Map<String, Object> request);
    }
}
