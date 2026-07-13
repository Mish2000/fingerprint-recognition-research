package org.fingerprintresearch.sourceafis.v2;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

final class BuildInfo {
    private final Properties properties;

    private BuildInfo(Properties properties) {
        this.properties = properties;
    }

    static BuildInfo load() {
        Properties properties = new Properties();
        try (InputStream input = BuildInfo.class.getResourceAsStream("/sourceafis-sidecar.properties")) {
            if (input == null) {
                throw new IllegalStateException("sourceafis-sidecar.properties is missing from the runtime artifact.");
            }
            properties.load(input);
        } catch (IOException e) {
            throw new IllegalStateException("Cannot read sourceafis-sidecar.properties.", e);
        }
        return new BuildInfo(properties);
    }

    String sourceAfisVersion() {
        return required("sourceafis.version");
    }

    String sourceAfisMavenCoordinates() {
        return required("sourceafis.maven.coordinates");
    }

    String contractVersion() {
        return required("sidecar.contract.version");
    }

    String implementationVersion() {
        return required("sidecar.implementation.version");
    }

    private String required(String key) {
        String value = properties.getProperty(key);
        if (value == null || value.isBlank() || value.contains("${")) {
            throw new IllegalStateException("Build property " + key + " is missing or unfiltered.");
        }
        return value.trim();
    }
}
