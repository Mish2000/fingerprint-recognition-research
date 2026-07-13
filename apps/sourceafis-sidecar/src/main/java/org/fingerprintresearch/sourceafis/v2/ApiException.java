package org.fingerprintresearch.sourceafis.v2;

final class ApiException extends RuntimeException {
    final int statusCode;
    final String code;

    ApiException(int statusCode, String code, String message) {
        super(message);
        this.statusCode = statusCode;
        this.code = code;
    }
}
