package io.github.massimilianopili.mcp.metacognition;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Minimal Ollama embedding client using JDK HttpClient.
 * Calls /api/embed endpoint, parses response manually (no Jackson dependency).
 * Blocking — callers wrap in Mono.fromCallable().
 */
public class OllamaEmbeddingClient {

    private static final Logger log = LoggerFactory.getLogger(OllamaEmbeddingClient.class);

    private final String baseUrl;
    private final String model;
    private final int expectedDimensions;
    private final int requestTimeoutSec;
    private final HttpClient httpClient;

    public OllamaEmbeddingClient(String baseUrl, String model, int expectedDimensions,
                                 int connectTimeoutSec, int requestTimeoutSec) {
        this.baseUrl = baseUrl;
        this.model = model;
        this.expectedDimensions = expectedDimensions;
        this.requestTimeoutSec = requestTimeoutSec;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(connectTimeoutSec))
                .build();
    }

    /**
     * Embed a text string via Ollama /api/embed.
     * Returns double[] of size expectedDimensions.
     */
    public double[] embed(String text) throws Exception {
        String jsonBody = "{\"model\":\"" + escapeJson(model) + "\",\"input\":\"" + escapeJson(text) + "\"}";

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/embed"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .timeout(Duration.ofSeconds(requestTimeoutSec))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("Ollama embed failed: HTTP " + response.statusCode() + " — " + response.body());
        }

        return parseEmbeddingResponse(response.body());
    }

    /**
     * Format double[] as pgvector literal: "[0.1,0.2,...]"
     */
    public static String toVectorLiteral(double[] vec) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < vec.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(vec[i]);
        }
        sb.append(']');
        return sb.toString();
    }

    /**
     * Parse Ollama /api/embed response.
     * Format: {"embeddings":[[0.1,0.2,...]]} or {"embedding":[0.1,0.2,...]}
     */
    private double[] parseEmbeddingResponse(String json) {
        // Find the inner array of numbers
        // Look for "embeddings":[[  or "embedding":[
        int start = -1;
        int embeddingsIdx = json.indexOf("\"embeddings\"");
        int embeddingIdx = json.indexOf("\"embedding\"");

        if (embeddingsIdx >= 0) {
            // "embeddings":[[ — find the inner array start
            start = json.indexOf('[', embeddingsIdx);
            if (start >= 0) {
                start = json.indexOf('[', start + 1); // skip outer [
            }
        } else if (embeddingIdx >= 0) {
            // "embedding":[ — single array
            start = json.indexOf('[', embeddingIdx);
        }

        if (start < 0) {
            throw new RuntimeException("Cannot parse Ollama response: no embedding array found");
        }

        int end = json.indexOf(']', start);
        if (end < 0) {
            throw new RuntimeException("Cannot parse Ollama response: unclosed array");
        }

        String numbersStr = json.substring(start + 1, end);
        String[] parts = numbersStr.split(",");

        double[] result = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Double.parseDouble(parts[i].trim());
        }

        if (result.length != expectedDimensions) {
            log.warn("Ollama returned {} dimensions, expected {}", result.length, expectedDimensions);
        }

        return result;
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
