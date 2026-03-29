package io.github.massimilianopili.mcp.metacognition;

import io.github.massimilianopili.ai.reactive.annotation.ReactiveTool;
import org.springframework.ai.tool.annotation.ToolParam;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import javax.sql.DataSource;
import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;

@Service
public class MetacognitionTools {

    private static final Logger log = LoggerFactory.getLogger(MetacognitionTools.class);

    private final JdbcTemplate jdbc;
    private final MetacognitionProperties props;
    private final OllamaEmbeddingClient embedder;
    private volatile Object jedisPool; // redis.clients.jedis.JedisPool, nullable

    public MetacognitionTools(
            @Qualifier("metacognitionDataSource") DataSource dataSource,
            MetacognitionProperties props,
            @Qualifier("ollamaEmbeddingClient") OllamaEmbeddingClient embedder) {
        this.jdbc = new JdbcTemplate(dataSource);
        this.props = props;
        this.embedder = embedder;
        initRedis();
    }

    @SuppressWarnings("unchecked")
    private void initRedis() {
        try {
            Class<?> poolConfigClass = Class.forName("redis.clients.jedis.JedisPoolConfig");
            Class<?> poolClass = Class.forName("redis.clients.jedis.JedisPool");
            Object poolConfig = poolConfigClass.getDeclaredConstructor().newInstance();
            poolConfigClass.getMethod("setMaxTotal", int.class).invoke(poolConfig, props.getRedisPoolMaxTotal());
            poolConfigClass.getMethod("setMaxIdle", int.class).invoke(poolConfig, props.getRedisPoolMaxIdle());
            this.jedisPool = poolClass.getDeclaredConstructor(
                    Class.forName("org.apache.commons.pool2.impl.GenericObjectPoolConfig"),
                    String.class, int.class, int.class, String.class, int.class
            ).newInstance(poolConfig, props.getRedisHost(), props.getRedisPort(), props.getRedisTimeoutMs(), null, props.getRedisDb());
            log.info("Metacognition: Redis cache enabled ({}:{}/{})", props.getRedisHost(), props.getRedisPort(), props.getRedisDb());
        } catch (Exception e) {
            log.info("Metacognition: Redis unavailable, caching disabled: {}", e.getMessage());
            this.jedisPool = null;
        }
    }

    @PostConstruct
    void initSchema() {
        try {
            jdbc.execute("CREATE EXTENSION IF NOT EXISTS vector");
            jdbc.execute("""
                    CREATE TABLE IF NOT EXISTS metacognition_decisions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        session_id VARCHAR(100),
                        query TEXT NOT NULL,
                        query_embedding vector(1024),
                        recommended_agent VARCHAR(50),
                        actual_agent VARCHAR(50),
                        gp_mu DOUBLE PRECISION,
                        gp_sigma2 DOUBLE PRECISION,
                        outcome_quality DOUBLE PRECISION,
                        token_cost INTEGER,
                        wall_time_ms BIGINT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                    """);
            jdbc.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metacognition_agent
                    ON metacognition_decisions (actual_agent, created_at DESC)
                    """);
            jdbc.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metacognition_session
                    ON metacognition_decisions (session_id)
                    """);
            log.info("Metacognition tools: metacognition_decisions table initialized");
        } catch (Exception e) {
            log.warn("Metacognition tools: could not init schema: {}", e.getMessage());
        }
    }

    // ========================================================================
    // Tool 1: meta_predict_agent
    // ========================================================================

    @ReactiveTool(name = "meta_predict_agent",
            description = "Predict which Claude Code agent type will perform best for a given task. "
                    + "Uses Gaussian Process regression on historical task embeddings and outcomes. "
                    + "Returns ranked predictions with uncertainty (mu = expected quality 0-1, "
                    + "sigma2 = uncertainty). Lower sigma2 = higher confidence. "
                    + "Cold start mode (< 30 training points) uses keyword-based fallback.")
    public Mono<Map<String, Object>> predictAgent(
            @ToolParam(description = "Task description or query to predict agent for") String query,
            @ToolParam(description = "Comma-separated candidate agent types to evaluate", required = false) String candidateAgents) {

        return Mono.fromCallable(() -> {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("query", query);

            String[] agents = candidateAgents != null && !candidateAgents.isBlank()
                    ? candidateAgents.split(",")
                    : props.getDefaultAgents().split(",");
            for (int i = 0; i < agents.length; i++) agents[i] = agents[i].trim();

            Integer totalPoints = jdbc.queryForObject(
                    "SELECT COUNT(*) FROM metacognition_decisions WHERE outcome_quality IS NOT NULL",
                    Integer.class);
            int tp = totalPoints != null ? totalPoints : 0;
            boolean coldStart = tp < props.getColdStartThreshold();

            result.put("coldStart", coldStart);
            result.put("totalTrainingPoints", tp);

            if (coldStart) {
                List<Map<String, Object>> predictions = keywordFallback(query, agents);
                result.put("predictions", predictions);
                result.put("recommended", predictions.get(0).get("agentType"));
                result.put("confidence", "low");
                return result;
            }

            // Embed query
            double[] queryEmb;
            try {
                queryEmb = embedder.embed(query);
            } catch (Exception e) {
                log.warn("Ollama embedding failed, falling back to keywords: {}", e.getMessage());
                List<Map<String, Object>> predictions = keywordFallback(query, agents);
                result.put("predictions", predictions);
                result.put("recommended", predictions.get(0).get("agentType"));
                result.put("confidence", "low");
                result.put("embeddingError", e.getMessage());
                return result;
            }

            // Check Redis cache
            String cacheKey = cacheKey(queryEmb, agents);
            String cached = redisGet(cacheKey);
            if (cached != null) {
                result.put("source", "cache");
                result.put("predictions", parseCachedPredictions(cached));
                result.put("recommended", parseCachedRecommended(cached));
                result.put("confidence", parseCachedConfidence(cached));
                return result;
            }

            // GP prediction per agent type
            GaussianProcessEngine gp = new GaussianProcessEngine(
                    props.getGpLengthScale(), props.getGpNoiseVariance(), props.getGpMinSigma2());

            List<Map<String, Object>> predictions = new ArrayList<>();
            String bestAgent = null;
            double bestMu = -1.0;

            for (String agent : agents) {
                TrainingData td = loadTrainingData(agent);
                double[] posterior;
                if (td.embeddings.length == 0) {
                    posterior = new double[]{props.getGpDefaultMu(), props.getGpDefaultSigma2()};
                } else {
                    posterior = gp.predict(td.embeddings, td.outcomes, queryEmb);
                }
                double mu = posterior[0];
                double sigma2 = posterior[1];

                Map<String, Object> pred = new LinkedHashMap<>();
                pred.put("agentType", agent);
                pred.put("mu", Math.round(mu * 1000.0) / 1000.0);
                pred.put("sigma2", Math.round(sigma2 * 10000.0) / 10000.0);
                pred.put("trainingPoints", td.embeddings.length);
                predictions.add(pred);

                if (mu > bestMu) {
                    bestMu = mu;
                    bestAgent = agent;
                }
            }

            predictions.sort((a, b) -> Double.compare((double) b.get("mu"), (double) a.get("mu")));

            String confidence = bestMu > props.getConfidenceHighThreshold() ? "high"
                    : bestMu > props.getConfidenceMediumThreshold() ? "medium" : "low";
            result.put("recommended", bestAgent);
            result.put("predictions", predictions);
            result.put("confidence", confidence);
            result.put("source", "gp");

            // Cache result
            redisPut(cacheKey, serializePredictions(bestAgent, confidence, predictions));

            return result;
        });
    }

    // ========================================================================
    // Tool 2: meta_record_outcome
    // ========================================================================

    @ReactiveTool(name = "meta_record_outcome",
            description = "Record the outcome of an agent execution to train the GP model. "
                    + "Call this after an agent completes a task to improve future predictions. "
                    + "outcomeQuality is 0.0 (complete failure) to 1.0 (perfect execution).")
    public Mono<Map<String, Object>> recordOutcome(
            @ToolParam(description = "Session ID for this execution") String sessionId,
            @ToolParam(description = "The original task query") String query,
            @ToolParam(description = "Agent type that was used (e.g. 'academic-researcher', 'Explore')") String agentType,
            @ToolParam(description = "Quality of the outcome, 0.0 to 1.0") double outcomeQuality,
            @ToolParam(description = "Token cost of the execution", required = false) Integer tokenCost,
            @ToolParam(description = "Wall clock time in milliseconds", required = false) Long wallTimeMs) {

        return Mono.fromCallable(() -> {
            Map<String, Object> result = new LinkedHashMap<>();
            try {
                double[] embedding = embedder.embed(query);
                String vectorLiteral = OllamaEmbeddingClient.toVectorLiteral(embedding);

                jdbc.update("""
                                INSERT INTO metacognition_decisions
                                (session_id, query, query_embedding, actual_agent, outcome_quality, token_cost, wall_time_ms)
                                VALUES (?, ?, ?::vector, ?, ?, ?, ?)
                                """,
                        sessionId, query, vectorLiteral, agentType,
                        outcomeQuality, tokenCost, wallTimeMs);

                // Invalidate cache for this agent
                redisInvalidate();

                Integer totalPoints = jdbc.queryForObject(
                        "SELECT COUNT(*) FROM metacognition_decisions WHERE outcome_quality IS NOT NULL",
                        Integer.class);

                result.put("recorded", true);
                result.put("agentType", agentType);
                result.put("outcomeQuality", outcomeQuality);
                result.put("totalTrainingPoints", totalPoints != null ? totalPoints : 0);
            } catch (Exception e) {
                log.warn("meta_record_outcome failed: {}", e.getMessage());
                result.put("recorded", false);
                result.put("error", e.getMessage());
            }
            return result;
        });
    }

    // ========================================================================
    // Tool 3: meta_surprise
    // ========================================================================

    @ReactiveTool(name = "meta_surprise",
            description = "Compute Bayesian surprise for the current session. "
                    + "Measures how different this session's tool outcomes are from the historical "
                    + "distribution using KL divergence. High surprise means the session is behaving "
                    + "very differently from normal (either much better or much worse). "
                    + "Reads from tool_outcomes table (populated by tool-outcome-tracker hook).")
    public Mono<Map<String, Object>> surprise(
            @ToolParam(description = "Session ID to analyze") String sessionId,
            @ToolParam(description = "Filter to a specific tool name", required = false) String toolName) {

        return Mono.fromCallable(() -> {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("sessionId", sessionId);

            try {
                // Session stats
                String sessionSql = "SELECT COUNT(*) AS total, "
                        + "COUNT(*) FILTER (WHERE success) AS successes "
                        + "FROM tool_outcomes WHERE session_id = ?";
                Object[] sessionParams;
                if (toolName != null && !toolName.isBlank()) {
                    sessionSql += " AND tool_name = ?";
                    sessionParams = new Object[]{sessionId, toolName};
                    result.put("toolName", toolName);
                } else {
                    sessionParams = new Object[]{sessionId};
                }

                Map<String, Object> sessionStats = jdbc.queryForMap(sessionSql, sessionParams);

                // Historical stats (7 days, excluding current session)
                String histSql = "SELECT COUNT(*) AS total, "
                        + "COUNT(*) FILTER (WHERE success) AS successes "
                        + "FROM tool_outcomes WHERE created_at >= NOW() - INTERVAL '" + props.getSurpriseHistoryDays() + " days' "
                        + "AND session_id != ?";
                Object[] histParams;
                if (toolName != null && !toolName.isBlank()) {
                    histSql += " AND tool_name = ?";
                    histParams = new Object[]{sessionId, toolName};
                } else {
                    histParams = new Object[]{sessionId};
                }

                Map<String, Object> histStats = jdbc.queryForMap(histSql, histParams);

                long sessionTotal = ((Number) sessionStats.get("total")).longValue();
                long sessionSuccess = ((Number) sessionStats.get("successes")).longValue();
                long histTotal = ((Number) histStats.get("total")).longValue();
                long histSuccess = ((Number) histStats.get("successes")).longValue();

                if (sessionTotal == 0) {
                    result.put("surprise", 0.0);
                    result.put("interpretation", "No tool outcomes recorded for this session yet");
                    return result;
                }

                if (histTotal == 0) {
                    result.put("surprise", 0.0);
                    result.put("interpretation", "No historical data available for comparison");
                    result.put("sessionStats", Map.of(
                            "total", sessionTotal,
                            "successes", sessionSuccess,
                            "successRate", round3((double) sessionSuccess / sessionTotal)));
                    return result;
                }

                // KL divergence D_KL(posterior || prior) for Bernoulli
                double p = (double) sessionSuccess / sessionTotal;
                double q = (double) histSuccess / histTotal;

                // Smoothing to avoid log(0)
                p = Math.max(0.001, Math.min(0.999, p));
                q = Math.max(0.001, Math.min(0.999, q));

                double klDiv = p * Math.log(p / q) + (1.0 - p) * Math.log((1.0 - p) / (1.0 - q));

                String interpretation;
                if (klDiv < 0.05) interpretation = "Normal — session behaves as expected";
                else if (klDiv < 0.2) interpretation = "Mild surprise — slight deviation from historical pattern";
                else if (klDiv < 0.5) interpretation = "Moderate surprise — notable deviation, worth investigating";
                else interpretation = "High surprise — session behavior very different from normal";

                double rawP = (double) sessionSuccess / sessionTotal;
                double rawQ = (double) histSuccess / histTotal;
                if (rawP > rawQ) interpretation += " (session performing BETTER than average)";
                else if (rawP < rawQ) interpretation += " (session performing WORSE than average)";

                result.put("surprise", round4(klDiv));
                result.put("interpretation", interpretation);
                result.put("sessionStats", Map.of(
                        "total", sessionTotal,
                        "successes", sessionSuccess,
                        "successRate", round3(rawP)));
                result.put("historicalStats", Map.of(
                        "total", histTotal,
                        "successes", histSuccess,
                        "successRate", round3(rawQ)));

            } catch (Exception e) {
                log.warn("meta_surprise failed: {}", e.getMessage());
                result.put("error", e.getMessage());
                result.put("surprise", 0.0);
            }
            return result;
        });
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    private record TrainingData(double[][] embeddings, double[] outcomes) {}

    private TrainingData loadTrainingData(String agentType) {
        int maxPoints = props.getMaxTrainingPoints();
        List<Map<String, Object>> rows = jdbc.queryForList("""
                SELECT query_embedding::text AS emb, outcome_quality
                FROM metacognition_decisions
                WHERE actual_agent = ? AND outcome_quality IS NOT NULL AND query_embedding IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
                """, agentType, maxPoints);

        double[][] embeddings = new double[rows.size()][];
        double[] outcomes = new double[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            embeddings[i] = parseVectorLiteral((String) rows.get(i).get("emb"));
            outcomes[i] = ((Number) rows.get(i).get("outcome_quality")).doubleValue();
        }
        return new TrainingData(embeddings, outcomes);
    }

    private List<Map<String, Object>> keywordFallback(String query, String[] agents) {
        String q = query.toLowerCase();
        Map<String, Double> scores = new LinkedHashMap<>();
        for (String agent : agents) scores.put(agent, 0.3);

        if (q.contains("plan") || q.contains("design") || q.contains("architect"))
            scores.merge("Plan", 0.4, Double::sum);
        if (q.contains("implement") || q.contains("code") || q.contains("write") || q.contains("create") || q.contains("build"))
            scores.merge("general-purpose", 0.4, Double::sum);
        if (q.contains("research") || q.contains("find") || q.contains("search") || q.contains("paper") || q.contains("literature"))
            scores.merge("academic-researcher", 0.4, Double::sum);
        if (q.contains("explore") || q.contains("codebase") || q.contains("understand") || q.contains("find file"))
            scores.merge("Explore", 0.4, Double::sum);
        if (q.contains("review") || q.contains("check") || q.contains("audit") || q.contains("quality"))
            scores.merge("feature-dev:code-explorer", 0.3, Double::sum);
        if (q.contains("architect") || q.contains("design") || q.contains("structure") || q.contains("pattern"))
            scores.merge("feature-dev:code-architect", 0.3, Double::sum);

        return scores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .map(e -> {
                    Map<String, Object> m = new LinkedHashMap<>();
                    m.put("agentType", e.getKey());
                    m.put("mu", round3(e.getValue()));
                    m.put("sigma2", 1.0);
                    m.put("trainingPoints", 0);
                    return m;
                })
                .toList();
    }

    /** Parse pgvector text representation "[0.1,0.2,...]" into double[] */
    private static double[] parseVectorLiteral(String s) {
        if (s == null || s.isBlank()) return new double[0];
        String inner = s.trim();
        if (inner.startsWith("[")) inner = inner.substring(1);
        if (inner.endsWith("]")) inner = inner.substring(0, inner.length() - 1);
        String[] parts = inner.split(",");
        double[] result = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Double.parseDouble(parts[i].trim());
        }
        return result;
    }

    private static double round3(double v) { return Math.round(v * 1000.0) / 1000.0; }
    private static double round4(double v) { return Math.round(v * 10000.0) / 10000.0; }

    // ========================================================================
    // Redis cache (graceful degradation via reflection)
    // ========================================================================

    private String redisGet(String key) {
        if (jedisPool == null) return null;
        try {
            Object jedis = jedisPool.getClass().getMethod("getResource").invoke(jedisPool);
            try {
                String val = (String) jedis.getClass().getMethod("get", String.class).invoke(jedis, key);
                return val;
            } finally {
                jedis.getClass().getMethod("close").invoke(jedis);
            }
        } catch (Exception e) {
            log.debug("Redis GET failed: {}", e.getMessage());
            return null;
        }
    }

    private void redisPut(String key, String value) {
        if (jedisPool == null) return;
        try {
            Object jedis = jedisPool.getClass().getMethod("getResource").invoke(jedisPool);
            try {
                jedis.getClass().getMethod("setex", String.class, long.class, String.class)
                        .invoke(jedis, key, (long) props.getRedisCacheTtlSeconds(), value);
            } finally {
                jedis.getClass().getMethod("close").invoke(jedis);
            }
        } catch (Exception e) {
            log.debug("Redis SET failed: {}", e.getMessage());
        }
    }

    private void redisInvalidate() {
        if (jedisPool == null) return;
        try {
            Object jedis = jedisPool.getClass().getMethod("getResource").invoke(jedisPool);
            try {
                @SuppressWarnings("unchecked")
                Set<String> keys = (Set<String>) jedis.getClass()
                        .getMethod("keys", String.class).invoke(jedis, "metacog:*");
                if (keys != null && !keys.isEmpty()) {
                    jedis.getClass().getMethod("del", String[].class)
                            .invoke(jedis, (Object) keys.toArray(new String[0]));
                }
            } finally {
                jedis.getClass().getMethod("close").invoke(jedis);
            }
        } catch (Exception e) {
            log.debug("Redis invalidation failed: {}", e.getMessage());
        }
    }

    private String cacheKey(double[] embedding, String[] agents) {
        long hash = 0;
        for (int i = 0; i < Math.min(16, embedding.length); i++) {
            hash = hash * 31 + Double.hashCode(embedding[i]);
        }
        return "metacog:" + hash + ":" + String.join(",", agents);
    }

    /** Minimal JSON serialization for cache (no Jackson) */
    private String serializePredictions(String recommended, String confidence, List<Map<String, Object>> predictions) {
        StringBuilder sb = new StringBuilder("{\"recommended\":\"").append(recommended)
                .append("\",\"confidence\":\"").append(confidence).append("\",\"predictions\":[");
        for (int i = 0; i < predictions.size(); i++) {
            if (i > 0) sb.append(',');
            Map<String, Object> p = predictions.get(i);
            sb.append("{\"agentType\":\"").append(p.get("agentType"))
                    .append("\",\"mu\":").append(p.get("mu"))
                    .append(",\"sigma2\":").append(p.get("sigma2"))
                    .append(",\"trainingPoints\":").append(p.get("trainingPoints")).append('}');
        }
        sb.append("]}");
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> parseCachedPredictions(String json) {
        // Simple extraction: find "predictions":[ and parse each {agentType, mu, sigma2, trainingPoints}
        List<Map<String, Object>> result = new ArrayList<>();
        int idx = json.indexOf("\"predictions\":[");
        if (idx < 0) return result;
        int start = json.indexOf('[', idx) + 1;
        int end = json.lastIndexOf(']');
        if (start >= end) return result;

        String inner = json.substring(start, end);
        // Split on },{
        String[] items = inner.split("\\},\\{");
        for (String item : items) {
            item = item.replace("{", "").replace("}", "");
            Map<String, Object> m = new LinkedHashMap<>();
            for (String kv : item.split(",(?=\")")) {
                String[] pair = kv.split(":", 2);
                if (pair.length == 2) {
                    String k = pair[0].replace("\"", "").trim();
                    String v = pair[1].replace("\"", "").trim();
                    switch (k) {
                        case "agentType" -> m.put(k, v);
                        case "mu", "sigma2" -> m.put(k, Double.parseDouble(v));
                        case "trainingPoints" -> m.put(k, Integer.parseInt(v));
                    }
                }
            }
            if (!m.isEmpty()) result.add(m);
        }
        return result;
    }

    private String parseCachedRecommended(String json) {
        int idx = json.indexOf("\"recommended\":\"");
        if (idx < 0) return "unknown";
        int start = idx + 15;
        int end = json.indexOf('"', start);
        return json.substring(start, end);
    }

    private String parseCachedConfidence(String json) {
        int idx = json.indexOf("\"confidence\":\"");
        if (idx < 0) return "low";
        int start = idx + 14;
        int end = json.indexOf('"', start);
        return json.substring(start, end);
    }
}
