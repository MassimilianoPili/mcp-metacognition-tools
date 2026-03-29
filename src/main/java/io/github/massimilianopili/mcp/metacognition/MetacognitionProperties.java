package io.github.massimilianopili.mcp.metacognition;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "mcp.metacognition")
public class MetacognitionProperties {

    private boolean enabled = false;

    // PostgreSQL
    private String dbUrl = "jdbc:postgresql://postgres:5432/embeddings";
    private String dbUsername = "postgres";
    private String dbCredential;

    // Ollama
    private String ollamaBaseUrl = "http://ollama:11434";
    private String ollamaModel = "mxbai-embed-large";
    private int embeddingDimensions = 1024;

    // Redis (optional GP cache)
    private String redisHost = "redis";
    private int redisPort = 6379;
    private int redisDb = 7;
    private int redisCacheTtlSeconds = 300;

    // HikariCP pool
    private int dbPoolSize = 30;
    private int dbMinimumIdle = 0;
    private long dbConnectionTimeout = 10_000;
    private long dbLeakDetectionThreshold = 60_000;

    // Redis pool
    private int redisPoolMaxTotal = 2;
    private int redisPoolMaxIdle = 1;
    private int redisTimeoutMs = 2000;

    // GP hyperparameters
    private double gpLengthScale = 1.0;
    private double gpNoiseVariance = 0.1;
    private int coldStartThreshold = 30;
    private int maxTrainingPoints = 500;

    // GP defaults
    private double gpDefaultMu = 0.5;
    private double gpDefaultSigma2 = 1.0;
    private double gpMinSigma2 = 1e-10;

    // Confidence thresholds
    private double confidenceHighThreshold = 0.7;
    private double confidenceMediumThreshold = 0.4;

    // Surprise
    private int surpriseHistoryDays = 7;

    // Ollama timeouts
    private int ollamaConnectTimeoutSec = 10;
    private int ollamaRequestTimeoutSec = 30;

    // Chunking — conversation
    private int conversationChunkMaxChars = 2000;
    private int conversationChunkOverlapChars = 200;

    // Chunking — markdown
    private int markdownChunkMaxChars = 2000;
    private int markdownChunkOverlapChars = 200;

    // Agent types (comma-separated default list)
    private String defaultAgents = "general-purpose,Explore,Plan,academic-researcher,feature-dev:code-explorer,feature-dev:code-architect";

    // Getters and setters

    public boolean isEnabled() { return enabled; }
    public void setEnabled(boolean enabled) { this.enabled = enabled; }

    public String getDbUrl() { return dbUrl; }
    public void setDbUrl(String dbUrl) { this.dbUrl = dbUrl; }

    public String getDbUsername() { return dbUsername; }
    public void setDbUsername(String dbUsername) { this.dbUsername = dbUsername; }

    public String getDbCredential() { return dbCredential; }
    public void setDbCredential(String dbCredential) { this.dbCredential = dbCredential; }

    public int getDbPoolSize() { return dbPoolSize; }
    public void setDbPoolSize(int dbPoolSize) { this.dbPoolSize = dbPoolSize; }

    public int getDbMinimumIdle() { return dbMinimumIdle; }
    public void setDbMinimumIdle(int dbMinimumIdle) { this.dbMinimumIdle = dbMinimumIdle; }

    public long getDbConnectionTimeout() { return dbConnectionTimeout; }
    public void setDbConnectionTimeout(long dbConnectionTimeout) { this.dbConnectionTimeout = dbConnectionTimeout; }

    public long getDbLeakDetectionThreshold() { return dbLeakDetectionThreshold; }
    public void setDbLeakDetectionThreshold(long dbLeakDetectionThreshold) { this.dbLeakDetectionThreshold = dbLeakDetectionThreshold; }

    public int getRedisPoolMaxTotal() { return redisPoolMaxTotal; }
    public void setRedisPoolMaxTotal(int redisPoolMaxTotal) { this.redisPoolMaxTotal = redisPoolMaxTotal; }

    public int getRedisPoolMaxIdle() { return redisPoolMaxIdle; }
    public void setRedisPoolMaxIdle(int redisPoolMaxIdle) { this.redisPoolMaxIdle = redisPoolMaxIdle; }

    public int getRedisTimeoutMs() { return redisTimeoutMs; }
    public void setRedisTimeoutMs(int redisTimeoutMs) { this.redisTimeoutMs = redisTimeoutMs; }

    public String getOllamaBaseUrl() { return ollamaBaseUrl; }
    public void setOllamaBaseUrl(String ollamaBaseUrl) { this.ollamaBaseUrl = ollamaBaseUrl; }

    public String getOllamaModel() { return ollamaModel; }
    public void setOllamaModel(String ollamaModel) { this.ollamaModel = ollamaModel; }

    public int getEmbeddingDimensions() { return embeddingDimensions; }
    public void setEmbeddingDimensions(int embeddingDimensions) { this.embeddingDimensions = embeddingDimensions; }

    public String getRedisHost() { return redisHost; }
    public void setRedisHost(String redisHost) { this.redisHost = redisHost; }

    public int getRedisPort() { return redisPort; }
    public void setRedisPort(int redisPort) { this.redisPort = redisPort; }

    public int getRedisDb() { return redisDb; }
    public void setRedisDb(int redisDb) { this.redisDb = redisDb; }

    public int getRedisCacheTtlSeconds() { return redisCacheTtlSeconds; }
    public void setRedisCacheTtlSeconds(int redisCacheTtlSeconds) { this.redisCacheTtlSeconds = redisCacheTtlSeconds; }

    public double getGpLengthScale() { return gpLengthScale; }
    public void setGpLengthScale(double gpLengthScale) { this.gpLengthScale = gpLengthScale; }

    public double getGpNoiseVariance() { return gpNoiseVariance; }
    public void setGpNoiseVariance(double gpNoiseVariance) { this.gpNoiseVariance = gpNoiseVariance; }

    public int getColdStartThreshold() { return coldStartThreshold; }
    public void setColdStartThreshold(int coldStartThreshold) { this.coldStartThreshold = coldStartThreshold; }

    public int getMaxTrainingPoints() { return maxTrainingPoints; }
    public void setMaxTrainingPoints(int maxTrainingPoints) { this.maxTrainingPoints = maxTrainingPoints; }

    public double getGpDefaultMu() { return gpDefaultMu; }
    public void setGpDefaultMu(double gpDefaultMu) { this.gpDefaultMu = gpDefaultMu; }

    public double getGpDefaultSigma2() { return gpDefaultSigma2; }
    public void setGpDefaultSigma2(double gpDefaultSigma2) { this.gpDefaultSigma2 = gpDefaultSigma2; }

    public double getGpMinSigma2() { return gpMinSigma2; }
    public void setGpMinSigma2(double gpMinSigma2) { this.gpMinSigma2 = gpMinSigma2; }

    public double getConfidenceHighThreshold() { return confidenceHighThreshold; }
    public void setConfidenceHighThreshold(double confidenceHighThreshold) { this.confidenceHighThreshold = confidenceHighThreshold; }

    public double getConfidenceMediumThreshold() { return confidenceMediumThreshold; }
    public void setConfidenceMediumThreshold(double confidenceMediumThreshold) { this.confidenceMediumThreshold = confidenceMediumThreshold; }

    public int getSurpriseHistoryDays() { return surpriseHistoryDays; }
    public void setSurpriseHistoryDays(int surpriseHistoryDays) { this.surpriseHistoryDays = surpriseHistoryDays; }

    public int getOllamaConnectTimeoutSec() { return ollamaConnectTimeoutSec; }
    public void setOllamaConnectTimeoutSec(int ollamaConnectTimeoutSec) { this.ollamaConnectTimeoutSec = ollamaConnectTimeoutSec; }

    public int getOllamaRequestTimeoutSec() { return ollamaRequestTimeoutSec; }
    public void setOllamaRequestTimeoutSec(int ollamaRequestTimeoutSec) { this.ollamaRequestTimeoutSec = ollamaRequestTimeoutSec; }

    public int getConversationChunkMaxChars() { return conversationChunkMaxChars; }
    public void setConversationChunkMaxChars(int conversationChunkMaxChars) { this.conversationChunkMaxChars = conversationChunkMaxChars; }

    public int getConversationChunkOverlapChars() { return conversationChunkOverlapChars; }
    public void setConversationChunkOverlapChars(int conversationChunkOverlapChars) { this.conversationChunkOverlapChars = conversationChunkOverlapChars; }

    public int getMarkdownChunkMaxChars() { return markdownChunkMaxChars; }
    public void setMarkdownChunkMaxChars(int markdownChunkMaxChars) { this.markdownChunkMaxChars = markdownChunkMaxChars; }

    public int getMarkdownChunkOverlapChars() { return markdownChunkOverlapChars; }
    public void setMarkdownChunkOverlapChars(int markdownChunkOverlapChars) { this.markdownChunkOverlapChars = markdownChunkOverlapChars; }

    public String getDefaultAgents() { return defaultAgents; }
    public void setDefaultAgents(String defaultAgents) { this.defaultAgents = defaultAgents; }
}
