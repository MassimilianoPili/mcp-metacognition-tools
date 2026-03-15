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

    // GP hyperparameters
    private double gpLengthScale = 1.0;
    private double gpNoiseVariance = 0.1;
    private int coldStartThreshold = 30;
    private int maxTrainingPoints = 500;

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

    public String getDefaultAgents() { return defaultAgents; }
    public void setDefaultAgents(String defaultAgents) { this.defaultAgents = defaultAgents; }
}
