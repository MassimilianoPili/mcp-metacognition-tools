package io.github.massimilianopili.mcp.metacognition;

import com.zaxxer.hikari.HikariDataSource;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableConfigurationProperties(MetacognitionProperties.class)
public class MetacognitionConfig {

    @Bean(name = "metacognitionDataSource")
    public HikariDataSource metacognitionDataSource(MetacognitionProperties props) {
        HikariDataSource ds = new HikariDataSource();
        ds.setJdbcUrl(props.getDbUrl());
        ds.setUsername(props.getDbUsername());
        if (props.getDbCredential() != null && !props.getDbCredential().isBlank()) {
            ds.setPassword(props.getDbCredential());
        }
        ds.setMaximumPoolSize(props.getDbPoolSize());
        ds.setMinimumIdle(props.getDbMinimumIdle());
        ds.setConnectionTimeout(props.getDbConnectionTimeout());
        ds.setLeakDetectionThreshold(props.getDbLeakDetectionThreshold());
        ds.setPoolName("metacognition-pool");
        return ds;
    }

    @Bean(name = "ollamaEmbeddingClient")
    public OllamaEmbeddingClient ollamaEmbeddingClient(MetacognitionProperties props) {
        return new OllamaEmbeddingClient(
                props.getOllamaBaseUrl(),
                props.getOllamaModel(),
                props.getEmbeddingDimensions(),
                props.getOllamaConnectTimeoutSec(),
                props.getOllamaRequestTimeoutSec()
        );
    }
}
