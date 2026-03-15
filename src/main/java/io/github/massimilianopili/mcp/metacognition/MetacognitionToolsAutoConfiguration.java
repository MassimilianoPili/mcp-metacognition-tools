package io.github.massimilianopili.mcp.metacognition;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Import;

@AutoConfiguration
@ConditionalOnProperty(name = "mcp.metacognition.enabled", havingValue = "true", matchIfMissing = false)
@Import({MetacognitionConfig.class, MetacognitionTools.class})
public class MetacognitionToolsAutoConfiguration {
}
