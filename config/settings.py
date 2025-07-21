"""
Environment configuration and settings for the legal research system.
Uses pydantic-settings for environment validation and type safety.
"""
from pydantic import BaseSettings, Field, validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable validation."""
    
    # Database Configuration
    database_url: str = Field(
        ...,
        env="DATABASE_URL",
        description="PostgreSQL database connection URL"
    )
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(
        None,
        env="OPENAI_API_KEY",
        description="OpenAI API key for content generation"
    )
    anthropic_api_key: Optional[str] = Field(
        None,
        env="ANTHROPIC_API_KEY", 
        description="Anthropic API key for content generation"
    )
    llm_provider: str = Field(
        "openai",
        env="LLM_PROVIDER",
        description="LLM provider to use (openai, anthropic)"
    )
    max_tokens: int = Field(
        800,
        env="MAX_TOKENS",
        description="Maximum tokens for LLM responses"
    )
    temperature: float = Field(
        0.7,
        env="TEMPERATURE",
        description="Temperature setting for LLM responses"
    )
    
    # Telegram Configuration
    telegram_bot_token: str = Field(
        ...,
        env="TELEGRAM_BOT_TOKEN",
        description="Telegram bot token for content distribution"
    )
    telegram_channel_id: str = Field(
        ...,
        env="TELEGRAM_CHANNEL_ID",
        description="Telegram channel ID for posting articles"
    )
    
    # Scraping Configuration
    user_agent_rotation: bool = Field(
        True,
        env="USER_AGENT_ROTATION",
        description="Enable user agent rotation for scraping"
    )
    respect_robots_txt: bool = Field(
        True,
        env="RESPECT_ROBOTS_TXT",
        description="Respect robots.txt files when scraping"
    )
    rate_limit_delay: float = Field(
        2.0,
        env="RATE_LIMIT_DELAY",
        description="Delay in seconds between requests to same domain"
    )
    max_retries: int = Field(
        3,
        env="MAX_RETRIES",
        description="Maximum retry attempts for failed requests"
    )
    request_timeout: int = Field(
        30,
        env="REQUEST_TIMEOUT",
        description="HTTP request timeout in seconds"
    )
    
    # Quality Assurance
    min_word_count: int = Field(
        500,
        env="MIN_WORD_COUNT",
        description="Minimum word count for generated articles"
    )
    max_word_count: int = Field(
        600,
        env="MAX_WORD_COUNT", 
        description="Maximum word count for generated articles"
    )
    min_originality_score: float = Field(
        0.95,
        env="MIN_ORIGINALITY_SCORE",
        description="Minimum originality score for article approval"
    )
    
    # Monitoring and Logging
    log_level: str = Field(
        "INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    metrics_enabled: bool = Field(
        True,
        env="METRICS_ENABLED",
        description="Enable performance metrics collection"
    )
    
    # Redis Configuration (for queuing)
    redis_url: str = Field(
        "redis://localhost:6379",
        env="REDIS_URL",
        description="Redis connection URL for task queuing"
    )
    
    # Scheduling Configuration
    scraping_interval_hours: int = Field(
        4,
        env="SCRAPING_INTERVAL_HOURS",
        description="Hours between scraping cycles"
    )
    
    @validator("llm_provider")
    def validate_llm_provider(cls, v):
        """Validate LLM provider choice."""
        allowed_providers = ["openai", "anthropic"]
        if v not in allowed_providers:
            raise ValueError(f"LLM provider must be one of: {allowed_providers}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level choice."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    def validate_api_keys(self):
        """Validate that required API keys are present based on provider."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required when using Anthropic provider")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Validate API keys on import
settings.validate_api_keys()