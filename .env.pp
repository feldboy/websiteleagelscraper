# Pre-Production Environment Configuration
ENV=pp
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql+asyncpg://legalpp:pppass@pp-db:5432/legaldb_pp

# LLM Configuration
OPENAI_API_KEY=sk-your-openai-api-key-pp
ANTHROPIC_API_KEY=sk-your-anthropic-api-key-pp
LLM_PROVIDER=openai
MAX_TOKENS=800
TEMPERATURE=0.7

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your-telegram-bot-token-pp
TELEGRAM_CHANNEL_ID=your-pp-channel-id

# Scraping Configuration
USER_AGENT_ROTATION=true
RESPECT_ROBOTS_TXT=true
RATE_LIMIT_DELAY=2.0
MAX_RETRIES=3
REQUEST_TIMEOUT=30

# Quality Assurance
MIN_WORD_COUNT=500
MAX_WORD_COUNT=600
MIN_ORIGINALITY_SCORE=0.95

# Monitoring and Logging
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Redis Configuration
REDIS_URL=redis://pp-redis:6379

# Scheduling Configuration
SCRAPING_INTERVAL_HOURS=4