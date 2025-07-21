# Multi-Agent Legal Research & Content Generation System

A comprehensive 11-agent system that continuously monitors legal news sources, extracts structured data, and generates high-quality 500-600 word legal blog articles with automated distribution capabilities.

## 🎯 System Overview

This autonomous system transforms raw legal news into professional blog content through coordinated multi-agent workflows with comprehensive monitoring and quality assurance.

### Core Principles
- **Defensive Security Only**: System designed for legitimate legal research and content generation
- **Compliance First**: Respects robots.txt, rate limits, and fair use guidelines
- **Modular Architecture**: Each agent is self-contained with clear interfaces
- **Data Integrity**: Transactional database operations with validation

## 🏗️ Architecture

### 11 Specialized Agents

1. **Web Scraper Agent** (`agents/web_scraper.py`) - Compliant legal news scraping
2. **Verification Agent** (`agents/verification.py`) - Content quality validation
3. **Database Agent** (`agents/database.py`) - Persistent storage with integrity
4. **Extractor Agent** (`agents/extractor.py`) - Legal entity and data extraction
5. **Summarizer Agent** (`agents/summarizer.py`) - Professional article summarization
6. **Connector Agent** (`agents/connector.py`) - Cross-article relationship analysis
7. **Writer Agent** (`agents/writer.py`) - 500-600 word article generation
8. **Quality Assurance Agent** (`agents/quality_assurance.py`) - Content validation
9. **Supervisor Agent** (placeholder) - System monitoring and health checks
10. **Telegram Bot Agent** (placeholder) - Content distribution
11. **Scheduler Agent** (placeholder) - 4-hour automated cycles

### Core Components

- **Configuration System** (`config/`) - Environment settings, source definitions, LLM prompts
- **Data Models** (`models/`) - Pydantic models for all data structures
- **Tools** (`tools/`) - Shared utilities for scraping, LLM communication
- **Main Orchestrator** (`main.py`) - Workflow coordination

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL database
- LLM API access (OpenAI or Anthropic)
- Redis (for task queuing)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd websiteleagelscraper
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys and database settings
```

3. **Setup database:**
```bash
# Create PostgreSQL database
createdb legaldb

# Database tables will be created automatically on first run
```

### Configuration

Edit `.env` with your settings:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/legaldb

# LLM Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
# OR
ANTHROPIC_API_KEY=sk-your-anthropic-api-key-here
LLM_PROVIDER=openai

# Optional: Telegram Configuration
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHANNEL_ID=your-channel-id

# Scraping Configuration
RESPECT_ROBOTS_TXT=true
RATE_LIMIT_DELAY=2.0
```

## 🔧 Usage

### Run Complete Workflow

```bash
python main.py
```

This runs the full pipeline:
1. Scrapes legal news sources
2. Extracts structured data
3. Generates summaries
4. Analyzes connections and trends
5. Writes comprehensive article
6. Validates quality
7. Prepares for distribution

### Run Scraping Only

```bash
python main.py scraping-only
```

### Monitor Logs

```bash
tail -f logs/legal_research_system.log
```

## 🔍 System Features

### Web Scraping
- **Compliant Scraping**: Respects robots.txt and implements rate limiting
- **Source Management**: Configurable legal news sources
- **Error Handling**: Exponential backoff and retry logic
- **User Agent Rotation**: Prevents blocking

### Data Processing
- **Entity Extraction**: Legal entities, cases, dates, monetary amounts
- **Summarization**: Professional 100-150 word summaries
- **Connection Analysis**: Identifies relationships between articles
- **Trend Detection**: Recognizes emerging legal trends

### Content Generation
- **Structured Writing**: 500-600 word articles with proper sections
- **Quality Validation**: Originality, readability, legal precision
- **Professional Tone**: Suitable for legal professionals
- **Source Attribution**: Transparent content sourcing

### Quality Assurance
- **Plagiarism Detection**: Multi-layered originality checking
- **Content Validation**: Legal terminology, structure, appropriateness
- **LLM Enhancement**: AI-powered quality assessment
- **Standards Enforcement**: Configurable quality thresholds

## 📊 Data Models

### Key Data Structures

- **ScrapedData**: Raw article content with metadata
- **ExtractedData**: Structured legal information
- **ArticleSummary**: Professional summaries with legal significance
- **GeneratedArticle**: Final blog articles with quality metrics
- **ValidationResult**: Comprehensive quality assessment

### Database Schema

- **articles**: Scraped content with duplicate detection
- **extractions**: Structured legal data
- **quality_assessments**: Validation results

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v --cov=agents --cov=tools --cov-report=term-missing
```

### Code Quality

```bash
# Formatting
black . --check
isort . --check-only

# Linting
flake8 . --max-line-length=88

# Type checking
mypy . --strict
```

### Adding New Sources

1. Edit `config/sources.py`
2. Add source configuration with selectors
3. Test scraping compliance

## 📈 Monitoring

### System Health Checks

The system provides comprehensive monitoring:

- **Scraping Statistics**: Success rates, response times
- **Processing Metrics**: Extraction quality, summary scores
- **Quality Trends**: Validation pass rates, common issues
- **Database Health**: Article counts, processing status

### Performance Metrics

- **Throughput**: Articles processed per hour
- **Quality Scores**: Average validation scores
- **Error Rates**: Failed operations by type
- **Resource Usage**: LLM token consumption, costs

## 🔒 Security & Compliance

### Defensive Design
- **No Malicious Use**: System refuses harmful requests
- **Rate Limiting**: Prevents server overload
- **Robots.txt Compliance**: Respects website policies
- **Data Privacy**: No personal information storage

### Best Practices
- **API Key Security**: Environment variable storage
- **Database Security**: Prepared statements, input validation
- **Content Validation**: Prevents inappropriate content
- **Error Handling**: Graceful failure management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**Database Connection Errors:**
- Verify PostgreSQL is running
- Check DATABASE_URL in .env
- Ensure database exists

**LLM API Errors:**
- Verify API keys are correct
- Check rate limits and quotas
- Ensure provider setting matches available key

**Scraping Failures:**
- Check internet connectivity
- Verify source websites are accessible
- Review robots.txt compliance

### Getting Help

1. Check the logs in `logs/legal_research_system.log`
2. Review configuration in `.env`
3. Run with debug logging: `LOG_LEVEL=DEBUG python main.py`
4. Open an issue with system information and error logs

## 🎯 Roadmap

### Planned Features

- [ ] Telegram bot integration
- [ ] Automated scheduling system
- [ ] Enhanced trend analysis
- [ ] Multiple output formats
- [ ] API endpoints for integration
- [ ] Real-time monitoring dashboard
- [ ] Machine learning quality improvements

### Performance Optimizations

- [ ] Parallel article processing
- [ ] Caching layer for LLM responses
- [ ] Database query optimization
- [ ] Memory usage improvements

## 📚 Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [SQLAlchemy Async Guide](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Beautiful Soup Documentation](https://beautiful-soup-4.readthedocs.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Documentation](https://docs.anthropic.com/)

---

*Built with ❤️ for the legal community*