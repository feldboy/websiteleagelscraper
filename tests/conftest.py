"""
Pytest configuration and shared fixtures for the legal research system tests.
"""

import asyncio
import pytest
import tempfile
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from typing import Generator, Dict, Any

from config.settings import Settings
from agents.database import database_agent
from models.scraped_data import ScrapedData
from models.extracted_data import ExtractedData, LegalEntity, EntityType
from agents.summarizer import ArticleSummary
from agents.writer import GeneratedArticle


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        database_url="sqlite+aiosqlite:///test.db",
        llm_provider="openai",
        openai_api_key="test-key",
        telegram_bot_token="test-bot-token",
        telegram_channel_id="test-channel",
        respect_robots_txt=False,  # Disable for testing
        rate_limit_delay=0.1,
        max_retries=1,
    )


@pytest.fixture
async def mock_database():
    """Mock database for testing."""
    # Use in-memory SQLite for tests
    original_url = database_agent.engine

    # Initialize test database
    await database_agent.initialize()

    yield database_agent

    # Cleanup
    if database_agent.engine:
        await database_agent.close()


@pytest.fixture
def sample_scraped_data() -> ScrapedData:
    """Sample scraped data for testing."""
    return ScrapedData(
        url="https://example.com/legal-article",
        source_name="Test Legal News",
        title="Supreme Court Rules on Important Case",
        content="""
        The Supreme Court ruled today in a landmark decision that will affect 
        legal proceedings nationwide. The case, Smith v. Jones, involved complex 
        constitutional questions about due process and equal protection.
        
        Chief Justice Roberts wrote the majority opinion, stating that the lower 
        court's decision was inconsistent with established precedent. The ruling 
        clarifies important aspects of federal jurisdiction in civil rights cases.
        
        Legal experts expect this decision to have far-reaching implications for 
        similar cases pending in federal courts across the country. The American 
        Bar Association praised the clarity of the Court's reasoning.
        """,
        author="Legal Reporter",
        publish_date=datetime(2024, 1, 15),
        scraped_at=datetime.now(),
        status_code=200,
        headers={"Content-Type": "text/html"},
        user_agent="Test-Agent",
        response_time=1.5,
    )


@pytest.fixture
def sample_extracted_data() -> ExtractedData:
    """Sample extracted data for testing."""
    return ExtractedData(
        article_id="test-article-123",
        source_url="https://example.com/legal-article",
        entities=[
            LegalEntity(
                name="Chief Justice Roberts",
                entity_type=EntityType.JUDGE,
                relevance_score=0.9,
            ),
            LegalEntity(
                name="Supreme Court", entity_type=EntityType.COURT, relevance_score=0.95
            ),
        ],
        legal_topics=["constitutional_law", "civil_rights"],
        key_quotes=[
            "The lower court's decision was inconsistent with established precedent"
        ],
        extraction_confidence=0.85,
        completeness_score=0.8,
    )


@pytest.fixture
def sample_article_summary() -> ArticleSummary:
    """Sample article summary for testing."""
    return ArticleSummary(
        article_id="test-article-123",
        summary="The Supreme Court issued a landmark ruling in Smith v. Jones, clarifying important aspects of federal jurisdiction in civil rights cases. Chief Justice Roberts wrote the majority opinion, stating the lower court's decision was inconsistent with precedent.",
        key_points=[
            "Supreme Court ruled in Smith v. Jones",
            "Clarified federal jurisdiction in civil rights",
            "Chief Justice Roberts wrote majority opinion",
        ],
        legal_significance="This ruling provides important clarity on constitutional questions and will affect legal proceedings nationwide.",
        urgency_level="high",
        impact_scope="national",
    )


@pytest.fixture
def sample_generated_article() -> GeneratedArticle:
    """Sample generated article for testing."""
    return GeneratedArticle(
        title="Supreme Court Delivers Landmark Ruling on Civil Rights Jurisdiction",
        content="""
        ## Supreme Court Clarifies Federal Jurisdiction

        In a significant development for civil rights law, the Supreme Court today 
        issued a unanimous decision in Smith v. Jones that clarifies the scope of 
        federal court jurisdiction in constitutional cases. The ruling, authored by 
        Chief Justice Roberts, overturns a controversial lower court decision that 
        had created uncertainty in the legal community.

        ## Key Legal Implications

        The Court's decision establishes clear precedent for how federal courts 
        should handle cases involving both constitutional and state law claims. 
        Legal experts note that this clarification will streamline litigation 
        processes and provide greater certainty for both plaintiffs and defendants 
        in civil rights cases.

        ## Impact on Future Cases

        This ruling is expected to affect numerous pending cases across federal 
        circuits. The American Bar Association has already issued guidance to its 
        members on how to apply the new precedent in ongoing litigation.

        ## Looking Forward

        Legal practitioners should review their current cases in light of this 
        decision. The Court's clear reasoning provides a roadmap for future 
        constitutional challenges and reinforces the importance of precedent in 
        American jurisprudence.
        """,
        summary="Supreme Court ruling in Smith v. Jones clarifies federal jurisdiction in civil rights cases, providing important precedent for future litigation.",
        tags=[
            "supreme court",
            "civil rights",
            "federal jurisdiction",
            "constitutional law",
        ],
        quality_score=0.85,
        originality_score=0.92,
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "success": True,
        "content": "Test LLM response content",
        "tokens_used": 100,
        "cost_estimate": 0.01,
        "response_time": 1.5,
    }


@pytest.fixture
def mock_web_response():
    """Mock web response for testing."""
    return {
        "status": 200,
        "content": """
        <html>
        <head><title>Test Legal Article</title></head>
        <body>
            <h1>Supreme Court Rules on Important Case</h1>
            <div class="article-content">
                <p>The Supreme Court ruled today in a landmark decision...</p>
                <p>Legal experts expect this decision to have far-reaching implications...</p>
            </div>
        </body>
        </html>
        """,
        "headers": {"Content-Type": "text/html"},
        "url": "https://example.com/legal-article",
    }


@pytest.fixture
def mock_telegram_response():
    """Mock Telegram API response for testing."""
    return {
        "ok": True,
        "result": {
            "message_id": 123,
            "chat": {"id": "test-channel", "type": "channel"},
            "date": 1642694400,
            "text": "Test message content",
        },
    }


class AsyncContextManagerMock:
    """Mock async context manager for testing."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"ok": True, "result": {}})
    mock_response.text = AsyncMock(return_value="Test response")

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=AsyncContextManagerMock(mock_response))
    mock_session.post = MagicMock(return_value=AsyncContextManagerMock(mock_response))

    return mock_session


@pytest.fixture
def temp_log_file():
    """Create temporary log file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_file = f.name

    yield log_file

    # Cleanup
    try:
        os.unlink(log_file)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_system_metrics():
    """Mock system metrics for testing."""
    return {
        "memory": {
            "total_gb": 16.0,
            "available_gb": 8.0,
            "used_gb": 8.0,
            "percent": 50.0,
        },
        "cpu": {"percent": 25.0, "count": 8},
        "disk": {
            "total_gb": 500.0,
            "free_gb": 400.0,
            "used_gb": 100.0,
            "percent": 20.0,
        },
    }


# Test data generators
def create_test_article(
    title: str = "Test Article", content_length: int = 100
) -> Dict[str, Any]:
    """Create test article data."""
    content = " ".join(["Test content"] * content_length)
    return {
        "id": "test-id-123",
        "title": title,
        "content": content,
        "source_name": "Test Source",
        "scraped_at": datetime.now(),
        "url": "https://example.com/test",
    }


def create_test_extraction(article_id: str = "test-123") -> Dict[str, Any]:
    """Create test extraction data."""
    return {
        "id": f"extraction-{article_id}",
        "article_id": article_id,
        "entities": [{"name": "Test Court", "type": "court", "relevance_score": 0.8}],
        "legal_topics": ["litigation"],
        "cases": [],
        "dates": [],
        "monetary_amounts": [],
        "key_quotes": ["Test quote from article"],
        "extraction_confidence": 0.8,
        "completeness_score": 0.7,
    }
