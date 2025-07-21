"""
Tests for the web scraper agent.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from agents.web_scraper import WebScraperAgent, ContentExtractor
from tools.scraping_tools import WebScraper, ScrapingResult
from config.sources import SourceConfig
from models.scraped_data import ScrapingStatus


class TestContentExtractor:
    """Test the ContentExtractor class."""
    
    def test_extract_content_success(self, mock_web_response):
        """Test successful content extraction."""
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1",
            content_selector=".article-content",
            min_content_length=50
        )
        
        extractor = ContentExtractor()
        result = extractor.extract_content(
            mock_web_response["content"], 
            config, 
            mock_web_response["url"]
        )
        
        assert result is not None
        assert result.title == "Supreme Court Rules on Important Case"
        assert "Supreme Court ruled today" in result.content
        assert result.source_name == "Test Source"
        assert len(result.content) >= config.min_content_length
    
    def test_extract_content_insufficient_length(self, mock_web_response):
        """Test content extraction with insufficient content length."""
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1",
            content_selector=".article-content",
            min_content_length=1000  # Set very high minimum
        )
        
        extractor = ContentExtractor()
        result = extractor.extract_content(
            mock_web_response["content"], 
            config, 
            mock_web_response["url"]
        )
        
        assert result is None
    
    def test_extract_content_no_title_selector(self):
        """Test content extraction when title selector doesn't match."""
        html = """
        <html>
        <body>
            <div class="content">Some article content here.</div>
        </body>
        </html>
        """
        
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1.missing",  # Won't match
            content_selector=".content",
            min_content_length=10
        )
        
        extractor = ContentExtractor()
        result = extractor.extract_content(html, config, "https://example.com/test")
        
        assert result is not None
        assert result.title == "No title found"  # Fallback title


class TestWebScraperAgent:
    """Test the WebScraperAgent class."""
    
    @pytest.fixture
    def scraper_agent(self):
        """Create a WebScraperAgent instance for testing."""
        return WebScraperAgent()
    
    @pytest.mark.asyncio
    async def test_scrape_and_process_article_success(self, scraper_agent, mock_web_response):
        """Test successful article scraping and processing."""
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1",
            content_selector=".article-content",
            min_content_length=50
        )
        
        # Mock the WebScraper
        with patch('agents.web_scraper.WebScraper') as mock_scraper_class:
            mock_scraper = AsyncMock()
            mock_scraper_class.return_value.__aenter__.return_value = mock_scraper
            
            # Mock successful scraping result
            mock_scraper.scrape.return_value = ScrapingResult(
                success=True,
                status=ScrapingStatus.SUCCESS,
                url="https://example.com/test",
                user_agent="test-agent",
                status_code=200,
                content=mock_web_response["content"],
                headers=mock_web_response["headers"],
                scraped_at=datetime.now()
            )
            
            result = await scraper_agent._scrape_and_process_article(
                "https://example.com/test", 
                config
            )
            
            assert result.success
            assert result.status == ScrapingStatus.SUCCESS
            assert result.data is not None
            assert result.data.title == "Supreme Court Rules on Important Case"
    
    @pytest.mark.asyncio
    async def test_scrape_and_process_article_scraping_failure(self, scraper_agent):
        """Test article processing when scraping fails."""
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1",
            content_selector=".article-content"
        )
        
        # Mock the WebScraper to return failure
        with patch('agents.web_scraper.WebScraper') as mock_scraper_class:
            mock_scraper = AsyncMock()
            mock_scraper_class.return_value.__aenter__.return_value = mock_scraper
            
            # Mock failed scraping result
            mock_scraper.scrape.return_value = ScrapingResult(
                success=False,
                status=ScrapingStatus.FAILED,
                url="https://example.com/test",
                user_agent="test-agent",
                error="Connection timeout"
            )
            
            result = await scraper_agent._scrape_and_process_article(
                "https://example.com/test", 
                config
            )
            
            assert not result.success
            assert result.status == ScrapingStatus.FAILED
            assert result.error_message == "Connection timeout"
    
    @pytest.mark.asyncio
    async def test_discover_article_urls(self, scraper_agent):
        """Test article URL discovery from listing page."""
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1",
            content_selector=".article-content"
        )
        
        listing_html = """
        <html>
        <body>
            <a class="headline" href="/article1">Article 1</a>
            <a class="headline" href="/article2">Article 2</a>
            <a class="headline" href="https://example.com/article3">Article 3</a>
        </body>
        </html>
        """
        
        # Mock the WebScraper
        with patch('agents.web_scraper.WebScraper') as mock_scraper_class:
            mock_scraper = AsyncMock()
            mock_scraper_class.return_value.__aenter__.return_value = mock_scraper
            
            # Mock successful listing page scraping
            mock_scraper.scrape.return_value = ScrapingResult(
                success=True,
                status=ScrapingStatus.SUCCESS,
                url="https://example.com/news",
                user_agent="test-agent",
                content=listing_html
            )
            
            urls = await scraper_agent._discover_article_urls(config, max_articles=10)
            
            assert len(urls) == 3
            assert str(urls[0]) == "https://example.com/article1"
            assert str(urls[1]) == "https://example.com/article2"
            assert str(urls[2]) == "https://example.com/article3"
    
    @pytest.mark.asyncio
    async def test_discover_article_urls_with_limit(self, scraper_agent):
        """Test article URL discovery with limit."""
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1",
            content_selector=".article-content"
        )
        
        listing_html = """
        <html>
        <body>
            <a class="headline" href="/article1">Article 1</a>
            <a class="headline" href="/article2">Article 2</a>
            <a class="headline" href="/article3">Article 3</a>
            <a class="headline" href="/article4">Article 4</a>
        </body>
        </html>
        """
        
        # Mock the WebScraper
        with patch('agents.web_scraper.WebScraper') as mock_scraper_class:
            mock_scraper = AsyncMock()
            mock_scraper_class.return_value.__aenter__.return_value = mock_scraper
            
            mock_scraper.scrape.return_value = ScrapingResult(
                success=True,
                status=ScrapingStatus.SUCCESS,
                url="https://example.com/news",
                user_agent="test-agent",
                content=listing_html
            )
            
            urls = await scraper_agent._discover_article_urls(config, max_articles=2)
            
            assert len(urls) == 2  # Should be limited to 2
    
    @pytest.mark.asyncio
    async def test_discover_article_urls_failure(self, scraper_agent):
        """Test article URL discovery when listing page fails."""
        config = SourceConfig(
            name="Test Source",
            url="https://example.com",
            article_list_path="/news",
            article_selector="a.headline",
            title_selector="h1",
            content_selector=".article-content"
        )
        
        # Mock the WebScraper to fail
        with patch('agents.web_scraper.WebScraper') as mock_scraper_class:
            mock_scraper = AsyncMock()
            mock_scraper_class.return_value.__aenter__.return_value = mock_scraper
            
            mock_scraper.scrape.return_value = ScrapingResult(
                success=False,
                status=ScrapingStatus.FAILED,
                url="https://example.com/news",
                user_agent="test-agent",
                error="404 Not Found"
            )
            
            urls = await scraper_agent._discover_article_urls(config)
            
            assert len(urls) == 0
    
    def test_get_batch_status(self, scraper_agent):
        """Test getting batch status."""
        # Initially should be empty
        status = scraper_agent.get_batch_status("non-existent")
        assert status is None
        
        statuses = scraper_agent.get_all_batch_statuses()
        assert len(statuses) == 0