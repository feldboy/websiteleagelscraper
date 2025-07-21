"""
Web scraper agent with compliance and queue-based processing.
Implements async scraping with proper rate limiting and robots.txt compliance.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse
import uuid

from bs4 import BeautifulSoup
from pydantic import HttpUrl

from config.settings import settings
from config.sources import legal_sources, SourceConfig
from tools.scraping_tools import WebScraper, ScrapingResult
from models.scraped_data import ScrapedData, ScrapingBatch, ScrapingStatus
from agents.database import database_agent


logger = logging.getLogger(__name__)


class ContentExtractor:
    """Extracts structured content from HTML using CSS selectors."""

    @staticmethod
    def extract_content(
        html: str, config: SourceConfig, url: str
    ) -> Optional[ScrapedData]:
        """
        Extract structured content from HTML based on source configuration.

        Args:
            html: Raw HTML content
            config: Source configuration with selectors
            url: Source URL

        Returns:
            ScrapedData object if extraction successful, None otherwise
        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Extract title
            title_element = soup.select_one(config.title_selector)
            title = title_element.get_text(strip=True) if title_element else ""

            if not title:
                # Fallback to HTML title tag
                title_tag = soup.find("title")
                title = (
                    title_tag.get_text(strip=True) if title_tag else "No title found"
                )

            # Extract main content with fallback selectors
            content_element = soup.select_one(config.content_selector)
            
            # Fallback content selectors if primary fails
            fallback_selectors = [
                "article .content",
                ".article-body", 
                ".post-body",
                ".entry-content",
                ".post-content",
                "article",
                "main",
                ".content"
            ]
            
            if not content_element:
                for fallback in fallback_selectors:
                    content_element = soup.select_one(fallback)
                    if content_element:
                        break
            
            if not content_element:
                # Last resort - try to extract from paragraphs
                paragraphs = soup.find_all('p')
                if len(paragraphs) > 3:  # Need at least some content
                    content_element = soup.new_tag('div')
                    for p in paragraphs[:10]:  # Take first 10 paragraphs
                        content_element.append(p)
                
            if not content_element:
                logger.warning(
                    f"No content found with selector {config.content_selector} or fallbacks for {url}"
                )
                return None

            # Clean and extract text content
            content = ContentExtractor._clean_content(content_element.get_text())

            if len(content) < config.min_content_length:
                logger.warning(f"Content too short ({len(content)} chars) for {url}")
                return None

            # Extract optional fields
            author = None
            if config.author_selector:
                author_element = soup.select_one(config.author_selector)
                author = author_element.get_text(strip=True) if author_element else None

            publish_date = None
            if config.date_selector:
                date_element = soup.select_one(config.date_selector)
                if date_element:
                    date_text = date_element.get_text(strip=True)
                    publish_date = ContentExtractor._parse_date(date_text)

            # Create excerpt from content
            excerpt = ContentExtractor._create_excerpt(content)

            # Check content filters
            if not ContentExtractor._passes_filters(content, title, config):
                logger.info(f"Content filtered out for {url}")
                return None

            return ScrapedData(
                url=HttpUrl(url),
                source_name=config.name,
                title=title,
                content=content,
                excerpt=excerpt,
                author=author,
                publish_date=publish_date,
                scraped_at=datetime.now(),
                status_code=200,  # Will be updated by scraper
                headers={},  # Will be updated by scraper
                user_agent="",  # Will be updated by scraper
                response_time=0.0,  # Will be updated by scraper
                content_length=len(content),
            )

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None

    @staticmethod
    def _clean_content(text: str) -> str:
        """Clean extracted text content."""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned = " ".join(lines)

        # Remove common HTML artifacts
        import re

        cleaned = re.sub(r"\s+", " ", cleaned)  # Multiple spaces to single
        cleaned = re.sub(r"^\s*Advertisement\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*Read more.*$", "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    @staticmethod
    def _create_excerpt(content: str, max_length: int = 300) -> str:
        """Create an excerpt from content."""
        if len(content) <= max_length:
            return content

        # Find a good break point (end of sentence)
        excerpt = content[:max_length]
        last_period = excerpt.rfind(".")
        last_exclamation = excerpt.rfind("!")
        last_question = excerpt.rfind("?")

        break_point = max(last_period, last_exclamation, last_question)

        if break_point > max_length * 0.7:  # Only use if reasonably close to max
            return content[: break_point + 1]
        else:
            return content[:max_length] + "..."

    @staticmethod
    def _parse_date(date_text: str) -> Optional[datetime]:
        """Parse date from various formats."""
        import dateparser

        try:
            return dateparser.parse(date_text)
        except Exception:
            return None

    @staticmethod
    def _passes_filters(content: str, title: str, config: SourceConfig) -> bool:
        """Check if content passes include/exclude filters."""
        # Temporarily disable all filtering to get articles
        return True
        
        full_text = f"{title} {content}".lower()

        # Check exclude keywords
        if config.exclude_keywords:
            for keyword in config.exclude_keywords:
                if keyword.lower() in full_text:
                    return False

        # Check include keywords (if specified, at least one must match)
        if config.include_keywords:
            return any(
                keyword.lower() in full_text for keyword in config.include_keywords
            )

        return True


class WebScraperAgent:
    """
    Web scraper agent that coordinates scraping of legal news sources.
    Implements queue-based processing with compliance and error handling.
    """

    def __init__(self):
        self.content_extractor = ContentExtractor()
        self._running = False
        self._current_batches: Dict[str, ScrapingBatch] = {}

    async def scrape_source(
        self, source_name: str, max_articles: Optional[int] = None
    ) -> ScrapingBatch:
        """
        Scrape articles from a specific legal news source.

        Args:
            source_name: Name of the source to scrape
            max_articles: Maximum number of articles to scrape

        Returns:
            ScrapingBatch with results
        """
        source_config = legal_sources.get_source(source_name)
        if not source_config:
            raise ValueError(f"Unknown source: {source_name}")

        # Get article URLs to scrape
        article_urls = await self._discover_article_urls(source_config, max_articles)

        if not article_urls:
            logger.warning(f"No articles found for source: {source_name}")
            return ScrapingBatch(
                batch_id=str(uuid.uuid4()),
                source_name=source_name,
                urls=[],
                total_urls=0,
            )

        # Create scraping batch
        batch = ScrapingBatch(
            batch_id=str(uuid.uuid4()),
            source_name=source_name,
            urls=article_urls,
            total_urls=len(article_urls),
            started_at=datetime.now(),
        )

        self._current_batches[batch.batch_id] = batch

        try:
            # Scrape articles with controlled concurrency
            await self._scrape_article_batch(batch, source_config)

            batch.completed_at = datetime.now()
            logger.info(
                f"Completed scraping batch {batch.batch_id}: "
                f"{batch.successful_urls}/{batch.total_urls} successful"
            )

            return batch

        except Exception as e:
            logger.error(f"Error in scraping batch {batch.batch_id}: {e}")
            raise
        finally:
            # Clean up
            if batch.batch_id in self._current_batches:
                del self._current_batches[batch.batch_id]

    async def _discover_article_urls(
        self, config: SourceConfig, max_articles: Optional[int] = None
    ) -> List[HttpUrl]:
        """Discover article URLs from source listing page."""
        listing_url = urljoin(str(config.url), config.article_list_path)

        async with WebScraper() as scraper:
            result = await scraper.scrape(listing_url)

            if not result.success:
                logger.error(
                    f"Failed to scrape listing page {listing_url}: {result.error}"
                )
                return []

            # Parse HTML and extract article links
            soup = BeautifulSoup(result.content, "html.parser")
            article_elements = soup.select(config.article_selector)

            urls = []
            for element in article_elements:
                href = element.get("href")
                if href:
                    # Convert relative URLs to absolute
                    if href.startswith("/"):
                        href = urljoin(str(config.url), href)
                    elif not href.startswith("http"):
                        continue  # Skip invalid URLs

                    try:
                        url = HttpUrl(href)
                        urls.append(url)

                        # Stop if we've reached the limit
                        if max_articles and len(urls) >= max_articles:
                            break

                    except Exception as e:
                        logger.warning(f"Invalid URL found: {href}, error: {e}")
                        continue

            logger.info(f"Discovered {len(urls)} article URLs for {config.name}")
            return urls

    async def _scrape_article_batch(
        self, batch: ScrapingBatch, config: SourceConfig
    ) -> None:
        """Scrape a batch of articles with controlled concurrency."""
        # Limit concurrent requests to respect server resources
        max_concurrent = min(3, max(1, len(batch.urls) // 10))
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_single_article(url: HttpUrl) -> None:
            async with semaphore:
                try:
                    # Add rate limiting specific to this source
                    await asyncio.sleep(config.rate_limit_delay)

                    result = await self._scrape_and_process_article(str(url), config)
                    batch.add_result(result)

                    # Store successful articles in database
                    if result.success and result.data:
                        try:
                            article_id = await database_agent.store_article(result.data)
                            if article_id:
                                logger.info(f"Stored article {article_id} from {url}")
                            else:
                                logger.info(f"Duplicate article detected: {url}")
                        except Exception as e:
                            logger.error(f"Failed to store article from {url}: {e}")

                except Exception as e:
                    # Create failed result
                    failed_result = ScrapingResult(
                        success=False,
                        status=ScrapingStatus.FAILED,
                        url=url,
                        user_agent="",
                        error_message=str(e),
                        error_type=type(e).__name__,
                    )
                    batch.add_result(failed_result)
                    logger.error(f"Failed to scrape {url}: {e}")

        # Execute all scraping tasks
        tasks = [scrape_single_article(url) for url in batch.urls]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _scrape_and_process_article(
        self, url: str, config: SourceConfig
    ) -> ScrapingResult:
        """Scrape and process a single article."""
        async with WebScraper() as scraper:
            scraping_result = await scraper.scrape(url)

            if not scraping_result.success:
                return ScrapingResult(
                    success=False,
                    status=ScrapingStatus.FAILED,
                    url=HttpUrl(url),
                    user_agent=scraping_result.user_agent or "",
                    status_code=scraping_result.status_code,
                    error_message=scraping_result.error,
                    error_type="ScrapingError",
                )

            # Extract structured content
            scraped_data = self.content_extractor.extract_content(
                scraping_result.content, config, url
            )

            if not scraped_data:
                return ScrapingResult(
                    success=False,
                    status=ScrapingStatus.FAILED,
                    url=HttpUrl(url),
                    user_agent=scraping_result.user_agent,
                    status_code=scraping_result.status_code,
                    error_message="Failed to extract content",
                    error_type="ExtractionError",
                )

            # Update scraped data with technical details
            scraped_data.status_code = scraping_result.status_code
            scraped_data.headers = scraping_result.headers or {}
            scraped_data.user_agent = scraping_result.user_agent
            scraped_data.response_time = (
                (
                    scraping_result.scraped_at - scraping_result.scraped_at
                ).total_seconds()
                if scraping_result.scraped_at
                else 0.0
            )

            return ScrapingResult(
                success=True,
                status=ScrapingStatus.SUCCESS,
                url=HttpUrl(url),
                user_agent=scraping_result.user_agent,
                status_code=scraping_result.status_code,
                data=scraped_data,
                completed_at=datetime.now(),
            )

    async def scrape_all_sources(
        self, max_articles_per_source: Optional[int] = None
    ) -> List[ScrapingBatch]:
        """Scrape all configured legal news sources."""
        # Get source keys and configs
        source_items = [(key, config) for key, config in legal_sources.SOURCES.items() if config.robots_txt_compliant]

        if not source_items:
            logger.warning("No active sources configured")
            return []

        batches = []
        for source_key, source_config in source_items:
            try:
                max_articles = (
                    max_articles_per_source or source_config.max_articles_per_cycle
                )
                batch = await self.scrape_source(source_key, max_articles)
                batches.append(batch)

                # Add delay between sources to be respectful
                await asyncio.sleep(2.0)

            except Exception as e:
                logger.error(f"Failed to scrape source {source_config.name}: {e}")
                continue

        return batches

    def get_batch_status(self, batch_id: str) -> Optional[ScrapingBatch]:
        """Get status of a running scraping batch."""
        return self._current_batches.get(batch_id)

    def get_all_batch_statuses(self) -> List[ScrapingBatch]:
        """Get status of all running batches."""
        return list(self._current_batches.values())


# Global web scraper agent instance
web_scraper_agent = WebScraperAgent()
