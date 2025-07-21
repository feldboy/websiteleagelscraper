"""
Scraping tools with compliance, rate limiting, and user-agent rotation.
Implements robots.txt checking and exponential backoff for defensive scraping.
"""
import asyncio
import aiohttp
import random
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Rate limiter to prevent overwhelming target servers."""
    
    last_requests: Dict[str, datetime] = field(default_factory=dict)
    min_delay: float = 2.0
    
    async def wait(self, domain: str) -> None:
        """Wait appropriate time before making request to domain."""
        current_time = datetime.now()
        
        if domain in self.last_requests:
            time_since_last = (current_time - self.last_requests[domain]).total_seconds()
            if time_since_last < self.min_delay:
                wait_time = self.min_delay - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
                await asyncio.sleep(wait_time)
        
        self.last_requests[domain] = current_time


class UserAgentRotator:
    """Rotates user agents to avoid being blocked."""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36", 
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0"
    ]
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(self.USER_AGENTS)


class RobotsTxtChecker:
    """Checks robots.txt compliance before making requests."""
    
    def __init__(self):
        self._robots_cache: Dict[str, Tuple[RobotFileParser, datetime]] = {}
        self._cache_duration = timedelta(hours=24)  # Cache robots.txt for 24 hours
    
    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not settings.respect_robots_txt:
            return True
            
        try:
            parsed_url = urlparse(url)
            domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_url = urljoin(domain, "/robots.txt")
            
            # Check cache first
            if domain in self._robots_cache:
                rp, cached_time = self._robots_cache[domain]
                if datetime.now() - cached_time < self._cache_duration:
                    return rp.can_fetch(user_agent, url)
            
            # Fetch and parse robots.txt
            rp = RobotFileParser()
            rp.set_url(robots_url)
            
            # Use aiohttp to fetch robots.txt
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(robots_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            robots_content = await response.text()
                            # Write to temp file for robotparser
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                                f.write(robots_content)
                                temp_path = f.name
                            
                            rp.set_url(f"file://{temp_path}")
                            rp.read()
                            
                            # Cache the result
                            self._robots_cache[domain] = (rp, datetime.now())
                            
                            # Clean up temp file
                            import os
                            os.unlink(temp_path)
                            
                            return rp.can_fetch(user_agent, url)
                        else:
                            # If robots.txt not found, assume allowed
                            logger.debug(f"No robots.txt found for {domain}, assuming allowed")
                            return True
                except Exception as e:
                    logger.warning(f"Error fetching robots.txt for {domain}: {e}")
                    return True  # If we can't check, assume allowed
                    
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            return True  # If we can't check, assume allowed


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    
    success: bool
    url: str
    status_code: Optional[int] = None
    content: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    user_agent: Optional[str] = None
    scraped_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0


class WebScraper:
    """
    Async web scraper with compliance, rate limiting, and retry logic.
    Designed for responsible legal news scraping.
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.user_agent_rotator = UserAgentRotator()
        self.robots_checker = RobotsTxtChecker()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.request_timeout),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=3)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def scrape(self, url: str, retry_count: int = 0) -> ScrapingResult:
        """
        Scrape a single URL with full compliance and error handling.
        
        Args:
            url: URL to scrape
            retry_count: Current retry attempt
            
        Returns:
            ScrapingResult with success status and content or error
        """
        if not self.session:
            raise RuntimeError("WebScraper must be used as async context manager")
        
        user_agent = self.user_agent_rotator.get_random_user_agent()
        
        try:
            # Check robots.txt compliance
            if not await self.robots_checker.can_fetch(url, user_agent):
                return ScrapingResult(
                    success=False,
                    url=url,
                    error="Robots.txt disallows scraping this URL",
                    retry_count=retry_count
                )
            
            # Apply rate limiting
            domain = urlparse(url).netloc
            await self.rate_limiter.wait(domain)
            
            # Make the request
            headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            
            async with self.session.get(url, headers=headers) as response:
                content = await response.text()
                
                return ScrapingResult(
                    success=True,
                    url=url,
                    status_code=response.status,
                    content=content,
                    headers=dict(response.headers),
                    user_agent=user_agent,
                    scraped_at=datetime.now(),
                    retry_count=retry_count
                )
                
        except asyncio.TimeoutError:
            error_msg = f"Timeout after {settings.request_timeout}s"
            logger.warning(f"Timeout scraping {url}: {error_msg}")
            
            # Retry with exponential backoff
            if retry_count < settings.max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.info(f"Retrying {url} in {wait_time}s (attempt {retry_count + 1})")
                await asyncio.sleep(wait_time)
                return await self.scrape(url, retry_count + 1)
            
            return ScrapingResult(
                success=False,
                url=url,
                error=error_msg,
                retry_count=retry_count
            )
            
        except Exception as e:
            error_msg = f"Scraping error: {str(e)}"
            logger.error(f"Error scraping {url}: {error_msg}")
            
            # Retry on certain errors
            if retry_count < settings.max_retries and self._should_retry(e):
                wait_time = 2 ** retry_count
                logger.info(f"Retrying {url} in {wait_time}s (attempt {retry_count + 1})")
                await asyncio.sleep(wait_time)
                return await self.scrape(url, retry_count + 1)
            
            return ScrapingResult(
                success=False,
                url=url,
                error=error_msg,
                retry_count=retry_count
            )
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception warrants a retry."""
        # Retry on network errors, temporary server errors
        retry_exceptions = (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            aiohttp.ServerTimeoutError,
        )
        
        # Check for specific HTTP status codes that warrant retry
        if isinstance(exception, aiohttp.ClientResponseError):
            retry_status_codes = [429, 500, 502, 503, 504]
            return exception.status in retry_status_codes
        
        return isinstance(exception, retry_exceptions)
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 3) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently with concurrency limits.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of ScrapingResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> ScrapingResult:
            async with semaphore:
                return await self.scrape(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)


# Convenience function for quick scraping
async def scrape_url(url: str) -> ScrapingResult:
    """Convenience function to scrape a single URL."""
    async with WebScraper() as scraper:
        return await scraper.scrape(url)


async def scrape_urls(urls: List[str], max_concurrent: int = 3) -> List[ScrapingResult]:
    """Convenience function to scrape multiple URLs."""
    async with WebScraper() as scraper:
        return await scraper.scrape_multiple(urls, max_concurrent)