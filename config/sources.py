"""
Configuration for authorized legal source URLs and scraping parameters.
Includes per-source settings for rate limits, selectors, and compliance.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, HttpUrl, Field


class SourceConfig(BaseModel):
    """Configuration for a single legal news source."""
    
    name: str = Field(..., description="Human-readable source name")
    url: HttpUrl = Field(..., description="Base URL of the legal source")
    article_list_path: str = Field(..., description="Path to article listing page")
    article_selector: str = Field(..., description="CSS selector for article links")
    title_selector: str = Field(..., description="CSS selector for article titles")
    content_selector: str = Field(..., description="CSS selector for article content")
    date_selector: Optional[str] = Field(None, description="CSS selector for publication date")
    author_selector: Optional[str] = Field(None, description="CSS selector for author")
    
    # Rate limiting and compliance
    rate_limit_delay: float = Field(2.0, description="Seconds between requests to this source")
    max_articles_per_cycle: int = Field(50, description="Maximum articles to scrape per cycle")
    robots_txt_compliant: bool = Field(True, description="Whether to check robots.txt")
    user_agent_required: bool = Field(True, description="Whether user agent is required")
    
    # Content filtering
    min_content_length: int = Field(200, description="Minimum article content length")
    exclude_keywords: List[str] = Field(default_factory=list, description="Keywords to exclude articles")
    include_keywords: List[str] = Field(default_factory=list, description="Keywords to prioritize")
    
    # JavaScript rendering
    requires_javascript: bool = Field(False, description="Whether source requires JS rendering")
    page_load_delay: float = Field(2.0, description="Delay after page load if JS required")


class LegalSources:
    """Registry of authorized legal news sources."""
    
    # Note: These are example sources for demonstration
    # In production, verify compliance with each source's terms of service
    SOURCES: Dict[str, SourceConfig] = {
        "law360": SourceConfig(
            name="Law360",
            url="https://www.law360.com",
            article_list_path="/sections",
            article_selector="a.headline-link",
            title_selector="h1.headline",
            content_selector=".article-text",
            date_selector=".article-date",
            author_selector=".author-name",
            rate_limit_delay=3.0,
            max_articles_per_cycle=25,
            include_keywords=["legal", "court", "lawsuit", "ruling", "regulation"],
            exclude_keywords=["opinion", "editorial", "advertisement"]
        ),
        
        "reuters_legal": SourceConfig(
            name="Reuters Legal",
            url="https://www.reuters.com",
            article_list_path="/legal",
            article_selector="a[data-testid='Heading']",
            title_selector="h1[data-testid='ArticleHeader:headline']",
            content_selector="div[data-testid='ArticleBody']",
            date_selector="time",
            author_selector="a[data-testid='Author']",
            rate_limit_delay=2.5,
            max_articles_per_cycle=30,
            include_keywords=["legal", "court", "law", "regulation", "compliance"],
            exclude_keywords=["sports", "entertainment", "weather"]
        ),
        
        "bloomberg_law": SourceConfig(
            name="Bloomberg Law",
            url="https://news.bloomberglaw.com",
            article_list_path="/",
            article_selector="a.headline-link",
            title_selector="h1.headline",
            content_selector=".story-body",
            date_selector=".timestamp",
            author_selector=".byline",
            rate_limit_delay=4.0,
            max_articles_per_cycle=20,
            requires_javascript=True,
            page_load_delay=3.0,
            include_keywords=["legal", "regulatory", "compliance", "litigation"],
            exclude_keywords=["market", "earnings", "stock"]
        ),
        
        "legal_news_line": SourceConfig(
            name="Legal News Line",
            url="https://legalnewsline.com",
            article_list_path="/",
            article_selector="a.post-title",
            title_selector="h1.entry-title",
            content_selector=".entry-content",
            date_selector=".post-date",
            author_selector=".author-name",
            rate_limit_delay=2.0,
            max_articles_per_cycle=40,
            include_keywords=["lawsuit", "court", "legal", "attorney", "judge"],
            exclude_keywords=["advertisement", "sponsored"]
        )
    }
    
    @classmethod
    def get_source(cls, source_name: str) -> Optional[SourceConfig]:
        """Get configuration for a specific source."""
        return cls.SOURCES.get(source_name)
    
    @classmethod
    def get_all_sources(cls) -> List[SourceConfig]:
        """Get all configured sources."""
        return list(cls.SOURCES.values())
    
    @classmethod
    def get_active_sources(cls) -> List[SourceConfig]:
        """Get sources that are robots.txt compliant."""
        return [source for source in cls.SOURCES.values() if source.robots_txt_compliant]
    
    @classmethod
    def add_source(cls, name: str, config: SourceConfig) -> None:
        """Add a new source configuration."""
        cls.SOURCES[name] = config
    
    @classmethod
    def remove_source(cls, name: str) -> bool:
        """Remove a source configuration."""
        if name in cls.SOURCES:
            del cls.SOURCES[name]
            return True
        return False


# Export for easy access
legal_sources = LegalSources()