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
    date_selector: Optional[str] = Field(
        None, description="CSS selector for publication date"
    )
    author_selector: Optional[str] = Field(None, description="CSS selector for author")

    # Rate limiting and compliance
    rate_limit_delay: float = Field(
        2.0, description="Seconds between requests to this source"
    )
    max_articles_per_cycle: int = Field(
        50, description="Maximum articles to scrape per cycle"
    )
    robots_txt_compliant: bool = Field(True, description="Whether to check robots.txt")
    user_agent_required: bool = Field(
        True, description="Whether user agent is required"
    )

    # Content filtering
    min_content_length: int = Field(100, description="Minimum article content length")
    exclude_keywords: List[str] = Field(
        default_factory=list, description="Keywords to exclude articles"
    )
    include_keywords: List[str] = Field(
        default_factory=list, description="Keywords to prioritize"
    )

    # JavaScript rendering
    requires_javascript: bool = Field(
        False, description="Whether source requires JS rendering"
    )
    page_load_delay: float = Field(
        2.0, description="Delay after page load if JS required"
    )


class LegalSources:
    """Registry of authorized legal news sources."""

    # Comprehensive legal news sources with proper compliance
    # In production, verify compliance with each source's terms of service
    SOURCES: Dict[str, SourceConfig] = {
        # Major News Agencies - Legal Coverage
        "reuters_legal": SourceConfig(
            name="Reuters Legal",
            url="https://www.reuters.com",
            article_list_path="/legal/",
            article_selector="a[data-testid='Link'], .story-title a, h3 a",
            title_selector="h1[data-testid='ArticleHeadline'], h1.ArticleHeader__headline",
            content_selector="div[data-testid='ArticleBody'], .StandardArticleBody_body",
            date_selector="time, .ArticleHeader__date",
            author_selector=".Attribution__byline, .ArticleHeader__byline",
            rate_limit_delay=3.0,
            max_articles_per_cycle=15,
            include_keywords=["legal", "court", "lawsuit", "ruling", "regulation", "government"],
            exclude_keywords=["sports", "entertainment"],
        ),
        "reuters_government": SourceConfig(
            name="Reuters Government",
            url="https://www.reuters.com",
            article_list_path="/legal/government/",
            article_selector="a[data-testid='Link'], .story-title a, h3 a",
            title_selector="h1[data-testid='ArticleHeadline'], h1.ArticleHeader__headline",
            content_selector="div[data-testid='ArticleBody'], .StandardArticleBody_body",
            date_selector="time, .ArticleHeader__date",
            author_selector=".Attribution__byline, .ArticleHeader__byline",
            rate_limit_delay=3.0,
            max_articles_per_cycle=10,
            include_keywords=["government", "regulation", "policy", "federal", "DOJ"],
            exclude_keywords=["sports", "entertainment"],
        ),
        
        # Professional Legal Publications
        "aba_journal": SourceConfig(
            name="ABA Journal",
            url="https://www.abajournal.com",
            article_list_path="/",
            article_selector="h2 a, h3 a, .entry-title a, .headline a",
            title_selector="h1.entry-title, h1.headline, h1",
            content_selector=".entry-content, .article-content, .post-content",
            date_selector="time, .date, .entry-date",
            author_selector=".author, .byline, .entry-author",
            rate_limit_delay=2.5,
            max_articles_per_cycle=12,
            include_keywords=["legal", "law", "court", "bar", "attorney"],
            exclude_keywords=["advertisement", "sponsored"],
        ),
        "law360": SourceConfig(
            name="Law360",
            url="https://www.law360.com",
            article_list_path="/",
            article_selector="a[href*='/articles/'], .headline a",
            title_selector="h1, .headline, .article-title",
            content_selector=".article-text, .article-content, .body",
            date_selector=".article-date, time, .date",
            author_selector=".author, .byline",
            rate_limit_delay=4.0,
            max_articles_per_cycle=8,
            include_keywords=["legal", "law", "court", "litigation"],
            exclude_keywords=["paywall", "subscriber"],
        ),
        "scotusblog": SourceConfig(
            name="SCOTUSblog",
            url="https://www.scotusblog.com",
            article_list_path="/",
            article_selector="h2 a, h3 a, .entry-title a",
            title_selector="h1.entry-title, h1",
            content_selector=".entry-content, .post-content",
            date_selector="time, .date, .entry-date",
            author_selector=".author, .byline",
            rate_limit_delay=2.0,
            max_articles_per_cycle=10,
            include_keywords=["supreme court", "SCOTUS", "constitutional", "court"],
            exclude_keywords=["advertisement"],
        ),
        
        # Government Sources  
        "us_courts": SourceConfig(
            name="U.S. Courts",
            url="https://www.uscourts.gov",
            article_list_path="/news",
            article_selector="a[href*='/news/'], h3 a, .news-title a",
            title_selector="h1, .page-title, .news-title",
            content_selector=".content, .main-content, .news-content",
            date_selector="time, .date, .news-date",
            author_selector=".author, .byline",
            rate_limit_delay=2.0,
            max_articles_per_cycle=8,
            include_keywords=["court", "federal", "judiciary", "legal"],
            exclude_keywords=["internal", "administrative"],
        ),
        
        # Tech-Legal Coverage
        "wired_law": SourceConfig(
            name="WIRED Law",
            url="https://www.wired.com",
            article_list_path="/tag/law/",
            article_selector="h3 a, .headline a, .title a",
            title_selector="h1, .headline, .article-title",
            content_selector=".article-content, .post-content, .body",
            date_selector="time, .date, .publish-date",
            author_selector=".author, .byline",
            rate_limit_delay=3.0,
            max_articles_per_cycle=10,
            include_keywords=["law", "legal", "privacy", "regulation", "tech"],
            exclude_keywords=["gadgets", "gaming"],
        ),
        "wired_tech_policy": SourceConfig(
            name="WIRED Tech Policy",
            url="https://www.wired.com",
            article_list_path="/tag/tech-policy-and-law/",
            article_selector="h3 a, .headline a, .title a",
            title_selector="h1, .headline, .article-title",
            content_selector=".article-content, .post-content, .body",
            date_selector="time, .date, .publish-date",
            author_selector=".author, .byline",
            rate_limit_delay=3.0,
            max_articles_per_cycle=8,
            include_keywords=["tech policy", "regulation", "privacy", "data"],
            exclude_keywords=["gadgets", "gaming"],
        ),
        
        # General News with Legal Focus
        "axios": SourceConfig(
            name="Axios",
            url="https://www.axios.com",
            article_list_path="/",
            article_selector="h3 a, .headline a, .title a",
            title_selector="h1, .headline, .article-title",
            content_selector=".article-content, .post-content, .body",
            date_selector="time, .date, .publish-date",
            author_selector=".author, .byline",
            rate_limit_delay=2.5,
            max_articles_per_cycle=12,
            include_keywords=["legal", "court", "law", "regulation", "policy"],
            exclude_keywords=["sports", "entertainment"],
        ),
        
        # Legal Resources and Blogs
        "justia": SourceConfig(
            name="Justia",
            url="https://www.justia.com",
            article_list_path="/",
            article_selector="h2 a, h3 a, .title a",
            title_selector="h1, .page-title, .article-title",
            content_selector=".content, .article-content, .main-content",
            date_selector="time, .date, .publish-date",
            author_selector=".author, .byline",
            rate_limit_delay=2.0,
            max_articles_per_cycle=10,
            include_keywords=["legal", "law", "court", "case"],
            exclude_keywords=["advertisement"],
        ),
        "criminal_law_blog": SourceConfig(
            name="Criminal Law Library Blog",
            url="https://www.criminallawlibraryblog.com",
            article_list_path="/",
            article_selector="h2 a, h3 a, .entry-title a",
            title_selector="h1.entry-title, h1",
            content_selector=".entry-content, .post-content",
            date_selector="time, .date, .entry-date",
            author_selector=".author, .byline",
            rate_limit_delay=2.0,
            max_articles_per_cycle=8,
            include_keywords=["criminal", "law", "court", "case"],
            exclude_keywords=["advertisement"],
        ),
        "jd_supra": SourceConfig(
            name="JD Supra",
            url="https://www.jdsupra.com",
            article_list_path="/",
            article_selector="h2 a, h3 a, .title a",
            title_selector="h1, .article-title, .post-title",
            content_selector=".article-content, .post-content, .content",
            date_selector="time, .date, .publish-date",
            author_selector=".author, .byline",
            rate_limit_delay=3.0,
            max_articles_per_cycle=10,
            include_keywords=["legal", "law", "regulation", "compliance"],
            exclude_keywords=["advertisement", "sponsored"],
        ),
        
        # Specialized Legal Sources
        "hk_law": SourceConfig(
            name="Holland & Knight",
            url="https://www.hklaw.com",
            article_list_path="/en/insights/publications/",
            article_selector="h3 a, .title a, .insight-title a",
            title_selector="h1, .page-title, .insight-title",
            content_selector=".content, .insight-content, .article-body",
            date_selector="time, .date, .publish-date",
            author_selector=".author, .byline",
            rate_limit_delay=3.0,
            max_articles_per_cycle=8,
            include_keywords=["legal", "law", "regulation", "antitrust", "DOJ"],
            exclude_keywords=["internal", "client"],
        ),
        
        # Legacy sources (keeping existing ones that work)
        "legal_reader": SourceConfig(
            name="Legal Reader",
            url="https://www.legalreader.com",
            article_list_path="/",
            article_selector="h2 a, h3 a, .entry-title a",
            title_selector="h1.entry-title, h1",
            content_selector=".entry-content, .post-content",
            date_selector="time.published, .date",
            author_selector=".author, .by-author",
            rate_limit_delay=2.0,
            max_articles_per_cycle=15,
            include_keywords=[],  # Accept all legal content
            exclude_keywords=["sports", "entertainment", "weather"],
        ),
        "above_the_law": SourceConfig(
            name="Above the Law",
            url="https://abovethelaw.com",
            article_list_path="/",
            article_selector="h2 a, h3 a, .entry-title a",
            title_selector="h1.entry-title, h1",
            content_selector=".entry-content, .post-content",
            date_selector="time, .entry-date",
            author_selector=".entry-author, .author",
            rate_limit_delay=2.5,
            max_articles_per_cycle=12,
            include_keywords=[],  # Accept all legal content
            exclude_keywords=["advertisement", "sponsored"],
        ),
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
        return [
            source for source in cls.SOURCES.values() if source.robots_txt_compliant
        ]

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
