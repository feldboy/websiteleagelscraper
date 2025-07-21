"""
Pydantic models for scraped content and scraping results.
Defines data structure for raw scraped legal news articles.
"""
from pydantic import BaseModel, HttpUrl, Field, validator
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class ScrapingStatus(str, Enum):
    """Status of a scraping operation."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ROBOTS_BLOCKED = "robots_blocked"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"


class ScrapedData(BaseModel):
    """Raw scraped data from a legal news source."""
    
    # Source information
    url: HttpUrl = Field(..., description="Source URL of the article")
    source_name: str = Field(..., description="Name of the news source")
    
    # Content
    title: str = Field(..., min_length=1, max_length=500, description="Article title")
    content: str = Field(..., min_length=100, description="Full article content")
    excerpt: Optional[str] = Field(None, max_length=500, description="Article excerpt/summary")
    
    # Metadata
    author: Optional[str] = Field(None, max_length=200, description="Article author")
    publish_date: Optional[datetime] = Field(None, description="Publication date")
    scraped_at: datetime = Field(default_factory=datetime.now, description="When article was scraped")
    
    # Technical details
    status_code: int = Field(..., ge=200, le=599, description="HTTP status code")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP response headers")
    user_agent: str = Field(..., description="User agent used for scraping")
    response_time: float = Field(..., ge=0.0, description="Response time in seconds")
    
    # Processing flags
    content_length: int = Field(..., ge=0, description="Length of content in characters")
    language: Optional[str] = Field(None, max_length=10, description="Detected language code")
    encoding: Optional[str] = Field(None, max_length=50, description="Content encoding")
    
    @validator('content_length', always=True)
    def validate_content_length(cls, v, values):
        """Ensure content_length matches actual content length."""
        if 'content' in values:
            return len(values['content'])
        return v
    
    @validator('title')
    def validate_title(cls, v):
        """Ensure title is not empty or whitespace only."""
        if not v.strip():
            raise ValueError('Title cannot be empty or whitespace only')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        """Ensure content meets minimum quality standards."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        
        # Check for minimum meaningful content
        if len(v.strip().split()) < 50:
            raise ValueError('Content must contain at least 50 words')
        
        return v.strip()


class ScrapingResult(BaseModel):
    """Result of a scraping operation with success/failure details."""
    
    # Status
    success: bool = Field(..., description="Whether scraping was successful")
    status: ScrapingStatus = Field(..., description="Detailed status of scraping")
    
    # Request details
    url: HttpUrl = Field(..., description="URL that was scraped")
    user_agent: str = Field(..., description="User agent used")
    attempt_number: int = Field(1, ge=1, le=10, description="Attempt number (for retries)")
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now, description="When scraping started")
    completed_at: Optional[datetime] = Field(None, description="When scraping completed")
    response_time: Optional[float] = Field(None, ge=0.0, description="Response time in seconds")
    
    # Results
    data: Optional[ScrapedData] = Field(None, description="Scraped data if successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Type of error that occurred")
    
    # Technical details
    status_code: Optional[int] = Field(None, ge=100, le=599, description="HTTP status code")
    robots_txt_allowed: Optional[bool] = Field(None, description="Whether robots.txt allowed scraping")
    rate_limited: bool = Field(False, description="Whether request was rate limited")
    
    @validator('completed_at')
    def validate_completed_at(cls, v, values):
        """Ensure completed_at is after started_at."""
        if v and 'started_at' in values and v < values['started_at']:
            raise ValueError('completed_at must be after started_at')
        return v
    
    @validator('data')
    def validate_data_consistency(cls, v, values):
        """Ensure data is present only for successful scraping."""
        if values.get('success') and not v:
            raise ValueError('data must be present for successful scraping')
        if not values.get('success') and v:
            raise ValueError('data should not be present for failed scraping')
        return v
    
    @validator('error_message')
    def validate_error_message(cls, v, values):
        """Ensure error message is present for failed scraping."""
        if not values.get('success') and not v:
            raise ValueError('error_message must be present for failed scraping')
        return v
    
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration of scraping operation in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def is_retryable(self) -> bool:
        """Determine if this failed scraping is worth retrying."""
        if self.success:
            return False
        
        retryable_statuses = [
            ScrapingStatus.TIMEOUT,
            ScrapingStatus.RATE_LIMITED,
            ScrapingStatus.FAILED  # Generic failures might be temporary
        ]
        
        non_retryable_statuses = [
            ScrapingStatus.ROBOTS_BLOCKED  # Permanent block
        ]
        
        if self.status in non_retryable_statuses:
            return False
        
        if self.status in retryable_statuses:
            return True
        
        # Check status code for retryable errors
        if self.status_code:
            retryable_codes = [429, 500, 502, 503, 504]
            return self.status_code in retryable_codes
        
        return False


class ScrapingBatch(BaseModel):
    """A batch of scraping operations for tracking and management."""
    
    batch_id: str = Field(..., description="Unique identifier for this batch")
    source_name: str = Field(..., description="Name of the source being scraped")
    urls: list[HttpUrl] = Field(..., min_items=1, description="URLs to scrape in this batch")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now, description="When batch was created")
    started_at: Optional[datetime] = Field(None, description="When batch processing started")
    completed_at: Optional[datetime] = Field(None, description="When batch processing completed")
    
    # Progress tracking
    total_urls: int = Field(..., ge=1, description="Total number of URLs in batch")
    completed_urls: int = Field(0, ge=0, description="Number of URLs completed")
    successful_urls: int = Field(0, ge=0, description="Number of successfully scraped URLs")
    failed_urls: int = Field(0, ge=0, description="Number of failed URLs")
    
    # Results
    results: list[ScrapingResult] = Field(default_factory=list, description="Results for each URL")
    
    @validator('total_urls', always=True)
    def validate_total_urls(cls, v, values):
        """Ensure total_urls matches length of urls list."""
        if 'urls' in values:
            return len(values['urls'])
        return v
    
    @validator('completed_urls')
    def validate_completed_urls(cls, v, values):
        """Ensure completed_urls doesn't exceed total_urls."""
        if 'total_urls' in values and v > values['total_urls']:
            raise ValueError('completed_urls cannot exceed total_urls')
        return v
    
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_urls == 0:
            return 0.0
        return (self.completed_urls / self.total_urls) * 100
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.completed_urls == 0:
            return 0.0
        return (self.successful_urls / self.completed_urls) * 100
    
    def is_complete(self) -> bool:
        """Check if batch processing is complete."""
        return self.completed_urls == self.total_urls
    
    def add_result(self, result: ScrapingResult) -> None:
        """Add a scraping result to the batch."""
        self.results.append(result)
        self.completed_urls = len(self.results)
        
        if result.success:
            self.successful_urls += 1
        else:
            self.failed_urls += 1
        
        # Update completed_at if this was the last URL
        if self.is_complete() and not self.completed_at:
            self.completed_at = datetime.now()