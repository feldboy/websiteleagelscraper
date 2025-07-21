"""
Verification agent for validating scraped content quality and integrity.
Implements content validation pipeline with error handling and re-scraping triggers.
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from pydantic import ValidationError

from models.scraped_data import ScrapedData, ScrapingResult, ScrapingStatus
from agents.database import database_agent
from config.settings import settings


logger = logging.getLogger(__name__)


class ValidationIssue(str, Enum):
    """Types of validation issues that can be detected."""
    CONTENT_TOO_SHORT = "content_too_short"
    TITLE_MISSING = "title_missing"
    CONTENT_DUPLICATE = "content_duplicate"
    LOW_QUALITY_CONTENT = "low_quality_content"
    ENCODING_ISSUES = "encoding_issues"
    MALFORMED_HTML = "malformed_html"
    SUSPICIOUS_CONTENT = "suspicious_content"
    INVALID_DATE = "invalid_date"
    MISSING_METADATA = "missing_metadata"


@dataclass
class ValidationResult:
    """Result of content validation."""
    
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class ContentValidator:
    """Validates scraped content for quality and integrity."""
    
    def __init__(self):
        self.min_word_count = 50
        self.min_sentence_count = 3
        self.min_paragraph_count = 2
    
    def validate_scraped_data(self, data: ScrapedData) -> ValidationResult:
        """
        Comprehensive validation of scraped data.
        
        Args:
            data: ScrapedData object to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        issues = []
        warnings = []
        recommendations = []
        metadata = {}
        
        # Validate basic structure
        try:
            # This will raise ValidationError if data doesn't match schema
            data.dict()
        except ValidationError as e:
            issues.append(ValidationIssue.MALFORMED_HTML)
            warnings.append(f"Data validation error: {e}")
        
        # Content length validation
        word_count = len(data.content.split())
        metadata['word_count'] = word_count
        
        if word_count < self.min_word_count:
            issues.append(ValidationIssue.CONTENT_TOO_SHORT)
            warnings.append(f"Content too short: {word_count} words (min: {self.min_word_count})")
        
        # Title validation
        if not data.title or len(data.title.strip()) < 5:
            issues.append(ValidationIssue.TITLE_MISSING)
            warnings.append("Title is missing or too short")
        
        # Content quality checks
        quality_score = self._assess_content_quality(data.content)
        metadata['quality_score'] = quality_score
        
        if quality_score < 0.5:
            issues.append(ValidationIssue.LOW_QUALITY_CONTENT)
            warnings.append(f"Low content quality score: {quality_score:.2f}")
        
        # Encoding and text issues
        encoding_issues = self._check_encoding_issues(data.content)
        if encoding_issues:
            issues.append(ValidationIssue.ENCODING_ISSUES)
            warnings.extend(encoding_issues)
        
        # Suspicious content detection
        suspicious_indicators = self._detect_suspicious_content(data.content, data.title)
        if suspicious_indicators:
            issues.append(ValidationIssue.SUSPICIOUS_CONTENT)
            warnings.extend(suspicious_indicators)
        
        # Date validation
        if data.publish_date:
            date_issues = self._validate_date(data.publish_date)
            if date_issues:
                issues.append(ValidationIssue.INVALID_DATE)
                warnings.extend(date_issues)
        
        # Metadata completeness
        metadata_score = self._assess_metadata_completeness(data)
        metadata['metadata_completeness'] = metadata_score
        
        if metadata_score < 0.3:
            issues.append(ValidationIssue.MISSING_METADATA)
            warnings.append("Missing important metadata fields")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, metadata)
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(issues, quality_score, metadata_score)
        
        # Determine if content is valid
        critical_issues = [
            ValidationIssue.CONTENT_TOO_SHORT,
            ValidationIssue.TITLE_MISSING,
            ValidationIssue.MALFORMED_HTML
        ]
        is_valid = not any(issue in critical_issues for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metadata=metadata
        )
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess the quality of content text."""
        if not content:
            return 0.0
        
        score = 1.0
        
        # Check sentence structure
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < self.min_sentence_count:
            score -= 0.3
        
        # Check paragraph structure
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < self.min_paragraph_count:
            score -= 0.2
        
        # Check for repetitive content
        words = content.lower().split()
        if len(set(words)) / len(words) < 0.3:  # Low vocabulary diversity
            score -= 0.2
        
        # Check for legal terminology (positive indicator)
        legal_terms = [
            'court', 'judge', 'lawsuit', 'plaintiff', 'defendant', 'attorney',
            'legal', 'law', 'ruling', 'decision', 'case', 'trial', 'appeal',
            'jurisdiction', 'statute', 'regulation', 'compliance', 'litigation'
        ]
        
        legal_term_count = sum(1 for term in legal_terms if term in content.lower())
        if legal_term_count >= 3:
            score += 0.1  # Bonus for legal content
        
        # Check for proper capitalization
        if content.islower() or content.isupper():
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _check_encoding_issues(self, content: str) -> List[str]:
        """Check for encoding and character issues."""
        issues = []
        
        # Check for common encoding problems
        encoding_indicators = ['�', '?', '\x00', '\ufffd']
        for indicator in encoding_indicators:
            if indicator in content:
                issues.append(f"Potential encoding issue: found '{indicator}' character")
        
        # Check for excessive HTML entities
        html_entity_count = len(re.findall(r'&[a-zA-Z]+;', content))
        if html_entity_count > 10:
            issues.append(f"Excessive HTML entities found: {html_entity_count}")
        
        # Check for control characters
        control_chars = re.findall(r'[\x00-\x1f\x7f-\x9f]', content)
        if len(control_chars) > 5:
            issues.append(f"Control characters found: {len(control_chars)}")
        
        return issues
    
    def _detect_suspicious_content(self, content: str, title: str) -> List[str]:
        """Detect potentially suspicious or low-quality content."""
        indicators = []
        
        # Check for paywalls or access restrictions
        paywall_phrases = [
            'subscribe to continue', 'login to read', 'premium content',
            'paywall', 'subscription required', 'register to access'
        ]
        
        full_text = f"{title} {content}".lower()
        for phrase in paywall_phrases:
            if phrase in full_text:
                indicators.append(f"Potential paywall detected: '{phrase}'")
        
        # Check for placeholder content
        placeholder_phrases = [
            'lorem ipsum', 'placeholder text', 'coming soon',
            'under construction', 'content not available'
        ]
        
        for phrase in placeholder_phrases:
            if phrase in full_text:
                indicators.append(f"Placeholder content detected: '{phrase}'")
        
        # Check for excessive advertising content
        ad_phrases = [
            'advertisement', 'sponsored content', 'promoted post',
            'click here', 'buy now', 'limited time offer'
        ]
        
        ad_count = sum(1 for phrase in ad_phrases if phrase in full_text)
        if ad_count >= 3:
            indicators.append(f"Excessive advertising content: {ad_count} indicators")
        
        # Check for error pages
        error_phrases = [
            '404 not found', 'page not found', 'access denied',
            'server error', 'temporarily unavailable'
        ]
        
        for phrase in error_phrases:
            if phrase in full_text:
                indicators.append(f"Error page content: '{phrase}'")
        
        return indicators
    
    def _validate_date(self, publish_date: datetime) -> List[str]:
        """Validate publication date for reasonableness."""
        issues = []
        
        now = datetime.now()
        
        # Check if date is in the future
        if publish_date > now + timedelta(days=1):
            issues.append(f"Publication date is in the future: {publish_date}")
        
        # Check if date is too old (more than 10 years)
        if publish_date < now - timedelta(days=3650):
            issues.append(f"Publication date is very old: {publish_date}")
        
        # Check for obviously wrong dates (before year 1990)
        if publish_date.year < 1990:
            issues.append(f"Publication date seems incorrect: {publish_date}")
        
        return issues
    
    def _assess_metadata_completeness(self, data: ScrapedData) -> float:
        """Assess completeness of metadata fields."""
        total_fields = 5
        present_fields = 0
        
        if data.title and len(data.title.strip()) > 0:
            present_fields += 1
        
        if data.author and len(data.author.strip()) > 0:
            present_fields += 1
        
        if data.publish_date:
            present_fields += 1
        
        if data.excerpt and len(data.excerpt.strip()) > 0:
            present_fields += 1
        
        if data.source_name and len(data.source_name.strip()) > 0:
            present_fields += 1
        
        return present_fields / total_fields
    
    def _generate_recommendations(self, issues: List[ValidationIssue], metadata: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        if ValidationIssue.CONTENT_TOO_SHORT in issues:
            recommendations.append("Consider scraping from a different URL or checking content selectors")
        
        if ValidationIssue.LOW_QUALITY_CONTENT in issues:
            recommendations.append("Review content extraction selectors to avoid navigation/footer content")
        
        if ValidationIssue.ENCODING_ISSUES in issues:
            recommendations.append("Check encoding settings and HTML parsing configuration")
        
        if ValidationIssue.SUSPICIOUS_CONTENT in issues:
            recommendations.append("Verify source URL and check for paywall or access restrictions")
        
        if ValidationIssue.MISSING_METADATA in issues:
            recommendations.append("Review metadata extraction selectors for author, date, etc.")
        
        # Quality-based recommendations
        quality_score = metadata.get('quality_score', 0.0)
        if quality_score < 0.7:
            recommendations.append("Consider improving content extraction to filter out non-article content")
        
        return recommendations
    
    def _calculate_confidence_score(
        self, 
        issues: List[ValidationIssue], 
        quality_score: float, 
        metadata_score: float
    ) -> float:
        """Calculate overall confidence score for the validation."""
        base_score = 1.0
        
        # Deduct points for issues
        critical_issues = [
            ValidationIssue.CONTENT_TOO_SHORT,
            ValidationIssue.TITLE_MISSING,
            ValidationIssue.MALFORMED_HTML
        ]
        
        for issue in issues:
            if issue in critical_issues:
                base_score -= 0.3
            else:
                base_score -= 0.1
        
        # Factor in quality and metadata scores
        combined_score = (base_score + quality_score + metadata_score) / 3
        
        return max(0.0, min(1.0, combined_score))


class VerificationAgent:
    """
    Verification agent that validates scraped content and triggers re-scraping.
    Implements validation pipeline with quality checks and error handling.
    """
    
    def __init__(self):
        self.validator = ContentValidator()
        self.retry_threshold = 0.5  # Minimum confidence score to avoid retry
    
    async def verify_scraped_data(self, data: ScrapedData) -> ValidationResult:
        """
        Verify scraped data quality and integrity.
        
        Args:
            data: ScrapedData object to verify
            
        Returns:
            ValidationResult with verification status
        """
        try:
            result = self.validator.validate_scraped_data(data)
            
            logger.info(
                f"Verification completed for {data.url}: "
                f"valid={result.is_valid}, confidence={result.confidence_score:.2f}"
            )
            
            if result.warnings:
                logger.warning(f"Validation warnings for {data.url}: {result.warnings}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during verification of {data.url}: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[ValidationIssue.MALFORMED_HTML],
                warnings=[f"Verification error: {e}"],
                recommendations=["Manual review required"],
                metadata={}
            )
    
    async def verify_scraping_result(self, result: ScrapingResult) -> ValidationResult:
        """Verify a complete scraping result."""
        if not result.success or not result.data:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[ValidationIssue.MALFORMED_HTML],
                warnings=["Scraping failed or no data returned"],
                recommendations=["Check scraping configuration and source availability"],
                metadata={}
            )
        
        return await self.verify_scraped_data(result.data)
    
    async def should_retry_scraping(self, result: ScrapingResult) -> Tuple[bool, str]:
        """
        Determine if scraping should be retried based on verification results.
        
        Args:
            result: ScrapingResult to evaluate
            
        Returns:
            Tuple of (should_retry, reason)
        """
        if not result.success:
            # Check if the original scraping failure is retryable
            if result.is_retryable():
                return True, f"Retryable scraping error: {result.error_message}"
            else:
                return False, f"Non-retryable scraping error: {result.error_message}"
        
        if not result.data:
            return True, "No data extracted from successful scraping"
        
        # Verify the scraped data
        validation = await self.verify_scraped_data(result.data)
        
        if not validation.is_valid:
            critical_issues = [
                ValidationIssue.CONTENT_TOO_SHORT,
                ValidationIssue.TITLE_MISSING,
                ValidationIssue.SUSPICIOUS_CONTENT
            ]
            
            has_critical_issues = any(issue in critical_issues for issue in validation.issues)
            if has_critical_issues:
                return True, f"Critical validation issues: {validation.issues}"
        
        if validation.confidence_score < self.retry_threshold:
            return True, f"Low confidence score: {validation.confidence_score:.2f}"
        
        return False, "Validation passed"
    
    async def batch_verify_articles(self, article_ids: List[str]) -> List[Tuple[str, ValidationResult]]:
        """
        Verify multiple articles from the database.
        
        Args:
            article_ids: List of article IDs to verify
            
        Returns:
            List of tuples (article_id, ValidationResult)
        """
        results = []
        
        for article_id in article_ids:
            try:
                # Get article from database
                articles = await database_agent.get_unprocessed_articles(limit=1)
                # This is a simplified version - in practice, you'd get specific articles by ID
                
                if articles:
                    article_data = articles[0]
                    
                    # Convert to ScrapedData for validation
                    scraped_data = ScrapedData(
                        url=article_data['url'],
                        source_name=article_data['source_name'],
                        title=article_data['title'],
                        content=article_data['content'],
                        scraped_at=article_data['scraped_at'],
                        status_code=200,
                        headers={},
                        user_agent="verification",
                        response_time=0.0
                    )
                    
                    validation = await self.verify_scraped_data(scraped_data)
                    results.append((article_id, validation))
                else:
                    # Article not found
                    validation = ValidationResult(
                        is_valid=False,
                        confidence_score=0.0,
                        issues=[ValidationIssue.MALFORMED_HTML],
                        warnings=["Article not found in database"],
                        recommendations=["Check article ID"],
                        metadata={}
                    )
                    results.append((article_id, validation))
                    
            except Exception as e:
                logger.error(f"Error verifying article {article_id}: {e}")
                validation = ValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    issues=[ValidationIssue.MALFORMED_HTML],
                    warnings=[f"Verification error: {e}"],
                    recommendations=["Manual review required"],
                    metadata={}
                )
                results.append((article_id, validation))
        
        return results
    
    async def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        try:
            db_stats = await database_agent.get_database_stats()
            
            return {
                'total_articles': db_stats.get('total_articles', 0),
                'processed_articles': db_stats.get('processed_articles', 0),
                'verification_rate': (
                    db_stats.get('processed_articles', 0) / 
                    max(db_stats.get('total_articles', 1), 1)
                ),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting verification stats: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}


# Global verification agent instance
verification_agent = VerificationAgent()