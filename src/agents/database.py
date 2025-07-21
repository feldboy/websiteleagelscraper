"""
Database agent for persistent storage with transactional operations.
Implements SQLAlchemy ORM with async support and data integrity validation.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import hashlib
import json

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (
    Column,
    String,
    DateTime,
    Text,
    Integer,
    Float,
    Boolean,
    JSON,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import select, delete, update
import uuid

from config.settings import settings
from models.scraped_data import ScrapedData, ScrapingResult
from models.extracted_data import ExtractedData, ExtractionQuality


logger = logging.getLogger(__name__)
Base = declarative_base()


class ArticleRecord(Base):
    """Database table for scraped articles."""

    __tablename__ = "articles"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Source information
    url = Column(String(2000), nullable=False, index=True)
    source_name = Column(String(100), nullable=False, index=True)
    content_hash = Column(
        String(64), nullable=False, unique=True, index=True
    )  # For duplicate detection

    # Content
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    excerpt = Column(String(500))
    author = Column(String(200))

    # Timestamps
    publish_date = Column(DateTime)
    scraped_at = Column(DateTime, nullable=False, default=func.now())
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    # Technical details
    status_code = Column(Integer, nullable=False)
    user_agent = Column(String(500), nullable=False)
    response_time = Column(Float, nullable=False)
    content_length = Column(Integer, nullable=False)
    language = Column(String(10))
    encoding = Column(String(50))

    # Processing status
    processed = Column(Boolean, default=False, index=True)
    extraction_completed = Column(Boolean, default=False, index=True)
    quality_checked = Column(Boolean, default=False, index=True)

    # Add database indexes for performance
    __table_args__ = (
        Index("idx_source_date", "source_name", "scraped_at"),
        Index("idx_processed_status", "processed", "extraction_completed"),
        Index("idx_content_hash", "content_hash"),
    )


class ExtractionRecord(Base):
    """Database table for extracted data."""

    __tablename__ = "extractions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Reference to source article
    article_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Extracted data (stored as JSON)
    entities = Column(JSON, nullable=False, default=list)
    cases = Column(JSON, nullable=False, default=list)
    dates = Column(JSON, nullable=False, default=list)
    monetary_amounts = Column(JSON, nullable=False, default=list)
    legal_topics = Column(JSON, nullable=False, default=list)
    key_quotes = Column(JSON, nullable=False, default=list)

    # Classification
    jurisdiction = Column(String(50))
    court_level = Column(String(50))
    practice_areas = Column(JSON, nullable=False, default=list)

    # Quality metrics
    extraction_confidence = Column(Float, default=0.0)
    completeness_score = Column(Float, default=0.0)

    # Additional structured data
    structured_data = Column(JSON, default=dict)

    # Timestamps
    extracted_at = Column(DateTime, nullable=False, default=func.now())
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    # Add indexes
    __table_args__ = (
        Index("idx_article_extraction", "article_id"),
        Index("idx_extraction_date", "extracted_at"),
        Index("idx_confidence_score", "extraction_confidence"),
    )


class QualityRecord(Base):
    """Database table for quality assessments."""

    __tablename__ = "quality_assessments"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Reference to extraction
    extraction_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Quality scores
    overall_score = Column(Float, nullable=False)
    entity_accuracy = Column(Float, nullable=False)
    date_accuracy = Column(Float, nullable=False)
    topic_relevance = Column(Float, nullable=False)
    completeness = Column(Float, nullable=False)

    # Assessment results
    issues_found = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    approved = Column(Boolean, default=False, index=True)
    needs_review = Column(Boolean, default=False, index=True)

    # Timestamps
    assessed_at = Column(DateTime, nullable=False, default=func.now())
    created_at = Column(DateTime, nullable=False, default=func.now())

    __table_args__ = (
        Index("idx_quality_approval", "approved", "needs_review"),
        Index("idx_quality_score", "overall_score"),
    )


class DatabaseAgent:
    """
    Database agent for managing persistent storage with transactional operations.
    Handles article storage, duplicate detection, and data integrity.
    """

    def __init__(self):
        self.engine = None
        self.session_factory = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return

        try:
            # Create async engine
            self.engine = create_async_engine(
                settings.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine, class_=AsyncSession, expire_on_commit=False
            )

            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info("Database agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False

    @asynccontextmanager
    async def get_session(self):
        """Get async database session with automatic cleanup."""
        if not self._initialized:
            await self.initialize()

        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for duplicate detection."""
        # Normalize content: remove extra whitespace, convert to lowercase
        normalized = " ".join(content.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def store_article(self, scraped_data: ScrapedData) -> Optional[str]:
        """
        Store scraped article with duplicate detection.

        Args:
            scraped_data: ScrapedData object to store

        Returns:
            Article ID if stored successfully, None if duplicate
        """
        content_hash = self._calculate_content_hash(scraped_data.content)

        async with self.get_session() as session:
            try:
                # Check for duplicate content
                existing = await session.execute(
                    select(ArticleRecord).where(
                        ArticleRecord.content_hash == content_hash
                    )
                )
                if existing.scalar():
                    logger.info(
                        f"Duplicate content detected for URL: {scraped_data.url}"
                    )
                    return None

                # Create new article record
                article = ArticleRecord(
                    url=str(scraped_data.url),
                    source_name=scraped_data.source_name,
                    content_hash=content_hash,
                    title=scraped_data.title,
                    content=scraped_data.content,
                    excerpt=scraped_data.excerpt,
                    author=scraped_data.author,
                    publish_date=scraped_data.publish_date,
                    scraped_at=scraped_data.scraped_at,
                    status_code=scraped_data.status_code,
                    user_agent=scraped_data.user_agent,
                    response_time=scraped_data.response_time,
                    content_length=scraped_data.content_length,
                    language=scraped_data.language,
                    encoding=scraped_data.encoding,
                )

                session.add(article)
                await session.commit()

                logger.info(f"Stored article: {article.id} from {scraped_data.url}")
                return str(article.id)

            except IntegrityError as e:
                await session.rollback()
                if "content_hash" in str(e):
                    logger.info(
                        f"Duplicate content hash detected for URL: {scraped_data.url}"
                    )
                    return None
                else:
                    logger.error(f"Integrity error storing article: {e}")
                    raise
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing article: {e}")
                raise

    async def store_extraction(self, extracted_data: ExtractedData) -> str:
        """
        Store extracted data with validation.

        Args:
            extracted_data: ExtractedData object to store

        Returns:
            Extraction ID
        """
        async with self.get_session() as session:
            try:
                # Verify article exists
                article_exists = await session.execute(
                    select(ArticleRecord).where(
                        ArticleRecord.id == extracted_data.article_id
                    )
                )
                if not article_exists.scalar():
                    raise ValueError(f"Article {extracted_data.article_id} not found")

                # Create extraction record
                extraction = ExtractionRecord(
                    article_id=extracted_data.article_id,
                    entities=[entity.dict() for entity in extracted_data.entities],
                    cases=[case.dict() for case in extracted_data.cases],
                    dates=[date_obj.dict() for date_obj in extracted_data.dates],
                    monetary_amounts=[
                        amount.dict() for amount in extracted_data.monetary_amounts
                    ],
                    legal_topics=[topic.value for topic in extracted_data.legal_topics],
                    key_quotes=extracted_data.key_quotes,
                    jurisdiction=(
                        extracted_data.jurisdiction.value
                        if extracted_data.jurisdiction
                        else None
                    ),
                    court_level=(
                        extracted_data.court_level.value
                        if extracted_data.court_level
                        else None
                    ),
                    practice_areas=extracted_data.practice_areas,
                    extraction_confidence=extracted_data.extraction_confidence,
                    completeness_score=extracted_data.completeness_score,
                    structured_data=extracted_data.structured_data,
                    extracted_at=extracted_data.extracted_at,
                )

                session.add(extraction)

                # Mark article as extraction completed
                await session.execute(
                    update(ArticleRecord)
                    .where(ArticleRecord.id == extracted_data.article_id)
                    .values(extraction_completed=True, processed=True)
                )

                await session.commit()

                logger.info(
                    f"Stored extraction: {extraction.id} for article {extracted_data.article_id}"
                )
                return str(extraction.id)

            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing extraction: {e}")
                raise

    async def store_quality_assessment(self, quality: ExtractionQuality) -> str:
        """
        Store quality assessment for an extraction.

        Args:
            quality: ExtractionQuality object to store

        Returns:
            Quality assessment ID
        """
        async with self.get_session() as session:
            try:
                # Create quality record
                quality_record = QualityRecord(
                    extraction_id=quality.extraction_id,
                    overall_score=quality.overall_score,
                    entity_accuracy=quality.entity_accuracy,
                    date_accuracy=quality.date_accuracy,
                    topic_relevance=quality.topic_relevance,
                    completeness=quality.completeness,
                    issues_found=quality.issues_found,
                    recommendations=quality.recommendations,
                    approved=quality.approved,
                    needs_review=quality.needs_review,
                    assessed_at=quality.assessed_at,
                )

                session.add(quality_record)

                # Update article quality check status
                extraction_record = await session.execute(
                    select(ExtractionRecord).where(
                        ExtractionRecord.id == quality.extraction_id
                    )
                )
                extraction = extraction_record.scalar()

                if extraction:
                    await session.execute(
                        update(ArticleRecord)
                        .where(ArticleRecord.id == extraction.article_id)
                        .values(quality_checked=True)
                    )

                await session.commit()

                logger.info(
                    f"Stored quality assessment: {quality_record.id} for extraction {quality.extraction_id}"
                )
                return str(quality_record.id)

            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing quality assessment: {e}")
                raise

    async def get_unprocessed_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles that haven't been processed for extraction."""
        async with self.get_session() as session:
            try:
                result = await session.execute(
                    select(ArticleRecord)
                    .where(ArticleRecord.processed == False)
                    .order_by(ArticleRecord.scraped_at.desc())
                    .limit(limit)
                )

                articles = []
                for record in result.scalars():
                    articles.append(
                        {
                            "id": str(record.id),
                            "url": record.url,
                            "title": record.title,
                            "content": record.content,
                            "source_name": record.source_name,
                            "scraped_at": record.scraped_at,
                        }
                    )

                return articles

            except Exception as e:
                logger.error(f"Error getting unprocessed articles: {e}")
                raise

    async def get_extraction_by_article_id(
        self, article_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get extraction data for a specific article."""
        async with self.get_session() as session:
            try:
                result = await session.execute(
                    select(ExtractionRecord).where(
                        ExtractionRecord.article_id == article_id
                    )
                )
                extraction = result.scalar()

                if extraction:
                    return {
                        "id": str(extraction.id),
                        "article_id": str(extraction.article_id),
                        "entities": extraction.entities,
                        "cases": extraction.cases,
                        "dates": extraction.dates,
                        "monetary_amounts": extraction.monetary_amounts,
                        "legal_topics": extraction.legal_topics,
                        "key_quotes": extraction.key_quotes,
                        "jurisdiction": extraction.jurisdiction,
                        "court_level": extraction.court_level,
                        "practice_areas": extraction.practice_areas,
                        "extraction_confidence": extraction.extraction_confidence,
                        "completeness_score": extraction.completeness_score,
                        "structured_data": extraction.structured_data,
                        "extracted_at": extraction.extracted_at,
                    }

                return None

            except Exception as e:
                logger.error(f"Error getting extraction: {e}")
                raise

    async def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        async with self.get_session() as session:
            try:
                # Count articles
                total_articles = await session.execute(
                    select(func.count(ArticleRecord.id))
                )
                processed_articles = await session.execute(
                    select(func.count(ArticleRecord.id)).where(
                        ArticleRecord.processed == True
                    )
                )

                # Count extractions
                total_extractions = await session.execute(
                    select(func.count(ExtractionRecord.id))
                )

                # Count quality assessments
                total_quality = await session.execute(
                    select(func.count(QualityRecord.id))
                )
                approved_quality = await session.execute(
                    select(func.count(QualityRecord.id)).where(
                        QualityRecord.approved == True
                    )
                )

                return {
                    "total_articles": total_articles.scalar() or 0,
                    "processed_articles": processed_articles.scalar() or 0,
                    "total_extractions": total_extractions.scalar() or 0,
                    "total_quality_assessments": total_quality.scalar() or 0,
                    "approved_articles": approved_quality.scalar() or 0,
                }

            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                raise


# Global database agent instance
database_agent = DatabaseAgent()
