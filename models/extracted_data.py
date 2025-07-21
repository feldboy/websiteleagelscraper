"""
Pydantic models for extracted and structured legal data.
Defines data models for processed legal information extracted from articles.
"""
from pydantic import BaseModel, Field, validator, HttpUrl
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import re


class EntityType(str, Enum):
    """Types of legal entities that can be extracted."""
    PERSON = "person"
    COMPANY = "company"
    COURT = "court"
    LAW_FIRM = "law_firm"
    GOVERNMENT = "government"
    JUDGE = "judge"
    ATTORNEY = "attorney"
    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"
    WITNESS = "witness"
    OTHER = "other"


class LegalTopic(str, Enum):
    """Legal practice areas and topics."""
    CORPORATE_LAW = "corporate_law"
    LITIGATION = "litigation"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    EMPLOYMENT_LAW = "employment_law"
    REAL_ESTATE = "real_estate"
    CRIMINAL_LAW = "criminal_law"
    FAMILY_LAW = "family_law"
    TAX_LAW = "tax_law"
    ENVIRONMENTAL_LAW = "environmental_law"
    HEALTHCARE_LAW = "healthcare_law"
    SECURITIES = "securities"
    ANTITRUST = "antitrust"
    BANKRUPTCY = "bankruptcy"
    IMMIGRATION = "immigration"
    REGULATORY = "regulatory"
    CONTRACT_LAW = "contract_law"
    TORT_LAW = "tort_law"
    CONSTITUTIONAL_LAW = "constitutional_law"
    ADMINISTRATIVE_LAW = "administrative_law"
    OTHER = "other"


class Jurisdiction(str, Enum):
    """Legal jurisdictions."""
    FEDERAL = "federal"
    STATE = "state"
    LOCAL = "local"
    INTERNATIONAL = "international"
    TRIBAL = "tribal"


class CourtLevel(str, Enum):
    """Levels of courts."""
    TRIAL = "trial"
    APPELLATE = "appellate"
    SUPREME = "supreme"
    ADMINISTRATIVE = "administrative"
    SPECIALTY = "specialty"


class LegalEntity(BaseModel):
    """A legal entity extracted from an article."""
    
    name: str = Field(..., min_length=1, max_length=200, description="Name of the entity")
    entity_type: EntityType = Field(..., description="Type of legal entity")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to the article (0-1)")
    
    # Optional additional information
    title: Optional[str] = Field(None, max_length=100, description="Professional title or position")
    organization: Optional[str] = Field(None, max_length=200, description="Associated organization")
    location: Optional[str] = Field(None, max_length=100, description="Location/jurisdiction")
    
    @validator('name')
    def validate_name(cls, v):
        """Ensure name is properly formatted."""
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class MonetaryAmount(BaseModel):
    """A monetary amount mentioned in legal context."""
    
    amount: float = Field(..., ge=0.0, description="Numerical amount")
    currency: str = Field("USD", max_length=3, description="Currency code (e.g., USD, EUR)")
    formatted_string: str = Field(..., description="Original formatted string from text")
    context: Optional[str] = Field(None, description="Context where amount was mentioned")
    
    @validator('currency')
    def validate_currency(cls, v):
        """Ensure currency is uppercase 3-letter code."""
        if len(v) != 3 or not v.isalpha():
            raise ValueError('Currency must be 3-letter code')
        return v.upper()


class LegalDate(BaseModel):
    """A date extracted from legal content."""
    
    date_value: date = Field(..., description="The actual date")
    date_type: str = Field(..., description="Type of date (filing, hearing, deadline, etc.)")
    context: Optional[str] = Field(None, description="Context where date was mentioned")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in date extraction")


class CaseReference(BaseModel):
    """A legal case reference or citation."""
    
    case_name: str = Field(..., min_length=1, description="Name of the case")
    parties: List[str] = Field(default_factory=list, description="Parties involved in the case")
    case_number: Optional[str] = Field(None, description="Court case number")
    court: Optional[str] = Field(None, description="Court where case was/is heard")
    year: Optional[int] = Field(None, ge=1800, le=2030, description="Year of the case")
    citation: Optional[str] = Field(None, description="Legal citation if available")
    
    @validator('case_name')
    def validate_case_name(cls, v):
        """Ensure case name is properly formatted."""
        if not v.strip():
            raise ValueError('Case name cannot be empty')
        # Remove extra whitespace
        return re.sub(r'\s+', ' ', v.strip())


class ExtractedData(BaseModel):
    """
    Structured data extracted from a legal article.
    Contains all relevant legal information in structured format.
    """
    
    # Source reference
    article_id: str = Field(..., description="Unique identifier for the source article")
    source_url: HttpUrl = Field(..., description="URL of the source article")
    extracted_at: datetime = Field(default_factory=datetime.now, description="When extraction was performed")
    
    # Extracted entities and information
    cases: List[CaseReference] = Field(default_factory=list, description="Legal cases mentioned")
    entities: List[LegalEntity] = Field(default_factory=list, description="Legal entities involved")
    dates: List[LegalDate] = Field(default_factory=list, description="Important dates mentioned")
    monetary_amounts: List[MonetaryAmount] = Field(default_factory=list, description="Financial amounts mentioned")
    
    # Content analysis
    legal_topics: List[LegalTopic] = Field(default_factory=list, description="Legal practice areas covered")
    key_quotes: List[str] = Field(default_factory=list, description="Significant quotes from the article")
    
    # Legal classification
    jurisdiction: Optional[Jurisdiction] = Field(None, description="Primary jurisdiction")
    court_level: Optional[CourtLevel] = Field(None, description="Court level if applicable")
    practice_areas: List[str] = Field(default_factory=list, description="Specific practice areas")
    
    # Structured metadata
    structured_data: Dict[str, Any] = Field(default_factory=dict, description="Additional structured information")
    
    # Quality metrics
    extraction_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence in extraction")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="How complete the extraction is")
    
    @validator('key_quotes')
    def validate_quotes(cls, v):
        """Ensure quotes are meaningful and not too long."""
        validated_quotes = []
        for quote in v:
            if quote and len(quote.strip()) >= 10:  # Minimum meaningful quote length
                # Truncate very long quotes
                if len(quote) > 500:
                    quote = quote[:497] + "..."
                validated_quotes.append(quote.strip())
        return validated_quotes
    
    @validator('legal_topics')
    def validate_legal_topics(cls, v):
        """Ensure legal topics are unique."""
        return list(set(v))
    
    def get_primary_entities(self, min_relevance: float = 0.7) -> List[LegalEntity]:
        """Get entities with high relevance scores."""
        return [entity for entity in self.entities if entity.relevance_score >= min_relevance]
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[LegalEntity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.entities if entity.entity_type == entity_type]
    
    def get_recent_dates(self, days_ago: int = 365) -> List[LegalDate]:
        """Get dates within the specified time period."""
        cutoff_date = date.today().replace(year=date.today().year - 1) if days_ago >= 365 else date.today()
        return [d for d in self.dates if d.date_value >= cutoff_date]
    
    def get_total_monetary_value(self, currency: str = "USD") -> float:
        """Calculate total monetary value in specified currency."""
        return sum(amount.amount for amount in self.monetary_amounts if amount.currency == currency)
    
    def has_litigation_indicators(self) -> bool:
        """Check if article contains litigation-related content."""
        litigation_topics = [LegalTopic.LITIGATION, LegalTopic.CRIMINAL_LAW, LegalTopic.TORT_LAW]
        litigation_entities = [EntityType.PLAINTIFF, EntityType.DEFENDANT, EntityType.JUDGE, EntityType.COURT]
        
        has_litigation_topics = any(topic in self.legal_topics for topic in litigation_topics)
        has_litigation_entities = any(
            entity.entity_type in litigation_entities for entity in self.entities
        )
        
        return has_litigation_topics or has_litigation_entities or len(self.cases) > 0


class ExtractionQuality(BaseModel):
    """Quality assessment of data extraction."""
    
    extraction_id: str = Field(..., description="Reference to the extraction")
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score (0-10)")
    
    # Individual quality metrics
    entity_accuracy: float = Field(..., ge=0.0, le=10.0, description="Accuracy of entity extraction")
    date_accuracy: float = Field(..., ge=0.0, le=10.0, description="Accuracy of date extraction")
    topic_relevance: float = Field(..., ge=0.0, le=10.0, description="Relevance of identified topics")
    completeness: float = Field(..., ge=0.0, le=10.0, description="Completeness of extraction")
    
    # Issues and recommendations
    issues_found: List[str] = Field(default_factory=list, description="Issues identified in extraction")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Quality flags
    approved: bool = Field(False, description="Whether extraction meets quality standards")
    needs_review: bool = Field(False, description="Whether extraction needs manual review")
    
    assessed_at: datetime = Field(default_factory=datetime.now, description="When assessment was performed")
    
    def calculate_overall_score(self) -> float:
        """Calculate overall score from individual metrics."""
        scores = [self.entity_accuracy, self.date_accuracy, self.topic_relevance, self.completeness]
        return sum(scores) / len(scores)
    
    def is_high_quality(self, threshold: float = 7.0) -> bool:
        """Check if extraction meets high quality threshold."""
        return self.overall_score >= threshold and not self.needs_review