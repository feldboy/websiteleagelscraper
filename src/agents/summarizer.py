"""
Summarization agent for creating concise legal article summaries.
Implements content summarization with legal significance analysis and key point extraction.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid
import re

from pydantic import BaseModel, Field

from tools.llm_client import llm_client
from config.prompts import LegalPrompts
from config.settings import settings
from agents.database import database_agent


logger = logging.getLogger(__name__)


class ArticleSummary(BaseModel):
    """Structured summary of a legal article."""

    article_id: str = Field(..., description="ID of the source article")
    source_name: str = Field(..., description="Name of the source publication")
    summary: str = Field(
        ..., min_length=50, description="Concise article summary"
    )
    key_points: List[str] = Field(
        ..., min_items=2, max_items=5, description="Key points from the article"
    )
    legal_significance: str = Field(
        ..., min_length=50, description="Legal significance and implications"
    )

    # Classification
    urgency_level: str = Field("medium", description="Urgency level: low, medium, high")
    impact_scope: str = Field(
        "industry",
        description="Impact scope: local, state, national, international, industry",
    )

    # Quality metrics
    summary_quality: float = Field(
        0.0, ge=0.0, le=1.0, description="Quality score of the summary"
    )
    relevance_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Relevance to legal audience"
    )

    # Metadata
    summarized_at: datetime = Field(
        default_factory=datetime.now, description="When summary was created"
    )
    word_count: int = Field(0, description="Word count of the summary")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.word_count:
            self.word_count = len(self.summary.split())


class SummaryQualityAssessment(BaseModel):
    """Quality assessment for article summaries."""

    summary_id: str = Field(..., description="ID of the summary being assessed")

    # Quality metrics (0-10 scale)
    clarity_score: float = Field(
        ..., ge=0.0, le=10.0, description="Clarity and readability"
    )
    accuracy_score: float = Field(..., ge=0.0, le=10.0, description="Factual accuracy")
    completeness_score: float = Field(
        ..., ge=0.0, le=10.0, description="Coverage of key information"
    )
    legal_precision_score: float = Field(
        ..., ge=0.0, le=10.0, description="Legal terminology precision"
    )

    # Overall assessment
    overall_score: float = Field(
        ..., ge=0.0, le=10.0, description="Overall quality score"
    )
    meets_standards: bool = Field(
        False, description="Whether summary meets quality standards"
    )

    # Issues and recommendations
    issues_identified: List[str] = Field(
        default_factory=list, description="Issues found in summary"
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )

    assessed_at: datetime = Field(
        default_factory=datetime.now, description="When assessment was performed"
    )


class SummaryProcessor:
    """Processes and validates article summaries."""

    def __init__(self):
        self.min_word_count = 100
        self.max_word_count = 200
        self.min_key_points = 2
        self.max_key_points = 5

    def validate_summary_structure(
        self, summary: str, key_points: List[str]
    ) -> List[str]:
        """Validate summary structure and content."""
        issues = []

        # Check word count
        word_count = len(summary.split())
        if word_count < self.min_word_count:
            issues.append(
                f"Summary too short: {word_count} words (min: {self.min_word_count})"
            )
        elif word_count > self.max_word_count:
            issues.append(
                f"Summary too long: {word_count} words (max: {self.max_word_count})"
            )

        # Check key points
        if len(key_points) < self.min_key_points:
            issues.append(
                f"Too few key points: {len(key_points)} (min: {self.min_key_points})"
            )
        elif len(key_points) > self.max_key_points:
            issues.append(
                f"Too many key points: {len(key_points)} (max: {self.max_key_points})"
            )

        # Check for proper sentence structure
        sentences = re.split(r"[.!?]+", summary)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 3:
            issues.append("Summary should contain at least 3 sentences")

        # Check for legal content indicators
        legal_terms = [
            "court",
            "judge",
            "ruling",
            "lawsuit",
            "legal",
            "law",
            "case",
            "attorney",
            "plaintiff",
            "defendant",
            "jurisdiction",
            "statute",
        ]

        summary_lower = summary.lower()
        legal_term_count = sum(1 for term in legal_terms if term in summary_lower)

        if legal_term_count == 0:
            issues.append("Summary lacks legal terminology - may not be legal content")

        return issues

    def assess_summary_quality(
        self, summary: ArticleSummary, original_content: str
    ) -> SummaryQualityAssessment:
        """Assess the quality of a generated summary."""
        # Calculate individual quality scores
        clarity_score = self._assess_clarity(summary.summary)
        accuracy_score = self._assess_accuracy(summary.summary, original_content)
        completeness_score = self._assess_completeness(
            summary.summary, summary.key_points, original_content
        )
        legal_precision_score = self._assess_legal_precision(summary.summary)

        # Calculate overall score
        overall_score = (
            clarity_score + accuracy_score + completeness_score + legal_precision_score
        ) / 4

        # Identify issues
        issues = self.validate_summary_structure(summary.summary, summary.key_points)

        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            clarity_score, accuracy_score, completeness_score, legal_precision_score
        )

        # Determine if meets standards
        meets_standards = overall_score >= 7.0 and len(issues) == 0

        return SummaryQualityAssessment(
            summary_id=summary.article_id,
            clarity_score=clarity_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            legal_precision_score=legal_precision_score,
            overall_score=overall_score,
            meets_standards=meets_standards,
            issues_identified=issues,
            improvement_suggestions=suggestions,
        )

    def _assess_clarity(self, summary: str) -> float:
        """Assess clarity and readability of summary."""
        score = 8.0  # Start with high score

        # Check sentence length
        sentences = re.split(r"[.!?]+", summary)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(
            len([s for s in sentences if s.strip()]), 1
        )

        if avg_sentence_length > 25:  # Very long sentences
            score -= 1.0
        elif avg_sentence_length > 20:
            score -= 0.5

        # Check for complex punctuation patterns
        complex_punctuation = len(re.findall(r"[;:()]", summary))
        if complex_punctuation > 3:
            score -= 0.5

        # Check for passive voice (simple heuristic)
        passive_indicators = len(
            re.findall(r"\b(?:was|were|been|being)\s+\w+ed\b", summary, re.IGNORECASE)
        )
        if passive_indicators > 2:
            score -= 0.5

        return max(0.0, min(10.0, score))

    def _assess_accuracy(self, summary: str, original_content: str) -> float:
        """Assess factual accuracy of summary against original content."""
        # This is a simplified assessment - in practice, you'd use more sophisticated NLP
        score = 8.0

        # Check if key terms from original appear in summary
        original_words = set(original_content.lower().split())
        summary_words = set(summary.lower().split())

        # Legal terms should be preserved
        legal_terms = [
            "court",
            "judge",
            "ruling",
            "lawsuit",
            "case",
            "attorney",
            "plaintiff",
            "defendant",
            "verdict",
            "settlement",
            "appeal",
        ]

        original_legal_terms = {term for term in legal_terms if term in original_words}
        summary_legal_terms = {term for term in legal_terms if term in summary_words}

        if original_legal_terms:
            preservation_ratio = len(summary_legal_terms) / len(original_legal_terms)
            if preservation_ratio < 0.5:
                score -= 2.0
            elif preservation_ratio < 0.7:
                score -= 1.0

        return max(0.0, min(10.0, score))

    def _assess_completeness(
        self, summary: str, key_points: List[str], original_content: str
    ) -> float:
        """Assess completeness of summary coverage."""
        score = 7.0

        # Check if summary covers main topics
        summary_lower = summary.lower()

        # Key legal concepts that should be covered
        important_concepts = [
            "what happened",
            "legal issue",
            "parties involved",
            "outcome",
            "implications",
            "precedent",
            "ruling",
            "decision",
        ]

        # Simple check for coverage indicators
        coverage_indicators = 0
        for concept in important_concepts:
            # Check if concept words appear in summary
            concept_words = concept.split()
            if any(word in summary_lower for word in concept_words):
                coverage_indicators += 1

        coverage_ratio = coverage_indicators / len(important_concepts)
        if coverage_ratio >= 0.5:
            score += 1.0
        elif coverage_ratio >= 0.3:
            score += 0.5
        else:
            score -= 1.0

        # Check key points quality
        if len(key_points) >= 3:
            score += 0.5

        avg_key_point_length = sum(len(kp.split()) for kp in key_points) / max(
            len(key_points), 1
        )
        if avg_key_point_length >= 8:  # Substantial key points
            score += 0.5

        return max(0.0, min(10.0, score))

    def _assess_legal_precision(self, summary: str) -> float:
        """Assess legal precision and terminology usage."""
        score = 7.0

        # Check for proper legal terminology
        legal_terms = [
            "alleged",
            "plaintiff",
            "defendant",
            "court",
            "ruling",
            "jurisdiction",
            "statute",
            "regulation",
            "precedent",
            "appeal",
            "motion",
            "brief",
        ]

        summary_lower = summary.lower()
        legal_term_count = sum(1 for term in legal_terms if term in summary_lower)

        if legal_term_count >= 3:
            score += 1.0
        elif legal_term_count >= 1:
            score += 0.5
        else:
            score -= 1.0

        # Check for imprecise language
        imprecise_terms = ["maybe", "possibly", "might", "could be", "seems like"]
        imprecise_count = sum(1 for term in imprecise_terms if term in summary_lower)

        if imprecise_count > 2:
            score -= 1.0
        elif imprecise_count > 0:
            score -= 0.5

        return max(0.0, min(10.0, score))

    def _generate_improvement_suggestions(
        self,
        clarity_score: float,
        accuracy_score: float,
        completeness_score: float,
        legal_precision_score: float,
    ) -> List[str]:
        """Generate improvement suggestions based on scores."""
        suggestions = []

        if clarity_score < 7.0:
            suggestions.append(
                "Simplify sentence structure and reduce complex punctuation"
            )

        if accuracy_score < 7.0:
            suggestions.append(
                "Ensure key legal terms and facts from original article are preserved"
            )

        if completeness_score < 7.0:
            suggestions.append(
                "Include more comprehensive coverage of the legal issue and its implications"
            )

        if legal_precision_score < 7.0:
            suggestions.append(
                "Use more precise legal terminology and avoid speculative language"
            )

        return suggestions


class SummarizerAgent:
    """
    Summarization agent that creates professional legal article summaries.
    Implements LLM-powered summarization with quality validation and legal significance analysis.
    """

    def __init__(self):
        self.processor = SummaryProcessor()

    async def summarize_article(
        self, article_id: str, title: str, content: str, source_name: str = "Unknown Source"
    ) -> ArticleSummary:
        """
        Create a professional summary of a legal article.

        Args:
            article_id: Unique identifier for the article
            title: Article title
            content: Article content

        Returns:
            ArticleSummary object
        """
        try:
            # Prepare content for summarization
            full_content = f"Title: {title}\n\nContent: {content}"

            # Generate summary using LLM
            summary_response = await self._generate_summary_with_llm(full_content)

            # Extract structured data from response
            summary_text = summary_response.get("summary", "")
            key_points = summary_response.get("key_points", [])
            legal_significance = summary_response.get("legal_significance", "")

            # Determine urgency and impact
            urgency_level = self._determine_urgency(content, summary_response)
            impact_scope = self._determine_impact_scope(content, summary_response)

            # Create summary object
            summary = ArticleSummary(
                article_id=article_id,
                source_name=source_name,
                summary=summary_text,
                key_points=key_points,
                legal_significance=legal_significance,
                urgency_level=urgency_level,
                impact_scope=impact_scope,
                summarized_at=datetime.now(),
            )

            # Assess summary quality
            quality_assessment = self.processor.assess_summary_quality(summary, content)
            summary.summary_quality = quality_assessment.overall_score / 10.0
            summary.relevance_score = self._calculate_relevance_score(
                content, summary_text
            )

            # If quality is too low, attempt to improve
            if not quality_assessment.meets_standards:
                logger.warning(
                    f"Summary quality below standards for article {article_id}, attempting improvement"
                )
                improved_summary = await self._improve_summary(
                    summary, quality_assessment, full_content
                )
                if improved_summary:
                    summary = improved_summary

            logger.info(
                f"Summarization completed for article {article_id}: "
                f"quality={summary.summary_quality:.2f}, relevance={summary.relevance_score:.2f}"
            )

            return summary

        except Exception as e:
            logger.error(f"Error summarizing article {article_id}: {e}")
            raise

    async def _generate_summary_with_llm(self, content: str) -> Dict[str, Any]:
        """Generate summary using LLM."""
        try:
            prompt = LegalPrompts.get_summarization_prompt(content)

            # Request structured output
            structured_prompt = f"""
            {prompt}
            
            Please provide your response in the following JSON format:
            {{
                "summary": "Your 100-150 word summary here",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "legal_significance": "Explanation of legal significance and implications",
                "urgency_indicators": ["indicator1", "indicator2"],
                "impact_indicators": ["scope1", "scope2"]
            }}
            """

            response = await llm_client.generate_json(
                structured_prompt, max_tokens=800, temperature=0.5
            )
            return response

        except Exception as e:
            logger.warning(
                f"Structured LLM summarization failed: {e}, falling back to text generation"
            )

            # Fallback to text generation
            prompt = LegalPrompts.get_summarization_prompt(content)
            text_response = await llm_client.generate(
                prompt, max_tokens=400, temperature=0.5
            )

            if text_response.success:
                return {
                    "summary": text_response.content,
                    "key_points": [
                        "Key information extracted",
                        "Legal implications noted",
                    ],
                    "legal_significance": "Requires manual analysis",
                    "urgency_indicators": [],
                    "impact_indicators": [],
                }
            else:
                raise Exception(f"LLM summarization failed: {text_response.error}")

    async def _improve_summary(
        self,
        summary: ArticleSummary,
        assessment: SummaryQualityAssessment,
        original_content: str,
    ) -> Optional[ArticleSummary]:
        """Attempt to improve summary based on quality assessment."""
        try:
            improvement_prompt = f"""
            Please improve the following legal article summary based on these issues:
            
            Issues: {', '.join(assessment.issues_identified)}
            Suggestions: {', '.join(assessment.improvement_suggestions)}
            
            Original Summary: {summary.summary}
            
            Key Points: {summary.key_points}
            
            Requirements:
            - 100-150 words exactly
            - Professional legal tone
            - Address the identified issues
            - Maintain factual accuracy
            
            Provide improved version in JSON format:
            {{
                "summary": "improved summary",
                "key_points": ["improved", "key", "points"],
                "legal_significance": "improved legal significance"
            }}
            """

            improved_response = await llm_client.generate_json(
                improvement_prompt, max_tokens=600, temperature=0.3
            )

            # Create improved summary
            improved_summary = ArticleSummary(
                article_id=summary.article_id,
                source_name=summary.source_name,
                summary=improved_response.get("summary", summary.summary),
                key_points=improved_response.get("key_points", summary.key_points),
                legal_significance=improved_response.get(
                    "legal_significance", summary.legal_significance
                ),
                urgency_level=summary.urgency_level,
                impact_scope=summary.impact_scope,
            )

            # Re-assess quality
            new_assessment = self.processor.assess_summary_quality(
                improved_summary, original_content
            )

            if new_assessment.overall_score > assessment.overall_score:
                logger.info(
                    f"Summary improved: {assessment.overall_score:.1f} -> {new_assessment.overall_score:.1f}"
                )
                improved_summary.summary_quality = new_assessment.overall_score / 10.0
                return improved_summary
            else:
                logger.warning(
                    "Summary improvement attempt did not yield better quality"
                )
                return None

        except Exception as e:
            logger.error(f"Error improving summary: {e}")
            return None

    def _determine_urgency(self, content: str, summary_data: Dict[str, Any]) -> str:
        """Determine urgency level of the legal news."""
        content_lower = content.lower()
        urgency_indicators = summary_data.get("urgency_indicators", [])

        # High urgency indicators
        high_urgency_terms = [
            "breaking",
            "urgent",
            "emergency",
            "immediate",
            "injunction",
            "restraining order",
            "preliminary ruling",
            "emergency motion",
        ]

        # Medium urgency indicators
        medium_urgency_terms = [
            "ruling",
            "decision",
            "verdict",
            "settlement",
            "appeal filed",
            "motion granted",
            "court order",
        ]

        high_count = sum(1 for term in high_urgency_terms if term in content_lower)
        medium_count = sum(1 for term in medium_urgency_terms if term in content_lower)

        if high_count > 0 or any(
            "urgent" in indicator.lower() for indicator in urgency_indicators
        ):
            return "high"
        elif medium_count > 1:
            return "medium"
        else:
            return "low"

    def _determine_impact_scope(
        self, content: str, summary_data: Dict[str, Any]
    ) -> str:
        """Determine the scope of impact of the legal news."""
        content_lower = content.lower()
        impact_indicators = summary_data.get("impact_indicators", [])

        # International scope indicators
        if any(
            term in content_lower
            for term in ["international", "treaty", "foreign", "global"]
        ):
            return "international"

        # National scope indicators
        if any(
            term in content_lower
            for term in ["federal", "supreme court", "congress", "national"]
        ):
            return "national"

        # State scope indicators
        if any(
            term in content_lower
            for term in ["state", "governor", "legislature", "state court"]
        ):
            return "state"

        # Local scope indicators
        if any(
            term in content_lower for term in ["local", "city", "county", "municipal"]
        ):
            return "local"

        # Default to industry if specific scope not determined
        return "industry"

    def _calculate_relevance_score(self, content: str, summary: str) -> float:
        """Calculate relevance score for legal audience."""
        score = 0.5  # Base score

        # Legal terminology presence
        legal_terms = [
            "court",
            "judge",
            "attorney",
            "lawsuit",
            "ruling",
            "legal",
            "law",
            "case",
            "plaintiff",
            "defendant",
            "jurisdiction",
            "statute",
            "regulation",
        ]

        summary_lower = summary.lower()
        legal_term_count = sum(1 for term in legal_terms if term in summary_lower)

        # Increase score based on legal term density
        legal_density = legal_term_count / max(len(summary.split()), 1)
        score += min(0.3, legal_density * 10)  # Cap at 0.3

        # Professional legal topics boost relevance
        professional_topics = [
            "precedent",
            "litigation",
            "compliance",
            "regulatory",
            "contract",
            "intellectual property",
            "corporate law",
            "securities",
        ]

        professional_count = sum(
            1 for topic in professional_topics if topic in summary_lower
        )
        if professional_count > 0:
            score += 0.2

        return min(1.0, score)

    async def batch_summarize_articles(
        self, article_ids: List[str]
    ) -> List[ArticleSummary]:
        """Summarize multiple articles in batch."""
        summaries = []

        for article_id in article_ids:
            try:
                # Get specific article from database by ID
                article = await database_agent.get_article_by_id(article_id)

                if article:
                    summary = await self.summarize_article(
                        article_id, article["title"], article["content"], article["source_name"]
                    )
                    summaries.append(summary)

                    # Add small delay between summaries to avoid rate limiting
                    await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Failed to summarize article {article_id}: {e}")
                continue

        return summaries


# Global summarizer agent instance
summarizer_agent = SummarizerAgent()
