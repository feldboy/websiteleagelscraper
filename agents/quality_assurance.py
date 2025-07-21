"""
Quality assurance agent for validating content before distribution.
Implements automated content validation, plagiarism detection, and quality standards enforcement.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import difflib

from pydantic import BaseModel, Field

from tools.llm_client import llm_client
from config.prompts import LegalPrompts
from config.settings import settings
from agents.writer import GeneratedArticle
from agents.database import database_agent


logger = logging.getLogger(__name__)


class QualityStandards(BaseModel):
    """Quality standards for legal content."""

    min_word_count: int = Field(500, description="Minimum word count")
    max_word_count: int = Field(650, description="Maximum word count")
    min_originality_score: float = Field(0.95, description="Minimum originality score")
    min_quality_score: float = Field(0.7, description="Minimum overall quality score")
    min_legal_terminology_density: float = Field(
        0.02, description="Minimum legal terminology density"
    )
    max_readability_complexity: float = Field(
        0.8, description="Maximum readability complexity"
    )

    # Content requirements
    required_sections: List[str] = Field(
        default=["introduction", "analysis", "implications"],
        description="Required article sections",
    )
    forbidden_phrases: List[str] = Field(
        default=["click here", "breaking news", "stay tuned", "sources say"],
        description="Phrases that indicate low-quality content",
    )

    # Legal-specific requirements
    requires_legal_citations: bool = Field(
        False, description="Whether legal citations are required"
    )
    requires_jurisdiction_mention: bool = Field(
        True, description="Whether jurisdiction must be mentioned"
    )
    min_legal_entities: int = Field(
        2, description="Minimum number of legal entities mentioned"
    )


class ValidationResult(BaseModel):
    """Result of content validation."""

    validation_id: str = Field(..., description="Unique validation identifier")
    article_id: str = Field(..., description="ID of validated article")

    # Overall assessment
    passed: bool = Field(..., description="Whether article passed validation")
    overall_score: float = Field(
        ..., ge=0.0, le=10.0, description="Overall quality score (0-10)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in assessment"
    )

    # Individual quality metrics
    word_count_compliant: bool = Field(
        ..., description="Word count within acceptable range"
    )
    originality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Originality score"
    )
    legal_quality_score: float = Field(
        ..., ge=0.0, le=10.0, description="Legal content quality"
    )
    readability_score: float = Field(
        ..., ge=0.0, le=10.0, description="Readability score"
    )
    factual_accuracy_score: float = Field(
        ..., ge=0.0, le=10.0, description="Factual accuracy score"
    )

    # Content analysis
    plagiarism_detected: bool = Field(
        False, description="Whether plagiarism was detected"
    )
    inappropriate_content: bool = Field(
        False, description="Whether inappropriate content found"
    )
    legal_terminology_adequate: bool = Field(
        ..., description="Adequate legal terminology usage"
    )
    structure_adequate: bool = Field(..., description="Adequate article structure")

    # Issues and recommendations
    critical_issues: List[str] = Field(
        default_factory=list, description="Critical issues found"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings and minor issues"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    # Metadata
    validated_at: datetime = Field(
        default_factory=datetime.now, description="When validation was performed"
    )
    validator_version: str = Field("1.0", description="Version of validation system")

    def is_publishable(self) -> bool:
        """Determine if article is ready for publication."""
        return (
            self.passed
            and not self.plagiarism_detected
            and not self.inappropriate_content
            and len(self.critical_issues) == 0
            and self.overall_score >= 7.0
        )


class PlagiarismDetector:
    """Detects potential plagiarism in generated content."""

    def __init__(self):
        self.similarity_threshold = 0.85
        self.min_matching_length = 20  # Minimum characters for similarity check

    async def check_plagiarism(
        self, content: str, title: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Check for plagiarism against known sources and databases.

        Args:
            content: Article content to check
            title: Article title

        Returns:
            Tuple of (is_plagiarized, similarity_score, matching_sources)
        """
        try:
            # Check against stored articles in database
            db_similarity = await self._check_database_similarity(content)

            # Check for common plagiarism patterns
            pattern_issues = self._check_plagiarism_patterns(content)

            # Calculate content fingerprint
            content_hash = self._calculate_content_fingerprint(content)

            # Check for exact duplicates
            exact_duplicate = await self._check_exact_duplicates(content_hash)

            # Combine results
            max_similarity = max(db_similarity["max_similarity"], 0.0)
            is_plagiarized = (
                exact_duplicate
                or max_similarity > self.similarity_threshold
                or len(pattern_issues) > 0
            )

            matching_sources = db_similarity.get("matching_sources", [])
            if exact_duplicate:
                matching_sources.append("Exact duplicate detected")
            matching_sources.extend(pattern_issues)

            logger.info(
                f"Plagiarism check completed: plagiarized={is_plagiarized}, similarity={max_similarity:.2f}"
            )

            return is_plagiarized, max_similarity, matching_sources

        except Exception as e:
            logger.error(f"Error in plagiarism detection: {e}")
            # Err on the side of caution
            return True, 1.0, [f"Plagiarism check failed: {str(e)}"]

    async def _check_database_similarity(self, content: str) -> Dict[str, Any]:
        """Check similarity against articles in database."""
        try:
            # Get recent articles for comparison
            recent_articles = await database_agent.get_unprocessed_articles(limit=100)

            max_similarity = 0.0
            matching_sources = []

            for article in recent_articles:
                stored_content = article.get("content", "")
                if len(stored_content) < self.min_matching_length:
                    continue

                # Calculate similarity using sequence matching
                similarity = self._calculate_text_similarity(content, stored_content)

                if similarity > max_similarity:
                    max_similarity = similarity

                if similarity > self.similarity_threshold:
                    matching_sources.append(
                        f"Similar to article: {article.get('title', 'Unknown')[:50]}..."
                    )

            return {
                "max_similarity": max_similarity,
                "matching_sources": matching_sources,
            }

        except Exception as e:
            logger.warning(f"Database similarity check failed: {e}")
            return {"max_similarity": 0.0, "matching_sources": []}

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Normalize texts
        text1_normalized = self._normalize_text(text1)
        text2_normalized = self._normalize_text(text2)

        # Use sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, text1_normalized, text2_normalized)
        return matcher.ratio()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r"\s+", " ", text.lower().strip())

        # Remove common filler words that don't affect meaning
        filler_words = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        ]
        words = normalized.split()
        filtered_words = [word for word in words if word not in filler_words]

        return " ".join(filtered_words)

    def _check_plagiarism_patterns(self, content: str) -> List[str]:
        """Check for common plagiarism patterns."""
        issues = []

        # Check for copied web content patterns
        web_patterns = [
            r"copyright\s+\d{4}",
            r"all\s+rights\s+reserved",
            r"terms\s+of\s+service",
            r"privacy\s+policy",
            r"this\s+website\s+uses\s+cookies",
        ]

        content_lower = content.lower()
        for pattern in web_patterns:
            if re.search(pattern, content_lower):
                issues.append(f"Web content pattern detected: {pattern}")

        # Check for academic paper patterns
        academic_patterns = [
            r"abstract:?\s*\n",
            r"keywords:?\s*\n",
            r"references?\s*\n",
            r"bibliography\s*\n",
            r"et\s+al\.",
            r"ibid\.",
            r"op\.\s*cit\.",
        ]

        for pattern in academic_patterns:
            if re.search(pattern, content_lower):
                issues.append(f"Academic content pattern detected: {pattern}")

        # Check for news article patterns
        news_patterns = [
            r"breaking:?\s*",
            r"developing\s+story",
            r"this\s+is\s+a\s+developing\s+story",
            r"more\s+to\s+follow",
            r"stay\s+tuned\s+for\s+updates",
        ]

        for pattern in news_patterns:
            if re.search(pattern, content_lower):
                issues.append(f"News content pattern detected: {pattern}")

        return issues

    def _calculate_content_fingerprint(self, content: str) -> str:
        """Calculate unique fingerprint for content."""
        # Create normalized content
        normalized = self._normalize_text(content)

        # Calculate SHA-256 hash
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def _check_exact_duplicates(self, content_hash: str) -> bool:
        """Check for exact duplicate content using content hash."""
        try:
            # In a real implementation, you'd check against a hash database
            # For now, return False as we don't have a persistent hash store
            return False
        except Exception as e:
            logger.warning(f"Exact duplicate check failed: {e}")
            return False


class ContentValidator:
    """Validates content against quality standards and legal requirements."""

    def __init__(self, standards: Optional[QualityStandards] = None):
        self.standards = standards or QualityStandards()
        self.legal_terminology = self._load_legal_terminology()

    def _load_legal_terminology(self) -> List[str]:
        """Load legal terminology for validation."""
        return [
            "court",
            "judge",
            "justice",
            "ruling",
            "decision",
            "case",
            "lawsuit",
            "litigation",
            "attorney",
            "lawyer",
            "counsel",
            "plaintiff",
            "defendant",
            "jurisdiction",
            "statute",
            "regulation",
            "law",
            "legal",
            "precedent",
            "appeal",
            "motion",
            "brief",
            "verdict",
            "settlement",
            "injunction",
            "contract",
            "agreement",
            "liability",
            "damages",
            "evidence",
            "testimony",
            "constitutional",
            "federal",
            "state",
            "municipal",
            "compliance",
            "violation",
            "penalty",
            "fine",
            "sentence",
            "conviction",
            "acquittal",
        ]

    def validate_article(self, article: GeneratedArticle) -> ValidationResult:
        """
        Validate an article against quality standards.

        Args:
            article: GeneratedArticle to validate

        Returns:
            ValidationResult with detailed validation information
        """
        validation_id = (
            f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{article.article_id[:8]}"
        )

        # Individual validation checks
        word_count_check = self._validate_word_count(article.content)
        legal_quality = self._validate_legal_quality(article.content)
        readability = self._validate_readability(article.content)
        structure = self._validate_structure(article.content)
        content_appropriateness = self._validate_content_appropriateness(
            article.content
        )

        # Collect issues and warnings
        critical_issues = []
        warnings = []
        recommendations = []

        # Word count validation
        if not word_count_check["compliant"]:
            critical_issues.append(word_count_check["issue"])

        # Legal quality validation
        if legal_quality["score"] < 6.0:
            critical_issues.append("Insufficient legal content quality")
        elif legal_quality["score"] < 7.0:
            warnings.append("Legal content quality could be improved")

        if not legal_quality["adequate_terminology"]:
            critical_issues.append("Inadequate legal terminology usage")

        # Readability validation
        if readability["score"] < 5.0:
            critical_issues.append("Poor readability - content too complex")
        elif readability["score"] < 7.0:
            warnings.append("Readability could be improved")

        # Structure validation
        if not structure["adequate"]:
            if structure["severity"] == "critical":
                critical_issues.append("Inadequate article structure")
            else:
                warnings.append("Article structure could be improved")

        # Content appropriateness
        if content_appropriateness["inappropriate"]:
            critical_issues.append("Inappropriate content detected")

        # Generate recommendations
        recommendations.extend(legal_quality.get("recommendations", []))
        recommendations.extend(readability.get("recommendations", []))
        recommendations.extend(structure.get("recommendations", []))

        # Calculate overall scores
        originality_score = (
            article.originality_score if hasattr(article, "originality_score") else 0.8
        )
        overall_score = self._calculate_overall_score(
            word_count_check, legal_quality, readability, structure, originality_score
        )

        # Determine pass/fail
        passed = (
            len(critical_issues) == 0
            and overall_score >= 7.0
            and originality_score >= self.standards.min_originality_score
        )

        return ValidationResult(
            validation_id=validation_id,
            article_id=article.article_id,
            passed=passed,
            overall_score=overall_score,
            confidence=0.85,  # Static confidence for now
            word_count_compliant=word_count_check["compliant"],
            originality_score=originality_score,
            legal_quality_score=legal_quality["score"],
            readability_score=readability["score"],
            factual_accuracy_score=8.0,  # Placeholder - would need fact-checking integration
            legal_terminology_adequate=legal_quality["adequate_terminology"],
            structure_adequate=structure["adequate"],
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _validate_word_count(self, content: str) -> Dict[str, Any]:
        """Validate article word count."""
        word_count = len(content.split())

        compliant = (
            self.standards.min_word_count <= word_count <= self.standards.max_word_count
        )

        issue = None
        if word_count < self.standards.min_word_count:
            issue = f"Article too short: {word_count} words (minimum: {self.standards.min_word_count})"
        elif word_count > self.standards.max_word_count:
            issue = f"Article too long: {word_count} words (maximum: {self.standards.max_word_count})"

        return {"compliant": compliant, "word_count": word_count, "issue": issue}

    def _validate_legal_quality(self, content: str) -> Dict[str, Any]:
        """Validate legal content quality."""
        content_lower = content.lower()
        words = content.split()

        # Count legal terminology
        legal_term_count = sum(
            1 for term in self.legal_terminology if term in content_lower
        )
        legal_density = legal_term_count / len(words) if words else 0

        # Check for legal patterns
        legal_patterns = [
            r"\bcourt\s+(?:ruled|decided|held|found)\b",
            r"\bjudge\s+\w+\s+(?:ruled|decided|stated)\b",
            r"\bplaintiff\s+(?:argued|claimed|alleged)\b",
            r"\bdefendant\s+(?:responded|argued|maintained)\b",
            r"\bthe\s+(?:case|lawsuit|litigation)\s+(?:involves|concerns|addresses)\b",
        ]

        pattern_count = sum(
            1 for pattern in legal_patterns if re.search(pattern, content_lower)
        )

        # Check for legal citations (simplified)
        citation_patterns = [
            r"\d+\s+\w+\s+\d+",  # Case citations
            r"\b\d+\s+U\.?S\.?\s+\d+",  # US Supreme Court
            r"\b\d+\s+F\.\d*d?\s+\d+",  # Federal reporters
        ]

        citation_count = sum(
            1 for pattern in citation_patterns if re.search(pattern, content)
        )

        # Calculate score
        base_score = 5.0

        # Legal terminology score
        if legal_density >= self.standards.min_legal_terminology_density:
            base_score += 2.0
        elif legal_density >= self.standards.min_legal_terminology_density * 0.7:
            base_score += 1.0

        # Legal patterns score
        if pattern_count >= 3:
            base_score += 2.0
        elif pattern_count >= 1:
            base_score += 1.0

        # Citations bonus
        if citation_count > 0:
            base_score += 1.0

        # Check for jurisdiction mention
        jurisdiction_mentioned = any(
            term in content_lower
            for term in ["federal", "state", "local", "court", "jurisdiction"]
        )

        if jurisdiction_mentioned:
            base_score += 0.5

        score = min(10.0, base_score)
        adequate_terminology = (
            legal_density >= self.standards.min_legal_terminology_density
        )

        recommendations = []
        if legal_density < self.standards.min_legal_terminology_density:
            recommendations.append("Include more legal terminology and concepts")
        if pattern_count < 2:
            recommendations.append(
                "Add more specific legal analysis and case discussion"
            )
        if not jurisdiction_mentioned and self.standards.requires_jurisdiction_mention:
            recommendations.append("Mention the relevant jurisdiction")

        return {
            "score": score,
            "legal_density": legal_density,
            "pattern_count": pattern_count,
            "citation_count": citation_count,
            "adequate_terminology": adequate_terminology,
            "recommendations": recommendations,
        }

    def _validate_readability(self, content: str) -> Dict[str, Any]:
        """Validate content readability."""
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {"score": 0.0, "recommendations": ["Content appears to be empty"]}

        words = content.split()

        # Calculate readability metrics
        avg_sentence_length = len(words) / len(sentences)

        # Count complex words (more than 2 syllables - simplified)
        complex_words = [word for word in words if len(word) > 8]
        complex_word_ratio = len(complex_words) / len(words) if words else 0

        # Check for transition words
        transition_words = [
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "additionally",
            "consequently",
            "meanwhile",
            "nevertheless",
            "nonetheless",
        ]
        transition_count = sum(
            1 for word in transition_words if word in content.lower()
        )

        # Calculate score
        base_score = 7.0

        # Sentence length penalty
        if avg_sentence_length > 25:
            base_score -= 2.0
        elif avg_sentence_length > 20:
            base_score -= 1.0

        # Complex word penalty
        if complex_word_ratio > 0.3:
            base_score -= 2.0
        elif complex_word_ratio > 0.2:
            base_score -= 1.0

        # Transition words bonus
        if transition_count >= 3:
            base_score += 1.0
        elif transition_count >= 1:
            base_score += 0.5

        score = max(0.0, min(10.0, base_score))

        recommendations = []
        if avg_sentence_length > 20:
            recommendations.append("Use shorter sentences for better readability")
        if complex_word_ratio > 0.2:
            recommendations.append("Simplify complex words where possible")
        if transition_count < 2:
            recommendations.append("Add transition words to improve flow")

        return {
            "score": score,
            "avg_sentence_length": avg_sentence_length,
            "complex_word_ratio": complex_word_ratio,
            "transition_count": transition_count,
            "recommendations": recommendations,
        }

    def _validate_structure(self, content: str) -> Dict[str, Any]:
        """Validate article structure."""
        # Check for headers
        headers = re.findall(r"^##\s+.+$", content, re.MULTILINE)

        # Check for paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # Check for introduction patterns
        intro_patterns = [
            r"^[A-Z][^.!?]*(?:recent|new|latest|breaking|announced|reported)",
            r"^In\s+a\s+(?:recent|new|significant|major)",
            r"^(?:A|The)\s+(?:court|judge|jury|legal)",
        ]

        has_intro = any(
            re.search(pattern, content, re.IGNORECASE) for pattern in intro_patterns
        )

        # Check for conclusion patterns
        conclusion_patterns = [
            r"(?:in\s+conclusion|to\s+conclude|finally|ultimately|going\s+forward)",
            r"(?:the\s+(?:implications|impact|significance)|this\s+(?:decision|ruling|case))",
            r"(?:legal\s+(?:experts|professionals)|practitioners?\s+should)",
        ]

        has_conclusion = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in conclusion_patterns
        )

        # Determine adequacy
        adequate = True
        severity = "minor"
        recommendations = []

        if len(headers) < 2:
            adequate = False
            severity = "major"
            recommendations.append("Add section headers to improve structure")

        if len(paragraphs) < 3:
            adequate = False
            severity = "critical"
            recommendations.append("Break content into more paragraphs")

        if not has_intro:
            severity = "major"
            recommendations.append("Add a clear introduction")

        if not has_conclusion:
            recommendations.append("Add a conclusion section")

        return {
            "adequate": adequate,
            "severity": severity,
            "header_count": len(headers),
            "paragraph_count": len(paragraphs),
            "has_intro": has_intro,
            "has_conclusion": has_conclusion,
            "recommendations": recommendations,
        }

    def _validate_content_appropriateness(self, content: str) -> Dict[str, Any]:
        """Validate content appropriateness."""
        content_lower = content.lower()

        # Check for forbidden phrases
        forbidden_found = []
        for phrase in self.standards.forbidden_phrases:
            if phrase.lower() in content_lower:
                forbidden_found.append(phrase)

        # Check for promotional content
        promotional_patterns = [
            r"click\s+here",
            r"call\s+now",
            r"limited\s+time",
            r"act\s+fast",
            r"don\'t\s+miss",
            r"special\s+offer",
        ]

        promotional_found = []
        for pattern in promotional_patterns:
            if re.search(pattern, content_lower):
                promotional_found.append(pattern)

        inappropriate = len(forbidden_found) > 0 or len(promotional_found) > 0

        return {
            "inappropriate": inappropriate,
            "forbidden_phrases": forbidden_found,
            "promotional_content": promotional_found,
        }

    def _calculate_overall_score(
        self,
        word_count: Dict[str, Any],
        legal_quality: Dict[str, Any],
        readability: Dict[str, Any],
        structure: Dict[str, Any],
        originality_score: float,
    ) -> float:
        """Calculate overall quality score."""
        # Weight different components
        weights = {
            "word_count": 0.1,
            "legal_quality": 0.3,
            "readability": 0.2,
            "structure": 0.2,
            "originality": 0.2,
        }

        # Component scores
        word_count_score = 10.0 if word_count["compliant"] else 0.0
        legal_score = legal_quality["score"]
        readability_score = readability["score"]
        structure_score = 8.0 if structure["adequate"] else 5.0
        originality_score_scaled = originality_score * 10.0

        # Calculate weighted average
        overall = (
            word_count_score * weights["word_count"]
            + legal_score * weights["legal_quality"]
            + readability_score * weights["readability"]
            + structure_score * weights["structure"]
            + originality_score_scaled * weights["originality"]
        )

        return min(10.0, max(0.0, overall))


class QualityAssuranceAgent:
    """
    Quality assurance agent that validates content before distribution.
    Implements comprehensive validation including plagiarism detection and quality standards.
    """

    def __init__(self, standards: Optional[QualityStandards] = None):
        self.standards = standards or QualityStandards()
        self.plagiarism_detector = PlagiarismDetector()
        self.content_validator = ContentValidator(self.standards)

    async def validate_article(self, article: GeneratedArticle) -> ValidationResult:
        """
        Perform comprehensive validation of a generated article.

        Args:
            article: GeneratedArticle to validate

        Returns:
            ValidationResult with detailed validation information
        """
        try:
            logger.info(
                f"Starting quality assurance validation for article {article.article_id}"
            )

            # Basic content validation
            validation_result = self.content_validator.validate_article(article)

            # Plagiarism detection
            is_plagiarized, similarity_score, matching_sources = (
                await self.plagiarism_detector.check_plagiarism(
                    article.content, article.title
                )
            )

            # Update validation result with plagiarism information
            validation_result.plagiarism_detected = is_plagiarized
            validation_result.originality_score = 1.0 - similarity_score

            if is_plagiarized:
                validation_result.critical_issues.append("Plagiarism detected")
                validation_result.critical_issues.extend(
                    matching_sources[:3]
                )  # Limit to 3 sources
                validation_result.passed = False

            # LLM-enhanced quality assessment
            llm_assessment = await self._get_llm_quality_assessment(article)
            if llm_assessment:
                validation_result = self._merge_llm_assessment(
                    validation_result, llm_assessment
                )

            # Final pass/fail determination
            validation_result.passed = validation_result.is_publishable()

            logger.info(
                f"Quality assurance completed for article {article.article_id}: "
                f"passed={validation_result.passed}, score={validation_result.overall_score:.1f}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Error in quality assurance validation: {e}")
            # Return failed validation on error
            return ValidationResult(
                validation_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                article_id=article.article_id,
                passed=False,
                overall_score=0.0,
                confidence=0.0,
                word_count_compliant=False,
                originality_score=0.0,
                legal_quality_score=0.0,
                readability_score=0.0,
                factual_accuracy_score=0.0,
                legal_terminology_adequate=False,
                structure_adequate=False,
                critical_issues=[f"Validation error: {str(e)}"],
            )

    async def _get_llm_quality_assessment(
        self, article: GeneratedArticle
    ) -> Optional[Dict[str, Any]]:
        """Get quality assessment from LLM."""
        try:
            prompt = LegalPrompts.get_quality_prompt(
                f"Title: {article.title}\n\nContent: {article.content}"
            )

            response = await llm_client.generate_json(
                prompt, max_tokens=600, temperature=0.3
            )

            return response

        except Exception as e:
            logger.warning(f"LLM quality assessment failed: {e}")
            return None

    def _merge_llm_assessment(
        self, validation_result: ValidationResult, llm_assessment: Dict[str, Any]
    ) -> ValidationResult:
        """Merge LLM assessment with validation result."""
        try:
            # Update scores with LLM input
            llm_overall = llm_assessment.get(
                "overall_score", validation_result.overall_score
            )
            validation_result.overall_score = (
                validation_result.overall_score + llm_overall
            ) / 2

            # Add LLM-identified issues
            llm_issues = llm_assessment.get("issues_found", [])
            validation_result.warnings.extend(
                llm_issues[:3]
            )  # Limit to 3 additional issues

            # Add LLM recommendations
            llm_recommendations = llm_assessment.get("recommendations", [])
            validation_result.recommendations.extend(llm_recommendations[:3])

            # Update confidence if LLM provides one
            llm_confidence = llm_assessment.get("confidence_score")
            if llm_confidence:
                validation_result.confidence = (
                    validation_result.confidence + llm_confidence
                ) / 2

        except Exception as e:
            logger.warning(f"Error merging LLM assessment: {e}")

        return validation_result

    async def batch_validate_articles(
        self, articles: List[GeneratedArticle]
    ) -> List[ValidationResult]:
        """Validate multiple articles in batch."""
        results = []

        for article in articles:
            try:
                result = await self.validate_article(article)
                results.append(result)

                # Add delay to avoid overwhelming the system
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to validate article {article.article_id}: {e}")
                continue

        return results

    def get_validation_statistics(
        self, validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """Generate statistics from validation results."""
        if not validation_results:
            return {}

        total_articles = len(validation_results)
        passed_articles = sum(1 for result in validation_results if result.passed)
        plagiarism_detected = sum(
            1 for result in validation_results if result.plagiarism_detected
        )

        avg_overall_score = (
            sum(result.overall_score for result in validation_results) / total_articles
        )
        avg_originality = (
            sum(result.originality_score for result in validation_results)
            / total_articles
        )

        common_issues = {}
        for result in validation_results:
            for issue in result.critical_issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1

        return {
            "total_articles": total_articles,
            "passed_articles": passed_articles,
            "pass_rate": passed_articles / total_articles if total_articles > 0 else 0,
            "plagiarism_detected": plagiarism_detected,
            "plagiarism_rate": (
                plagiarism_detected / total_articles if total_articles > 0 else 0
            ),
            "average_overall_score": avg_overall_score,
            "average_originality_score": avg_originality,
            "common_issues": sorted(
                common_issues.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


# Global quality assurance agent instance
quality_assurance_agent = QualityAssuranceAgent()
