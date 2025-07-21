"""
Tests for the quality assurance agent.
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime

from agents.quality_assurance import (
    QualityAssuranceAgent,
    ContentValidator,
    PlagiarismDetector,
    QualityStandards,
    ValidationResult,
)
from agents.writer import GeneratedArticle


class TestContentValidator:
    """Test the ContentValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a ContentValidator instance."""
        return ContentValidator()

    @pytest.fixture
    def high_quality_article(self) -> GeneratedArticle:
        """Create a high-quality article for testing."""
        return GeneratedArticle(
            title="Federal Court Issues Landmark Ruling on Contract Law",
            content="""
            ## Federal Court Clarifies Contract Interpretation Standards

            In a significant development for contract law, the Federal District Court 
            for the Southern District of New York issued a comprehensive ruling in 
            Johnson v. Corporate Services Inc. that establishes new precedent for 
            contract interpretation in commercial disputes. The court's decision, 
            authored by Judge Patricia Williams, addresses longstanding ambiguities 
            in how courts should approach ambiguous contract terms.

            ## Legal Analysis and Implications

            The ruling clarifies that courts must first examine the plain language 
            of contract terms before considering extrinsic evidence. This approach 
            aligns with established precedent while providing clearer guidance for 
            practitioners. The court emphasized that contract interpretation should 
            prioritize the parties' objective intent as expressed in the written 
            agreement.

            ## Impact on Commercial Litigation

            Legal experts anticipate this decision will streamline contract disputes 
            and provide greater predictability for businesses drafting commercial 
            agreements. The American Bar Association's Business Law Section has 
            already begun incorporating these principles into practice guidelines.

            ## Future Considerations

            This ruling establishes important precedent that will likely influence 
            similar cases in other federal circuits. Practitioners should review 
            existing contract templates and litigation strategies in light of these 
            new interpretation standards.
            """,
            summary="Federal court establishes new precedent for contract interpretation in commercial disputes.",
            tags=[
                "contract law",
                "federal court",
                "commercial litigation",
                "legal precedent",
            ],
            originality_score=0.95,
            quality_score=0.88,
        )

    @pytest.fixture
    def low_quality_article(self) -> GeneratedArticle:
        """Create a low-quality article for testing."""
        return GeneratedArticle(
            title="Short Article",
            content="This is a very short article that doesn't meet quality standards. It lacks legal terminology and proper structure.",
            summary="Short summary",
            tags=["test"],
            originality_score=0.6,
            quality_score=0.3,
        )

    def test_validate_high_quality_article(self, validator, high_quality_article):
        """Test validation of a high-quality article."""
        result = validator.validate_article(high_quality_article)

        assert result.passed
        assert result.word_count_compliant
        assert result.legal_terminology_adequate
        assert result.structure_adequate
        assert result.overall_score >= 7.0
        assert len(result.critical_issues) == 0

    def test_validate_low_quality_article(self, validator, low_quality_article):
        """Test validation of a low-quality article."""
        result = validator.validate_article(low_quality_article)

        assert not result.passed
        assert not result.word_count_compliant  # Too short
        assert result.overall_score < 7.0
        assert len(result.critical_issues) > 0
        assert any("too short" in issue.lower() for issue in result.critical_issues)

    def test_validate_word_count_compliance(self, validator):
        """Test word count validation."""
        # Test compliant word count
        word_count_check = validator._validate_word_count("word " * 550)  # 550 words
        assert word_count_check["compliant"]
        assert word_count_check["word_count"] == 550
        assert word_count_check["issue"] is None

        # Test too short
        word_count_check = validator._validate_word_count("word " * 400)  # 400 words
        assert not word_count_check["compliant"]
        assert "too short" in word_count_check["issue"]

        # Test too long
        word_count_check = validator._validate_word_count("word " * 700)  # 700 words
        assert not word_count_check["compliant"]
        assert "too long" in word_count_check["issue"]

    def test_validate_legal_quality(self, validator):
        """Test legal content quality validation."""
        legal_content = """
        The court ruled that the plaintiff's motion for summary judgment should be granted.
        The judge found that the defendant violated the contract terms and ordered damages.
        This case establishes important precedent for similar litigation.
        """

        result = validator._validate_legal_quality(legal_content)

        assert result["score"] > 6.0  # Should have decent legal quality
        assert result["adequate_terminology"]
        assert result["pattern_count"] > 0  # Should find legal patterns

    def test_validate_readability(self, validator):
        """Test readability validation."""
        readable_content = """
        This is a readable article with short sentences. The content flows well.
        However, there are important legal implications to consider. Therefore,
        practitioners should review their current strategies.
        """

        result = validator._validate_readability(readable_content)

        assert result["score"] >= 6.0
        assert result["avg_sentence_length"] < 30
        assert result["transition_count"] > 0

    def test_validate_structure(self, validator):
        """Test structure validation."""
        well_structured_content = """
        ## Introduction
        
        This article provides important analysis.
        
        ## Legal Analysis
        
        The court's decision establishes precedent.
        
        ## Conclusion
        
        Going forward, practitioners should consider these implications.
        """

        result = validator._validate_structure(well_structured_content)

        assert result["adequate"]
        assert result["header_count"] >= 2
        assert result["has_intro"]
        assert result["has_conclusion"]


class TestPlagiarismDetector:
    """Test the PlagiarismDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a PlagiarismDetector instance."""
        return PlagiarismDetector()

    @pytest.mark.asyncio
    async def test_check_plagiarism_original_content(self, detector):
        """Test plagiarism detection with original content."""
        original_content = """
        This is completely original legal analysis that has never been published before.
        The unique perspective presented here is based on novel interpretation of precedent.
        """

        # Mock database check to return no matches
        with patch.object(detector, "_check_database_similarity") as mock_db_check:
            mock_db_check.return_value = {"max_similarity": 0.2, "matching_sources": []}

            with patch.object(detector, "_check_exact_duplicates") as mock_exact_check:
                mock_exact_check.return_value = False

                is_plagiarized, similarity, sources = await detector.check_plagiarism(
                    original_content, "Original Legal Analysis"
                )

                assert not is_plagiarized
                assert similarity < detector.similarity_threshold
                assert len(sources) == 0

    @pytest.mark.asyncio
    async def test_check_plagiarism_high_similarity(self, detector):
        """Test plagiarism detection with high similarity content."""
        similar_content = "This content is very similar to existing articles."

        # Mock database check to return high similarity
        with patch.object(detector, "_check_database_similarity") as mock_db_check:
            mock_db_check.return_value = {
                "max_similarity": 0.9,  # Above threshold
                "matching_sources": ["Similar article found"],
            }

            with patch.object(detector, "_check_exact_duplicates") as mock_exact_check:
                mock_exact_check.return_value = False

                is_plagiarized, similarity, sources = await detector.check_plagiarism(
                    similar_content, "Similar Content"
                )

                assert is_plagiarized
                assert similarity > detector.similarity_threshold
                assert len(sources) > 0

    def test_check_plagiarism_patterns(self, detector):
        """Test detection of plagiarism patterns."""
        web_content = "Copyright 2024 All rights reserved. This website uses cookies."
        patterns = detector._check_plagiarism_patterns(web_content)

        assert len(patterns) > 0
        assert any("web content pattern" in pattern.lower() for pattern in patterns)

        academic_content = "Abstract: This paper presents... Keywords: legal analysis"
        patterns = detector._check_plagiarism_patterns(academic_content)

        assert len(patterns) > 0
        assert any(
            "academic content pattern" in pattern.lower() for pattern in patterns
        )

    def test_calculate_text_similarity(self, detector):
        """Test text similarity calculation."""
        text1 = "The court ruled in favor of the plaintiff."
        text2 = "The court ruled in favor of the plaintiff."  # Identical

        similarity = detector._calculate_text_similarity(text1, text2)
        assert similarity > 0.9  # Should be very high

        text3 = "The defendant won the case."  # Different
        similarity = detector._calculate_text_similarity(text1, text3)
        assert similarity < 0.5  # Should be low


class TestQualityAssuranceAgent:
    """Test the QualityAssuranceAgent class."""

    @pytest.fixture
    def qa_agent(self):
        """Create a QualityAssuranceAgent instance."""
        return QualityAssuranceAgent()

    @pytest.mark.asyncio
    async def test_validate_article_success(self, qa_agent, sample_generated_article):
        """Test successful article validation."""
        # Mock plagiarism detector to return no plagiarism
        with patch.object(
            qa_agent.plagiarism_detector, "check_plagiarism"
        ) as mock_plagiarism:
            mock_plagiarism.return_value = (False, 0.1, [])

            # Mock LLM assessment
            with patch.object(qa_agent, "_get_llm_quality_assessment") as mock_llm:
                mock_llm.return_value = {
                    "overall_score": 8.5,
                    "issues_found": [],
                    "recommendations": ["Great work!"],
                }

                result = await qa_agent.validate_article(sample_generated_article)

                assert isinstance(result, ValidationResult)
                assert result.passed
                assert not result.plagiarism_detected
                assert result.overall_score >= 7.0
                assert result.is_publishable()

    @pytest.mark.asyncio
    async def test_validate_article_plagiarism_detected(
        self, qa_agent, sample_generated_article
    ):
        """Test article validation with plagiarism detected."""
        # Mock plagiarism detector to return plagiarism
        with patch.object(
            qa_agent.plagiarism_detector, "check_plagiarism"
        ) as mock_plagiarism:
            mock_plagiarism.return_value = (True, 0.9, ["Similar source found"])

            result = await qa_agent.validate_article(sample_generated_article)

            assert not result.passed
            assert result.plagiarism_detected
            assert not result.is_publishable()
            assert any(
                "plagiarism" in issue.lower() for issue in result.critical_issues
            )

    @pytest.mark.asyncio
    async def test_validate_article_error_handling(
        self, qa_agent, sample_generated_article
    ):
        """Test article validation error handling."""
        # Mock plagiarism detector to raise exception
        with patch.object(
            qa_agent.plagiarism_detector, "check_plagiarism"
        ) as mock_plagiarism:
            mock_plagiarism.side_effect = Exception("Test error")

            result = await qa_agent.validate_article(sample_generated_article)

            assert not result.passed
            assert result.overall_score == 0.0
            assert any(
                "validation error" in issue.lower() for issue in result.critical_issues
            )

    @pytest.mark.asyncio
    async def test_batch_validate_articles(self, qa_agent, sample_generated_article):
        """Test batch validation of multiple articles."""
        articles = [
            sample_generated_article,
            sample_generated_article,
        ]  # Use same article twice

        # Mock plagiarism detector
        with patch.object(
            qa_agent.plagiarism_detector, "check_plagiarism"
        ) as mock_plagiarism:
            mock_plagiarism.return_value = (False, 0.1, [])

            results = await qa_agent.batch_validate_articles(articles)

            assert len(results) == 2
            assert all(isinstance(result, ValidationResult) for result in results)

    def test_quality_standards_validation(self):
        """Test quality standards configuration."""
        standards = QualityStandards(
            min_word_count=400, max_word_count=700, min_originality_score=0.9
        )

        assert standards.min_word_count == 400
        assert standards.max_word_count == 700
        assert standards.min_originality_score == 0.9
        assert len(standards.forbidden_phrases) > 0
