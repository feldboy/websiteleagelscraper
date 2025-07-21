"""
Content writing agent for generating professional legal blog articles.
Implements article generation with word count constraints and originality validation.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from pydantic import BaseModel, Field, validator

from tools.llm_client import llm_client
from config.prompts import LegalPrompts
from config.settings import settings
from agents.summarizer import ArticleSummary
from agents.connector import ConnectionAnalysis, TrendAnalysis


logger = logging.getLogger(__name__)


class GeneratedArticle(BaseModel):
    """A generated legal blog article."""

    article_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique article identifier",
    )
    title: str = Field(..., min_length=10, max_length=120, description="Article title")
    content: str = Field(
        ..., min_length=100, description="Article content"
    )

    # Metadata
    summary: str = Field(
        ..., min_length=50, max_length=200, description="Article summary"
    )
    tags: List[str] = Field(..., min_items=3, max_items=8, description="Article tags")
    category: str = Field("Legal News", description="Article category")

    # Quality metrics
    word_count: int = Field(0, description="Actual word count")
    originality_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Originality score"
    )
    quality_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall quality score"
    )
    readability_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Readability score"
    )

    # Source attribution
    source_summaries: List[str] = Field(
        default_factory=list, description="Source article summaries used"
    )
    connections_used: List[str] = Field(
        default_factory=list, description="Connections that informed the article"
    )
    trends_referenced: List[str] = Field(
        default_factory=list, description="Trends referenced in the article"
    )

    # Publishing metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="When article was created"
    )
    author: str = Field("Legal Research AI", description="Article author")
    status: str = Field("draft", description="Article status: draft, review, published")

    @validator("word_count", always=True)
    def calculate_word_count(cls, v, values):
        """Calculate word count from content."""
        if "content" in values:
            return len(values["content"].split())
        return v

    @validator("content")
    def validate_word_count_range(cls, v):
        """Ensure content meets word count requirements."""
        word_count = len(v.split())
        if word_count < 500:
            raise ValueError(f"Article too short: {word_count} words (minimum: 500)")
        if word_count > 650:
            raise ValueError(f"Article too long: {word_count} words (maximum: 650)")
        return v


class ArticleStructure(BaseModel):
    """Structure template for legal articles."""

    headline: str = Field(..., description="Compelling headline")
    introduction: str = Field(..., description="2-3 sentence introduction")
    main_sections: List[Dict[str, str]] = Field(
        ..., description="Main content sections with headers"
    )
    legal_implications: str = Field(..., description="Legal implications section")
    conclusion: str = Field(..., description="Forward-looking conclusion")

    def to_formatted_content(self) -> str:
        """Convert structure to formatted article content."""
        content_parts = []

        # Introduction
        content_parts.append(self.introduction)
        content_parts.append("")  # Empty line

        # Main sections
        for section in self.main_sections:
            header = section.get("header", "")
            content = section.get("content", "")
            if header:
                content_parts.append(f"## {header}")
            content_parts.append(content)
            content_parts.append("")  # Empty line

        # Legal implications
        if self.legal_implications:
            content_parts.append("## Legal Implications")
            content_parts.append(self.legal_implications)
            content_parts.append("")  # Empty line

        # Conclusion
        if self.conclusion:
            content_parts.append("## Looking Forward")
            content_parts.append(self.conclusion)

        return "\n".join(content_parts).strip()


class ContentQualityAnalyzer:
    """Analyzes content quality and readability."""

    def __init__(self):
        self.legal_terminology = [
            "court",
            "judge",
            "ruling",
            "decision",
            "case",
            "lawsuit",
            "legal",
            "attorney",
            "plaintiff",
            "defendant",
            "jurisdiction",
            "statute",
            "regulation",
            "compliance",
            "litigation",
            "precedent",
            "appeal",
            "motion",
            "brief",
            "verdict",
            "settlement",
            "contract",
            "law",
        ]

    def analyze_quality(self, article: GeneratedArticle) -> Dict[str, float]:
        """Analyze various quality metrics of the article."""
        content = article.content
        title = article.title

        # Calculate individual quality metrics
        originality = self._assess_originality(content)
        readability = self._assess_readability(content)
        legal_quality = self._assess_legal_quality(content)
        structure_quality = self._assess_structure_quality(content)
        title_quality = self._assess_title_quality(title, content)

        # Calculate overall quality
        overall_quality = (
            originality * 0.25
            + readability * 0.20
            + legal_quality * 0.25
            + structure_quality * 0.20
            + title_quality * 0.10
        )

        return {
            "originality": originality,
            "readability": readability,
            "legal_quality": legal_quality,
            "structure_quality": structure_quality,
            "title_quality": title_quality,
            "overall_quality": overall_quality,
        }

    def _assess_originality(self, content: str) -> float:
        """Assess content originality (simplified heuristic)."""
        # Check for repetitive phrases
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return 0.0

        # Check for sentence diversity
        unique_sentences = set(sentences)
        diversity_ratio = len(unique_sentences) / len(sentences)

        # Check for word diversity
        words = content.lower().split()
        unique_words = set(words)
        word_diversity = len(unique_words) / max(len(words), 1)

        # Check for common phrases that might indicate low originality
        common_phrases = [
            "according to reports",
            "it was reported",
            "sources say",
            "in a recent development",
            "breaking news",
            "stay tuned",
        ]

        common_phrase_count = sum(
            1 for phrase in common_phrases if phrase in content.lower()
        )
        phrase_penalty = min(0.3, common_phrase_count * 0.1)

        originality_score = (
            diversity_ratio * 0.4 + word_diversity * 0.6
        ) - phrase_penalty
        return max(0.0, min(1.0, originality_score))

    def _assess_readability(self, content: str) -> float:
        """Assess content readability."""
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) == 0:
            return 0.0

        words = content.split()

        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)

        # Readability score based on sentence length
        if avg_sentence_length <= 15:
            length_score = 1.0
        elif avg_sentence_length <= 20:
            length_score = 0.8
        elif avg_sentence_length <= 25:
            length_score = 0.6
        else:
            length_score = 0.4

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
        transition_score = min(1.0, transition_count / 5.0)

        # Check for complex punctuation
        complex_punct = len(re.findall(r"[;:()]", content))
        punct_penalty = min(0.2, complex_punct / 20.0)

        readability = (
            (length_score * 0.5 + transition_score * 0.3) - punct_penalty + 0.2
        )
        return max(0.0, min(1.0, readability))

    def _assess_legal_quality(self, content: str) -> float:
        """Assess quality of legal content."""
        content_lower = content.lower()

        # Count legal terminology
        legal_term_count = sum(
            1 for term in self.legal_terminology if term in content_lower
        )
        legal_density = legal_term_count / max(len(content.split()), 1)

        # Check for proper legal language patterns
        legal_patterns = [
            r"\bcourt\s+(?:ruled|decided|held|found)\b",
            r"\bjudge\s+\w+\s+(?:ruled|decided|stated)\b",
            r"\bplaintiff\s+(?:argued|claimed|alleged)\b",
            r"\bdefendant\s+(?:responded|argued|maintained)\b",
            r"\bthe\s+(?:case|lawsuit|litigation)\s+(?:involves|concerns|addresses)\b",
        ]

        pattern_matches = sum(
            1 for pattern in legal_patterns if re.search(pattern, content_lower)
        )
        pattern_score = min(1.0, pattern_matches / 3.0)

        # Check for specific legal concepts
        legal_concepts = [
            "precedent",
            "jurisdiction",
            "statute of limitations",
            "due process",
            "burden of proof",
            "reasonable doubt",
            "legal standing",
            "injunctive relief",
        ]

        concept_count = sum(1 for concept in legal_concepts if concept in content_lower)
        concept_score = min(1.0, concept_count / 2.0)

        # Combine scores
        legal_quality = (
            legal_density * 20 * 0.4 + pattern_score * 0.3 + concept_score * 0.3
        )
        return max(0.0, min(1.0, legal_quality))

    def _assess_structure_quality(self, content: str) -> float:
        """Assess structural quality of the article."""
        # Check for paragraph structure
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if len(paragraphs) < 3:
            structure_score = 0.3
        elif len(paragraphs) <= 6:
            structure_score = 1.0
        else:
            structure_score = 0.8

        # Check for headers (marked with ##)
        headers = re.findall(r"^##\s+.+$", content, re.MULTILINE)
        header_score = min(1.0, len(headers) / 3.0)

        # Check for logical flow indicators
        flow_indicators = [
            "first",
            "second",
            "finally",
            "in conclusion",
            "moreover",
            "furthermore",
            "in addition",
            "as a result",
            "consequently",
        ]

        flow_count = sum(
            1 for indicator in flow_indicators if indicator in content.lower()
        )
        flow_score = min(1.0, flow_count / 3.0)

        overall_structure = (
            structure_score * 0.5 + header_score * 0.3 + flow_score * 0.2
        )
        return max(0.0, min(1.0, overall_structure))

    def _assess_title_quality(self, title: str, content: str) -> float:
        """Assess quality of the article title."""
        if not title:
            return 0.0

        # Check title length
        title_words = len(title.split())
        if 6 <= title_words <= 12:
            length_score = 1.0
        elif 4 <= title_words <= 15:
            length_score = 0.8
        else:
            length_score = 0.5

        # Check for action words
        action_words = [
            "rules",
            "decides",
            "announces",
            "files",
            "wins",
            "loses",
            "settles",
            "appeals",
            "challenges",
            "approves",
            "rejects",
        ]

        action_score = (
            1.0 if any(word in title.lower() for word in action_words) else 0.5
        )

        # Check for legal terminology in title
        legal_in_title = any(
            term in title.lower() for term in self.legal_terminology[:10]
        )
        legal_score = 1.0 if legal_in_title else 0.7

        # Check title relevance to content
        title_words_set = set(title.lower().split())
        content_words_set = set(content.lower().split())
        relevance = len(title_words_set.intersection(content_words_set)) / len(
            title_words_set
        )
        relevance_score = min(1.0, relevance * 2)

        title_quality = (
            length_score * 0.3
            + action_score * 0.2
            + legal_score * 0.2
            + relevance_score * 0.3
        )
        return max(0.0, min(1.0, title_quality))


class WriterAgent:
    """
    Content writing agent that generates professional legal blog articles.
    Implements structured article generation with quality validation and originality checks.
    """

    def __init__(self):
        self.quality_analyzer = ContentQualityAnalyzer()
        self.min_quality_threshold = 0.7
        self.max_generation_attempts = 3

    async def write_article(
        self,
        summaries: List[ArticleSummary],
        connections: Optional[ConnectionAnalysis] = None,
        trends: Optional[List[TrendAnalysis]] = None,
    ) -> GeneratedArticle:
        """
        Generate a professional legal blog article from summaries and analysis.

        Args:
            summaries: List of article summaries to base the article on
            connections: Connection analysis between articles
            trends: Relevant legal trends

        Returns:
            GeneratedArticle object
        """
        try:
            if not summaries:
                raise ValueError("At least one summary is required to write an article")

            # Prepare context for article generation
            context = self._prepare_writing_context(summaries, connections, trends)

            # Generate article structure first
            structure = await self._generate_article_structure(context)

            # Generate full article content
            article = await self._generate_article_content(structure, context)

            # Analyze and improve quality if needed
            quality_metrics = self.quality_analyzer.analyze_quality(article)
            article.originality_score = quality_metrics["originality"]
            article.quality_score = quality_metrics["overall_quality"]
            article.readability_score = quality_metrics["readability"]

            # Attempt to improve if quality is below threshold
            if article.quality_score < self.min_quality_threshold:
                improved_article = await self._improve_article_quality(
                    article, context, quality_metrics
                )
                if improved_article:
                    article = improved_article

            # Set source attribution
            article.source_summaries = [summary.article_id for summary in summaries]
            if connections:
                article.connections_used = [
                    conn.description for conn in connections.direct_connections[:3]
                ]
            if trends:
                article.trends_referenced = [trend.trend_name for trend in trends[:2]]

            logger.info(
                f"Article generated: {article.word_count} words, "
                f"quality={article.quality_score:.2f}, originality={article.originality_score:.2f}"
            )

            return article

        except Exception as e:
            logger.error(f"Error generating article: {e}")
            raise

    def _prepare_writing_context(
        self,
        summaries: List[ArticleSummary],
        connections: Optional[ConnectionAnalysis],
        trends: Optional[List[TrendAnalysis]],
    ) -> Dict[str, Any]:
        """Prepare context information for article writing."""
        context = {
            "summaries": summaries,
            "summary_count": len(summaries),
            "main_topics": [],
            "key_entities": [],
            "legal_significance": [],
            "connections": [],
            "trends": [],
            "urgency_level": "medium",
            "impact_scope": "industry",
        }

        # Extract main topics and significance from summaries
        all_topics = []
        all_significance = []
        urgency_levels = []
        impact_scopes = []

        for summary in summaries:
            all_topics.extend(summary.key_points)
            all_significance.append(summary.legal_significance)
            urgency_levels.append(summary.urgency_level)
            impact_scopes.append(summary.impact_scope)

        context["main_topics"] = list(set(all_topics))[:5]  # Top 5 unique topics
        context["legal_significance"] = all_significance

        # Determine overall urgency and impact
        if "high" in urgency_levels:
            context["urgency_level"] = "high"
        elif "medium" in urgency_levels:
            context["urgency_level"] = "medium"

        if "national" in impact_scopes or "international" in impact_scopes:
            context["impact_scope"] = "national"

        # Add connection information
        if connections:
            context["connections"] = [
                {"type": conn.connection_type, "description": conn.description}
                for conn in connections.direct_connections[:3]
            ]

        # Add trend information
        if trends:
            context["trends"] = [
                {"name": trend.trend_name, "description": trend.trend_description}
                for trend in trends[:2]
            ]

        return context

    async def _generate_article_structure(
        self, context: Dict[str, Any]
    ) -> ArticleStructure:
        """Generate article structure using LLM."""
        try:
            # Create prompt for structure generation
            structure_prompt = f"""
            Create a structure for a legal blog article based on this information:
            
            Number of source articles: {context['summary_count']}
            Main topics: {', '.join(context['main_topics'])}
            Urgency level: {context['urgency_level']}
            Impact scope: {context['impact_scope']}
            
            Legal significance points:
            {chr(10).join(f"- {sig}" for sig in context['legal_significance'])}
            
            {'Connections found: ' + str(len(context['connections'])) if context['connections'] else ''}
            {'Trends identified: ' + str(len(context['trends'])) if context['trends'] else ''}
            
            Create a JSON structure with:
            {{
                "headline": "Compelling, professional headline (8-12 words)",
                "introduction": "2-3 sentence introduction setting context",
                "main_sections": [
                    {{"header": "Section Header", "content": "Section content outline"}},
                    {{"header": "Another Header", "content": "Another section outline"}}
                ],
                "legal_implications": "Legal implications section outline",
                "conclusion": "Forward-looking conclusion outline"
            }}
            
            Focus on creating a professional, informative structure for legal professionals.
            """

            structure_response = await llm_client.generate_json(
                structure_prompt, max_tokens=800, temperature=0.6
            )

            return ArticleStructure(**structure_response)

        except Exception as e:
            logger.warning(f"Error generating article structure: {e}")
            # Fallback to basic structure
            return ArticleStructure(
                headline="Legal Development Analysis",
                introduction="Recent legal developments have emerged that warrant professional attention and analysis.",
                main_sections=[
                    {
                        "header": "Key Developments",
                        "content": "Overview of main legal developments",
                    },
                    {
                        "header": "Analysis",
                        "content": "Detailed analysis of implications",
                    },
                ],
                legal_implications="Legal implications and potential impacts",
                conclusion="Summary and future outlook",
            )

    async def _generate_article_content(
        self, structure: ArticleStructure, context: Dict[str, Any]
    ) -> GeneratedArticle:
        """Generate full article content based on structure."""
        try:
            # Prepare comprehensive context for content generation
            summaries_text = "\n\n".join(
                [
                    f"Summary {i+1}: {summary.summary}\nKey Points: {', '.join(summary.key_points)}\nLegal Significance: {summary.legal_significance}"
                    for i, summary in enumerate(context["summaries"])
                ]
            )

            connections_text = (
                "\n".join(
                    [
                        f"Connection: {conn['description']}"
                        for conn in context["connections"]
                    ]
                )
                if context["connections"]
                else "No specific connections identified."
            )

            trends_text = (
                "\n".join(
                    [
                        f"Trend: {trend['name']} - {trend['description'][:100]}..."
                        for trend in context["trends"]
                    ]
                )
                if context["trends"]
                else "No specific trends identified."
            )

            # Generate content using LLM
            content_prompt = LegalPrompts.get_article_prompt(
                summaries=summaries_text,
                connections=connections_text,
                legal_significance="\n".join(context["legal_significance"]),
            )

            # Add specific structure requirements
            full_prompt = f"""
            {content_prompt}
            
            Use this structure as a guide:
            Headline: {structure.headline}
            Introduction: {structure.introduction}
            
            Main sections to cover:
            {chr(10).join(f"- {section['header']}: {section['content']}" for section in structure.main_sections)}
            
            Legal implications: {structure.legal_implications}
            Conclusion: {structure.conclusion}
            
            Requirements:
            - Exactly 500-600 words
            - Professional tone suitable for legal professionals
            - Include specific legal analysis and implications
            - Use proper legal terminology
            - Structure with clear sections using ## headers
            - Original content based on the provided summaries
            
            Format the response as a complete article with the headline as the title.
            """

            content_response = await llm_client.generate(
                full_prompt, max_tokens=1000, temperature=0.7
            )

            if not content_response.success:
                raise Exception(f"Content generation failed: {content_response.error}")

            # Parse the response to extract title and content
            article_text = content_response.content.strip()

            # Extract title (first line or first # header)
            lines = article_text.split("\n")
            title = structure.headline
            content_start_idx = 0

            if lines[0].startswith("#"):
                title = lines[0].lstrip("#").strip()
                content_start_idx = 1
            elif not lines[0].startswith("##") and len(lines[0].strip()) > 0:
                title = lines[0].strip()
                content_start_idx = 1

            # Extract content (everything after title)
            content = "\n".join(lines[content_start_idx:]).strip()

            # Generate summary and tags
            summary = await self._generate_article_summary(content)
            tags = self._generate_article_tags(content, context)

            return GeneratedArticle(
                title=title,
                content=content,
                summary=summary,
                tags=tags,
                category="Legal Analysis",
            )

        except Exception as e:
            logger.error(f"Error generating article content: {e}")
            raise

    async def _generate_article_summary(self, content: str) -> str:
        """Generate a brief summary of the article."""
        try:
            summary_prompt = f"""
            Create a brief 1-2 sentence summary of this legal article:
            
            {content[:500]}...
            
            The summary should capture the main legal development and its significance.
            Keep it professional and concise (50-100 words).
            """

            summary_response = await llm_client.generate(
                summary_prompt, max_tokens=150, temperature=0.5
            )

            if summary_response.success:
                return summary_response.content.strip()
            else:
                # Fallback summary
                return "Analysis of recent legal developments and their implications for the legal profession."

        except Exception as e:
            logger.warning(f"Error generating article summary: {e}")
            return "Legal analysis and professional insights on recent developments."

    def _generate_article_tags(
        self, content: str, context: Dict[str, Any]
    ) -> List[str]:
        """Generate relevant tags for the article."""
        tags = set()

        # Add tags based on legal topics mentioned
        legal_keywords = {
            "litigation": ["litigation", "lawsuit", "court case", "trial"],
            "corporate law": ["corporate", "business", "company", "corporation"],
            "intellectual property": ["patent", "trademark", "copyright", "ip"],
            "employment law": ["employment", "workplace", "employee", "labor"],
            "regulatory": ["regulation", "compliance", "regulatory", "agency"],
            "constitutional law": ["constitutional", "supreme court", "amendment"],
            "criminal law": ["criminal", "prosecution", "defendant", "guilty"],
            "contract law": ["contract", "agreement", "breach", "terms"],
        }

        content_lower = content.lower()

        for tag, keywords in legal_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.add(tag)

        # Add general legal tags
        if "court" in content_lower:
            tags.add("court ruling")
        if "settlement" in content_lower:
            tags.add("settlement")
        if "appeal" in content_lower:
            tags.add("appeal")
        if "judge" in content_lower:
            tags.add("judicial decision")

        # Add urgency/impact based tags
        if context["urgency_level"] == "high":
            tags.add("breaking news")
        if context["impact_scope"] == "national":
            tags.add("national impact")

        # Ensure we have at least 3 tags
        if len(tags) < 3:
            default_tags = ["legal news", "legal analysis", "professional insights"]
            tags.update(default_tags[: 3 - len(tags)])

        return list(tags)[:8]  # Limit to 8 tags

    async def _improve_article_quality(
        self,
        article: GeneratedArticle,
        context: Dict[str, Any],
        quality_metrics: Dict[str, float],
    ) -> Optional[GeneratedArticle]:
        """Attempt to improve article quality based on metrics."""
        try:
            # Identify specific issues
            issues = []
            if quality_metrics["readability"] < 0.7:
                issues.append(
                    "improve readability by using shorter sentences and simpler language"
                )
            if quality_metrics["legal_quality"] < 0.7:
                issues.append(
                    "include more legal terminology and specific legal analysis"
                )
            if quality_metrics["structure_quality"] < 0.7:
                issues.append(
                    "improve article structure with clearer sections and better flow"
                )
            if quality_metrics["originality"] < 0.7:
                issues.append("enhance originality by avoiding repetitive phrases")

            if not issues:
                return None  # No specific issues to fix

            improvement_prompt = f"""
            Please improve this legal article based on these specific issues:
            Issues to address: {'; '.join(issues)}
            
            Current article:
            Title: {article.title}
            Content: {article.content}
            
            Requirements:
            - Maintain 500-600 word count
            - Address the identified issues
            - Keep the professional legal tone
            - Maintain factual accuracy
            - Improve overall quality while preserving the core message
            
            Provide the improved article with the same structure (title and content).
            """

            improved_response = await llm_client.generate(
                improvement_prompt, max_tokens=1000, temperature=0.5
            )

            if improved_response.success:
                # Parse improved content
                improved_text = improved_response.content.strip()
                lines = improved_text.split("\n")

                improved_title = article.title
                improved_content = improved_text

                # Try to extract title if present
                if lines[0] and not lines[0].startswith("##"):
                    improved_title = lines[0].strip()
                    improved_content = "\n".join(lines[1:]).strip()

                # Create improved article
                improved_article = GeneratedArticle(
                    article_id=article.article_id,
                    title=improved_title,
                    content=improved_content,
                    summary=article.summary,
                    tags=article.tags,
                    category=article.category,
                    source_summaries=article.source_summaries,
                    connections_used=article.connections_used,
                    trends_referenced=article.trends_referenced,
                )

                # Check if improvement was successful
                new_quality_metrics = self.quality_analyzer.analyze_quality(
                    improved_article
                )
                if (
                    new_quality_metrics["overall_quality"]
                    > quality_metrics["overall_quality"]
                ):
                    logger.info(
                        f"Article quality improved: {quality_metrics['overall_quality']:.2f} -> {new_quality_metrics['overall_quality']:.2f}"
                    )
                    return improved_article

            return None

        except Exception as e:
            logger.error(f"Error improving article quality: {e}")
            return None


# Global writer agent instance
writer_agent = WriterAgent()
