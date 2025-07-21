"""
Connection analysis agent for identifying relationships between legal articles.
Implements cross-article analysis, thematic connection identification, and trend analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict, Counter
import uuid

from pydantic import BaseModel, Field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tools.llm_client import llm_client
from config.prompts import LegalPrompts
from agents.database import database_agent
from agents.summarizer import ArticleSummary


logger = logging.getLogger(__name__)


class ArticleConnection(BaseModel):
    """Represents a connection between two legal articles."""

    article_id_1: str = Field(..., description="ID of the first article")
    article_id_2: str = Field(..., description="ID of the second article")
    connection_type: str = Field(..., description="Type of connection")
    strength: float = Field(
        ..., ge=0.0, le=1.0, description="Strength of connection (0-1)"
    )
    description: str = Field(..., description="Description of the connection")

    # Supporting evidence
    shared_entities: List[str] = Field(
        default_factory=list, description="Shared legal entities"
    )
    shared_topics: List[str] = Field(
        default_factory=list, description="Shared legal topics"
    )
    shared_cases: List[str] = Field(
        default_factory=list, description="Shared case references"
    )

    discovered_at: datetime = Field(
        default_factory=datetime.now, description="When connection was discovered"
    )


class TrendAnalysis(BaseModel):
    """Analysis of legal trends across multiple articles."""

    trend_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique trend identifier"
    )
    trend_name: str = Field(..., description="Name of the identified trend")
    trend_description: str = Field(..., description="Detailed description of the trend")

    # Trend characteristics
    trend_strength: str = Field(..., description="Strength: weak, moderate, strong")
    trend_direction: str = Field(
        ..., description="Direction: emerging, growing, declining, stable"
    )
    temporal_pattern: str = Field(
        ..., description="Temporal pattern: recent, ongoing, cyclical"
    )

    # Supporting data
    supporting_articles: List[str] = Field(
        ..., description="Article IDs supporting this trend"
    )
    key_indicators: List[str] = Field(..., description="Key indicators of the trend")
    affected_areas: List[str] = Field(
        ..., description="Legal areas affected by the trend"
    )

    # Impact assessment
    impact_level: str = Field(..., description="Impact level: low, medium, high")
    geographic_scope: str = Field(
        ..., description="Geographic scope: local, state, national, international"
    )

    # Timeline
    earliest_mention: Optional[datetime] = Field(
        None, description="Earliest mention in articles"
    )
    latest_mention: Optional[datetime] = Field(
        None, description="Latest mention in articles"
    )
    analyzed_at: datetime = Field(
        default_factory=datetime.now, description="When analysis was performed"
    )


class ConnectionAnalysis(BaseModel):
    """Complete analysis of connections between articles."""

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique analysis identifier",
    )
    article_ids: List[str] = Field(..., description="Articles included in analysis")

    # Connections
    direct_connections: List[ArticleConnection] = Field(
        default_factory=list, description="Direct article connections"
    )
    thematic_clusters: List[Dict[str, Any]] = Field(
        default_factory=list, description="Thematic clusters of articles"
    )

    # Trends
    identified_trends: List[TrendAnalysis] = Field(
        default_factory=list, description="Identified legal trends"
    )

    # Analysis metadata
    connection_count: int = Field(0, description="Total number of connections found")
    cluster_count: int = Field(0, description="Number of thematic clusters")
    trend_count: int = Field(0, description="Number of trends identified")

    analyzed_at: datetime = Field(
        default_factory=datetime.now, description="When analysis was performed"
    )
    analysis_quality: float = Field(
        0.0, ge=0.0, le=1.0, description="Quality score of the analysis"
    )


class SimilarityAnalyzer:
    """Analyzes similarity between legal articles using various methods."""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2), min_df=2
        )

    def calculate_content_similarity(
        self, articles: List[Dict[str, str]]
    ) -> np.ndarray:
        """Calculate content similarity using TF-IDF vectors."""
        try:
            # Prepare texts for analysis
            texts = []
            for article in articles:
                # Combine title and content for analysis
                text = f"{article.get('title', '')} {article.get('content', '')}"
                texts.append(text)

            if len(texts) < 2:
                return np.array([])

            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            return similarity_matrix

        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return np.array([])

    def find_entity_overlaps(
        self, article_extractions: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], Dict[str, List[str]]]:
        """Find overlapping entities between articles."""
        overlaps = {}

        for i, extraction1 in enumerate(article_extractions):
            for j, extraction2 in enumerate(article_extractions[i + 1 :], i + 1):
                # Extract entity names
                entities1 = {
                    entity["name"].lower() for entity in extraction1.get("entities", [])
                }
                entities2 = {
                    entity["name"].lower() for entity in extraction2.get("entities", [])
                }

                # Find shared entities
                shared_entities = entities1.intersection(entities2)

                # Find shared topics
                topics1 = set(extraction1.get("legal_topics", []))
                topics2 = set(extraction2.get("legal_topics", []))
                shared_topics = topics1.intersection(topics2)

                # Find shared cases
                cases1 = {
                    case["case_name"].lower() for case in extraction1.get("cases", [])
                }
                cases2 = {
                    case["case_name"].lower() for case in extraction2.get("cases", [])
                }
                shared_cases = cases1.intersection(cases2)

                if shared_entities or shared_topics or shared_cases:
                    overlaps[(i, j)] = {
                        "entities": list(shared_entities),
                        "topics": list(shared_topics),
                        "cases": list(shared_cases),
                    }

        return overlaps

    def calculate_temporal_proximity(
        self, articles: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], float]:
        """Calculate temporal proximity between articles."""
        proximities = {}

        for i, article1 in enumerate(articles):
            for j, article2 in enumerate(articles[i + 1 :], i + 1):
                date1 = article1.get("scraped_at")
                date2 = article2.get("scraped_at")

                if date1 and date2:
                    # Calculate time difference in days
                    time_diff = abs((date1 - date2).days)

                    # Convert to proximity score (closer in time = higher score)
                    if time_diff == 0:
                        proximity = 1.0
                    elif time_diff <= 7:
                        proximity = 0.8
                    elif time_diff <= 30:
                        proximity = 0.5
                    elif time_diff <= 90:
                        proximity = 0.3
                    else:
                        proximity = 0.1

                    proximities[(i, j)] = proximity

        return proximities


class TrendDetector:
    """Detects legal trends from article analysis."""

    def __init__(self):
        self.min_articles_for_trend = 3
        self.trend_confidence_threshold = 0.6

    def detect_trends(
        self, articles: List[Dict[str, Any]], extractions: List[Dict[str, Any]]
    ) -> List[TrendAnalysis]:
        """Detect trends from article data and extractions."""
        trends = []

        # Analyze topic frequency over time
        topic_trends = self._analyze_topic_trends(articles, extractions)
        trends.extend(topic_trends)

        # Analyze entity frequency trends
        entity_trends = self._analyze_entity_trends(articles, extractions)
        trends.extend(entity_trends)

        # Analyze jurisdictional trends
        jurisdiction_trends = self._analyze_jurisdiction_trends(articles, extractions)
        trends.extend(jurisdiction_trends)

        return trends

    def _analyze_topic_trends(
        self, articles: List[Dict[str, Any]], extractions: List[Dict[str, Any]]
    ) -> List[TrendAnalysis]:
        """Analyze trends in legal topics."""
        trends = []

        # Group articles by time periods
        time_periods = self._group_by_time_periods(articles)

        # Track topic frequency across time periods
        topic_frequency = defaultdict(lambda: defaultdict(int))

        for period, period_articles in time_periods.items():
            period_extractions = [
                extractions[i]
                for i, article in enumerate(articles)
                if article in period_articles
            ]

            for extraction in period_extractions:
                for topic in extraction.get("legal_topics", []):
                    topic_frequency[topic][period] += 1

        # Identify trending topics
        for topic, period_counts in topic_frequency.items():
            if (
                len(period_counts) >= 2
                and sum(period_counts.values()) >= self.min_articles_for_trend
            ):
                # Calculate trend direction
                periods = sorted(period_counts.keys())
                early_count = sum(
                    period_counts[p] for p in periods[: len(periods) // 2]
                )
                late_count = sum(period_counts[p] for p in periods[len(periods) // 2 :])

                if late_count > early_count * 1.5:
                    direction = "growing"
                    strength = "strong" if late_count > early_count * 2 else "moderate"
                elif late_count < early_count * 0.7:
                    direction = "declining"
                    strength = "moderate"
                else:
                    direction = "stable"
                    strength = "weak"

                # Get supporting articles
                supporting_articles = []
                for i, extraction in enumerate(extractions):
                    if topic in extraction.get("legal_topics", []):
                        supporting_articles.append(articles[i]["id"])

                trend = TrendAnalysis(
                    trend_name=f"{topic.title()} Legal Topic Trend",
                    trend_description=f"Trend analysis for {topic} legal topic across recent articles",
                    trend_strength=strength,
                    trend_direction=direction,
                    temporal_pattern="recent",
                    supporting_articles=supporting_articles,
                    key_indicators=[
                        f"Topic: {topic}",
                        f"Article count: {sum(period_counts.values())}",
                    ],
                    affected_areas=[topic],
                    impact_level="medium" if sum(period_counts.values()) > 5 else "low",
                    geographic_scope="national",  # Default scope
                )

                trends.append(trend)

        return trends

    def _analyze_entity_trends(
        self, articles: List[Dict[str, Any]], extractions: List[Dict[str, Any]]
    ) -> List[TrendAnalysis]:
        """Analyze trends in entity mentions."""
        trends = []

        # Count entity frequencies
        entity_counts = Counter()
        entity_articles = defaultdict(set)

        for i, extraction in enumerate(extractions):
            for entity in extraction.get("entities", []):
                entity_name = entity["name"]
                entity_counts[entity_name] += 1
                entity_articles[entity_name].add(articles[i]["id"])

        # Identify frequently mentioned entities
        for entity_name, count in entity_counts.most_common(10):
            if count >= self.min_articles_for_trend:
                trend = TrendAnalysis(
                    trend_name=f"{entity_name} - High Profile Entity",
                    trend_description=f"Frequent mentions of {entity_name} across multiple legal articles",
                    trend_strength="strong" if count > 5 else "moderate",
                    trend_direction="emerging",
                    temporal_pattern="recent",
                    supporting_articles=list(entity_articles[entity_name]),
                    key_indicators=[
                        f"Entity: {entity_name}",
                        f"Mention count: {count}",
                    ],
                    affected_areas=["entity_analysis"],
                    impact_level="high" if count > 7 else "medium",
                    geographic_scope="national",
                )

                trends.append(trend)

        return trends

    def _analyze_jurisdiction_trends(
        self, articles: List[Dict[str, Any]], extractions: List[Dict[str, Any]]
    ) -> List[TrendAnalysis]:
        """Analyze trends in jurisdiction activity."""
        trends = []

        # Count jurisdiction activity
        jurisdiction_counts = Counter()

        for extraction in extractions:
            jurisdiction = extraction.get("jurisdiction")
            if jurisdiction:
                jurisdiction_counts[jurisdiction] += 1

        # Identify active jurisdictions
        total_articles = len(articles)
        for jurisdiction, count in jurisdiction_counts.items():
            if count >= self.min_articles_for_trend:
                percentage = (count / total_articles) * 100

                if percentage > 40:
                    impact = "high"
                    strength = "strong"
                elif percentage > 20:
                    impact = "medium"
                    strength = "moderate"
                else:
                    impact = "low"
                    strength = "weak"

                trend = TrendAnalysis(
                    trend_name=f"{jurisdiction.title()} Jurisdiction Activity",
                    trend_description=f"High activity in {jurisdiction} jurisdiction ({percentage:.1f}% of articles)",
                    trend_strength=strength,
                    trend_direction="ongoing",
                    temporal_pattern="recent",
                    supporting_articles=[
                        articles[i]["id"]
                        for i, e in enumerate(extractions)
                        if e.get("jurisdiction") == jurisdiction
                    ],
                    key_indicators=[
                        f"Jurisdiction: {jurisdiction}",
                        f"Activity: {percentage:.1f}%",
                    ],
                    affected_areas=["jurisdiction_analysis"],
                    impact_level=impact,
                    geographic_scope=jurisdiction,
                )

                trends.append(trend)

        return trends

    def _group_by_time_periods(
        self, articles: List[Dict[str, Any]], days_per_period: int = 7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group articles by time periods."""
        periods = defaultdict(list)

        for article in articles:
            scraped_date = article.get("scraped_at")
            if scraped_date:
                # Calculate period (week number)
                period_start = scraped_date - timedelta(days=scraped_date.weekday())
                period_key = period_start.strftime("%Y-W%U")
                periods[period_key].append(article)

        return dict(periods)


class ConnectionAgent:
    """
    Connection analysis agent that identifies relationships between legal articles.
    Implements cross-article analysis, thematic clustering, and trend detection.
    """

    def __init__(self):
        self.similarity_analyzer = SimilarityAnalyzer()
        self.trend_detector = TrendDetector()
        self.min_similarity_threshold = 0.3
        self.max_connections_per_article = 5

    async def analyze_article_connections(
        self, article_ids: List[str], time_window_days: int = 30
    ) -> ConnectionAnalysis:
        """
        Analyze connections between a set of articles.

        Args:
            article_ids: List of article IDs to analyze
            time_window_days: Time window for analysis

        Returns:
            ConnectionAnalysis with detailed connection information
        """
        try:
            # Get article data and extractions
            articles, extractions = await self._prepare_analysis_data(
                article_ids, time_window_days
            )

            if len(articles) < 2:
                logger.warning(
                    f"Insufficient articles for connection analysis: {len(articles)}"
                )
                return ConnectionAnalysis(article_ids=article_ids)

            # Calculate similarity matrix
            similarity_matrix = self.similarity_analyzer.calculate_content_similarity(
                articles
            )

            # Find entity overlaps
            entity_overlaps = self.similarity_analyzer.find_entity_overlaps(extractions)

            # Calculate temporal proximity
            temporal_proximities = (
                self.similarity_analyzer.calculate_temporal_proximity(articles)
            )

            # Generate connections
            connections = await self._generate_connections(
                articles,
                extractions,
                similarity_matrix,
                entity_overlaps,
                temporal_proximities,
            )

            # Create thematic clusters
            clusters = self._create_thematic_clusters(
                articles, extractions, similarity_matrix
            )

            # Detect trends
            trends = self.trend_detector.detect_trends(articles, extractions)

            # Enhance trends with LLM analysis
            enhanced_trends = await self._enhance_trends_with_llm(trends, articles)

            # Calculate analysis quality
            quality_score = self._calculate_analysis_quality(
                connections, clusters, enhanced_trends
            )

            analysis = ConnectionAnalysis(
                article_ids=article_ids,
                direct_connections=connections,
                thematic_clusters=clusters,
                identified_trends=enhanced_trends,
                connection_count=len(connections),
                cluster_count=len(clusters),
                trend_count=len(enhanced_trends),
                analysis_quality=quality_score,
            )

            logger.info(
                f"Connection analysis completed: {len(connections)} connections, "
                f"{len(clusters)} clusters, {len(enhanced_trends)} trends"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error in connection analysis: {e}")
            raise

    async def _prepare_analysis_data(
        self, article_ids: List[str], time_window_days: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare article data and extractions for analysis."""
        articles = []
        extractions = []

        # Get recent articles from database
        cutoff_date = datetime.now() - timedelta(days=time_window_days)

        # This is a simplified version - in practice, you'd filter by date and IDs
        db_articles = await database_agent.get_unprocessed_articles(limit=50)

        for article in db_articles:
            if article["scraped_at"] >= cutoff_date:
                articles.append(article)

                # Get extraction data
                extraction = await database_agent.get_extraction_by_article_id(
                    article["id"]
                )
                if extraction:
                    extractions.append(extraction)
                else:
                    # Create empty extraction if none exists
                    extractions.append(
                        {
                            "entities": [],
                            "legal_topics": [],
                            "cases": [],
                            "jurisdiction": None,
                        }
                    )

        return articles, extractions

    async def _generate_connections(
        self,
        articles: List[Dict[str, Any]],
        extractions: List[Dict[str, Any]],
        similarity_matrix: np.ndarray,
        entity_overlaps: Dict[Tuple[int, int], Dict[str, List[str]]],
        temporal_proximities: Dict[Tuple[int, int], float],
    ) -> List[ArticleConnection]:
        """Generate article connections based on various similarity measures."""
        connections = []

        if similarity_matrix.size == 0:
            return connections

        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                # Calculate combined connection strength
                content_similarity = (
                    similarity_matrix[i, j] if similarity_matrix.size > 0 else 0.0
                )
                temporal_proximity = temporal_proximities.get((i, j), 0.0)
                entity_overlap = entity_overlaps.get((i, j), {})

                # Calculate weighted connection strength
                connection_strength = (
                    content_similarity * 0.4
                    + temporal_proximity * 0.2
                    + self._calculate_entity_overlap_score(entity_overlap) * 0.4
                )

                if connection_strength >= self.min_similarity_threshold:
                    # Determine connection type
                    connection_type = self._determine_connection_type(
                        content_similarity, entity_overlap, temporal_proximity
                    )

                    # Generate connection description
                    description = await self._generate_connection_description(
                        articles[i], articles[j], entity_overlap, connection_type
                    )

                    connection = ArticleConnection(
                        article_id_1=articles[i]["id"],
                        article_id_2=articles[j]["id"],
                        connection_type=connection_type,
                        strength=connection_strength,
                        description=description,
                        shared_entities=entity_overlap.get("entities", []),
                        shared_topics=entity_overlap.get("topics", []),
                        shared_cases=entity_overlap.get("cases", []),
                    )

                    connections.append(connection)

        # Sort by strength and limit connections per article
        connections.sort(key=lambda c: c.strength, reverse=True)
        return self._limit_connections_per_article(connections)

    def _calculate_entity_overlap_score(
        self, entity_overlap: Dict[str, List[str]]
    ) -> float:
        """Calculate score based on entity overlap."""
        entities = len(entity_overlap.get("entities", []))
        topics = len(entity_overlap.get("topics", []))
        cases = len(entity_overlap.get("cases", []))

        # Weight different types of overlaps
        score = (entities * 0.4 + topics * 0.4 + cases * 0.2) / 10.0
        return min(1.0, score)

    def _determine_connection_type(
        self,
        content_similarity: float,
        entity_overlap: Dict[str, List[str]],
        temporal_proximity: float,
    ) -> str:
        """Determine the type of connection between articles."""
        if len(entity_overlap.get("cases", [])) > 0:
            return "related_cases"
        elif len(entity_overlap.get("entities", [])) > 2:
            return "shared_entities"
        elif len(entity_overlap.get("topics", [])) > 1:
            return "thematic_similarity"
        elif temporal_proximity > 0.7:
            return "temporal_proximity"
        elif content_similarity > 0.6:
            return "content_similarity"
        else:
            return "weak_connection"

    async def _generate_connection_description(
        self,
        article1: Dict[str, Any],
        article2: Dict[str, Any],
        entity_overlap: Dict[str, List[str]],
        connection_type: str,
    ) -> str:
        """Generate a human-readable description of the connection."""
        try:
            # Create simple description based on overlap
            if entity_overlap.get("cases"):
                return f"Both articles discuss related legal cases: {', '.join(entity_overlap['cases'][:2])}"
            elif entity_overlap.get("entities"):
                return f"Articles share common entities: {', '.join(entity_overlap['entities'][:2])}"
            elif entity_overlap.get("topics"):
                return f"Articles cover similar legal topics: {', '.join(entity_overlap['topics'][:2])}"
            else:
                return f"Articles connected by {connection_type.replace('_', ' ')}"

        except Exception as e:
            logger.warning(f"Error generating connection description: {e}")
            return f"Articles connected by {connection_type.replace('_', ' ')}"

    def _create_thematic_clusters(
        self,
        articles: List[Dict[str, Any]],
        extractions: List[Dict[str, Any]],
        similarity_matrix: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Create thematic clusters of articles."""
        if similarity_matrix.size == 0:
            return []

        clusters = []
        clustered_indices = set()

        # Simple clustering based on similarity threshold
        cluster_threshold = 0.4

        for i in range(len(articles)):
            if i in clustered_indices:
                continue

            # Start new cluster
            cluster_articles = [i]
            cluster_topics = set(extractions[i].get("legal_topics", []))

            # Find similar articles
            for j in range(i + 1, len(articles)):
                if j in clustered_indices:
                    continue

                if similarity_matrix[i, j] >= cluster_threshold:
                    cluster_articles.append(j)
                    cluster_topics.update(extractions[j].get("legal_topics", []))

            if (
                len(cluster_articles) >= 2
            ):  # Only create cluster if it has multiple articles
                cluster = {
                    "cluster_id": str(uuid.uuid4()),
                    "article_ids": [articles[idx]["id"] for idx in cluster_articles],
                    "article_count": len(cluster_articles),
                    "common_topics": list(cluster_topics),
                    "cluster_strength": (
                        np.mean(
                            [
                                similarity_matrix[
                                    cluster_articles[x], cluster_articles[y]
                                ]
                                for x in range(len(cluster_articles))
                                for y in range(x + 1, len(cluster_articles))
                            ]
                        )
                        if len(cluster_articles) > 1
                        else 0.0
                    ),
                }

                clusters.append(cluster)
                clustered_indices.update(cluster_articles)

        return clusters

    async def _enhance_trends_with_llm(
        self, trends: List[TrendAnalysis], articles: List[Dict[str, Any]]
    ) -> List[TrendAnalysis]:
        """Enhance trend analysis using LLM insights."""
        enhanced_trends = []

        for trend in trends[:5]:  # Limit to top 5 trends for LLM analysis
            try:
                # Get sample articles for this trend
                sample_articles = [
                    article
                    for article in articles
                    if article["id"] in trend.supporting_articles[:3]
                ]

                if sample_articles:
                    # Create summaries for LLM analysis
                    summaries_text = "\n\n".join(
                        [
                            f"Article: {article['title']}\nSummary: {article['content'][:300]}..."
                            for article in sample_articles
                        ]
                    )

                    # Use LLM to enhance trend analysis
                    enhanced_description = await self._get_llm_trend_analysis(
                        trend.trend_name, summaries_text
                    )

                    if enhanced_description:
                        trend.trend_description = enhanced_description

                enhanced_trends.append(trend)

            except Exception as e:
                logger.warning(f"Error enhancing trend {trend.trend_name}: {e}")
                enhanced_trends.append(trend)

        return enhanced_trends

    async def _get_llm_trend_analysis(
        self, trend_name: str, summaries: str
    ) -> Optional[str]:
        """Get enhanced trend analysis from LLM."""
        try:
            prompt = f"""
            Analyze the following legal trend and provide enhanced insights:
            
            Trend: {trend_name}
            
            Supporting Article Summaries:
            {summaries}
            
            Please provide:
            1. A comprehensive analysis of this legal trend
            2. Its potential implications for the legal field
            3. Key factors driving this trend
            4. Predicted future developments
            
            Keep the analysis professional and focused on legal significance.
            """

            response = await llm_client.generate(
                prompt, max_tokens=500, temperature=0.7
            )

            if response.success:
                return response.content
            else:
                logger.warning(f"LLM trend analysis failed: {response.error}")
                return None

        except Exception as e:
            logger.error(f"Error getting LLM trend analysis: {e}")
            return None

    def _limit_connections_per_article(
        self, connections: List[ArticleConnection]
    ) -> List[ArticleConnection]:
        """Limit the number of connections per article to avoid overcrowding."""
        article_connection_count = defaultdict(int)
        filtered_connections = []

        for connection in connections:
            if (
                article_connection_count[connection.article_id_1]
                < self.max_connections_per_article
                and article_connection_count[connection.article_id_2]
                < self.max_connections_per_article
            ):

                filtered_connections.append(connection)
                article_connection_count[connection.article_id_1] += 1
                article_connection_count[connection.article_id_2] += 1

        return filtered_connections

    def _calculate_analysis_quality(
        self,
        connections: List[ArticleConnection],
        clusters: List[Dict[str, Any]],
        trends: List[TrendAnalysis],
    ) -> float:
        """Calculate quality score for the connection analysis."""
        base_score = 0.5

        # Add points for successful analysis
        if len(connections) > 0:
            base_score += 0.2
            # Bonus for high-quality connections
            high_quality_connections = [c for c in connections if c.strength > 0.6]
            if len(high_quality_connections) > 0:
                base_score += 0.1

        if len(clusters) > 0:
            base_score += 0.1

        if len(trends) > 0:
            base_score += 0.2
            # Bonus for strong trends
            strong_trends = [t for t in trends if t.trend_strength == "strong"]
            if len(strong_trends) > 0:
                base_score += 0.1

        return min(1.0, base_score)


# Global connection agent instance
connection_agent = ConnectionAgent()
