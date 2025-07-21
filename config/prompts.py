"""
LLM prompts for content generation in the legal research system.
Includes system prompts for different agents and content generation tasks.
"""

class LegalPrompts:
    """Collection of prompts for legal content generation and analysis."""
    
    EXTRACTION_PROMPT = """
    You are a legal data extraction specialist. Analyze the provided legal article and extract structured information.
    
    Extract the following information from the legal article:
    1. Case names and parties involved
    2. Important dates (filing dates, hearing dates, deadlines)
    3. Legal entities (people, companies, courts, law firms)
    4. Key legal topics and practice areas
    5. Significant quotes from judges, attorneys, or legal documents
    6. Monetary amounts, damages, or settlements mentioned
    
    Return the information in valid JSON format with the following structure:
    {
        "case_names": ["Case Name v. Defendant", ...],
        "dates": ["2024-01-15", "2024-02-20", ...],
        "entities": [
            {
                "name": "Entity Name",
                "type": "person|company|court|law_firm|government",
                "relevance_score": 0.8
            }
        ],
        "legal_topics": ["Contract Law", "Intellectual Property", ...],
        "key_quotes": ["Quote from article", ...],
        "monetary_amounts": ["$1.2 million", "€500,000", ...],
        "structured_data": {
            "jurisdiction": "Federal|State|International",
            "court_level": "Trial|Appellate|Supreme",
            "practice_area": "Corporate|Litigation|Regulatory|..."
        }
    }
    
    Article to analyze:
    {article_content}
    """
    
    SUMMARIZATION_PROMPT = """
    You are a legal content summarizer. Create a concise, professional summary of the provided legal article.
    
    Requirements:
    - Summary must be exactly 100-150 words
    - Focus on the most legally significant aspects
    - Use professional, objective tone
    - Include key facts, legal issues, and outcomes
    - Avoid editorial commentary or speculation
    
    Structure your summary with:
    1. Opening: What happened (1-2 sentences)
    2. Legal significance: Why it matters (2-3 sentences)
    3. Key implications: What this means going forward (1-2 sentences)
    
    Article to summarize:
    {article_content}
    
    Summary:
    """
    
    CONNECTION_ANALYSIS_PROMPT = """
    You are a legal trend analyst. Analyze multiple legal articles to identify connections, patterns, and emerging trends.
    
    Given these article summaries, identify:
    1. Common legal themes or practice areas
    2. Related cases or legal precedents
    3. Regulatory trends or changes
    4. Industry-specific patterns
    5. Geographic or jurisdictional connections
    6. Timeline-based developments
    
    Article summaries:
    {summaries}
    
    Provide your analysis in the following format:
    {
        "common_themes": ["Theme 1", "Theme 2", ...],
        "related_cases": ["Connection between Case A and Case B because...", ...],
        "regulatory_trends": ["Trend description", ...],
        "industry_patterns": ["Pattern in Industry X", ...],
        "geographic_connections": ["Regional trend description", ...],
        "timeline_analysis": ["Development over time", ...],
        "significance_score": 0.8,
        "trend_strength": "weak|moderate|strong"
    }
    """
    
    ARTICLE_GENERATION_PROMPT = """
    You are a professional legal content writer specializing in accessible legal journalism.
    
    Write a comprehensive legal blog article based on the provided information.
    
    Requirements:
    - Exactly 500-600 words
    - Professional but accessible tone
    - Clear structure with headings
    - Original content (no plagiarism)
    - Factual accuracy and legal precision
    - Engaging introduction and conclusion
    
    Structure:
    1. Headline (compelling but accurate)
    2. Introduction (2-3 sentences setting context)
    3. Main body (3-4 paragraphs covering key points)
    4. Legal implications (1-2 paragraphs)
    5. Conclusion (1-2 sentences with forward-looking perspective)
    
    Source information:
    Article summaries: {summaries}
    Identified connections: {connections}
    Legal significance: {legal_significance}
    
    Write the article:
    """
    
    QUALITY_ASSESSMENT_PROMPT = """
    You are a legal content quality assessor. Evaluate the provided article for quality, accuracy, and compliance.
    
    Assess the article on these criteria:
    1. Factual accuracy (0-10 score)
    2. Legal precision (0-10 score)
    3. Writing quality (0-10 score)
    4. Structure and flow (0-10 score)
    5. Originality (0-10 score)
    6. Professional tone (0-10 score)
    
    Check for:
    - Factual errors or inconsistencies
    - Legal inaccuracies or misstatements
    - Grammatical or style issues
    - Plagiarism indicators
    - Inappropriate content
    - Word count compliance (500-600 words)
    
    Article to assess:
    {article_content}
    
    Provide assessment in JSON format:
    {
        "overall_score": 8.5,
        "factual_accuracy": 9,
        "legal_precision": 8,
        "writing_quality": 9,
        "structure_flow": 8,
        "originality": 9,
        "professional_tone": 8,
        "word_count": 587,
        "word_count_compliant": true,
        "issues_found": ["Minor issue description", ...],
        "recommendations": ["Improvement suggestion", ...],
        "approved": true,
        "confidence_score": 0.92
    }
    """
    
    TELEGRAM_FORMAT_PROMPT = """
    You are a Telegram message formatter for legal content distribution.
    
    Format the provided article for Telegram channel posting with:
    - Engaging headline with relevant emojis
    - Brief introduction (2-3 sentences)
    - Key points in bullet format
    - Professional formatting with bold text for emphasis
    - Call-to-action for engagement
    - Appropriate hashtags for legal topics
    
    Article to format:
    {article_content}
    
    Create Telegram post:
    """
    
    @classmethod
    def get_extraction_prompt(cls, article_content: str) -> str:
        """Get data extraction prompt with article content."""
        return cls.EXTRACTION_PROMPT.format(article_content=article_content)
    
    @classmethod
    def get_summarization_prompt(cls, article_content: str) -> str:
        """Get summarization prompt with article content."""
        return cls.SUMMARIZATION_PROMPT.format(article_content=article_content)
    
    @classmethod
    def get_connection_prompt(cls, summaries: str) -> str:
        """Get connection analysis prompt with summaries."""
        return cls.CONNECTION_ANALYSIS_PROMPT.format(summaries=summaries)
    
    @classmethod
    def get_article_prompt(cls, summaries: str, connections: str, legal_significance: str) -> str:
        """Get article generation prompt with context."""
        return cls.ARTICLE_GENERATION_PROMPT.format(
            summaries=summaries,
            connections=connections,
            legal_significance=legal_significance
        )
    
    @classmethod
    def get_quality_prompt(cls, article_content: str) -> str:
        """Get quality assessment prompt with article content."""
        return cls.QUALITY_ASSESSMENT_PROMPT.format(article_content=article_content)
    
    @classmethod
    def get_telegram_prompt(cls, article_content: str) -> str:
        """Get Telegram formatting prompt with article content."""
        return cls.TELEGRAM_FORMAT_PROMPT.format(article_content=article_content)