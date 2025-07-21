"""
Data extraction agent for parsing legal content into structured format.
Uses LLM-powered entity extraction with legal-specific parsing rules.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import uuid

from tools.llm_client import llm_client
from config.prompts import LegalPrompts
from models.extracted_data import (
    ExtractedData,
    LegalEntity,
    EntityType,
    MonetaryAmount,
    LegalDate,
    CaseReference,
    LegalTopic,
    Jurisdiction,
    CourtLevel,
)
from agents.database import database_agent


logger = logging.getLogger(__name__)


class LegalEntityExtractor:
    """Extracts legal entities from text using pattern matching and LLM."""

    def __init__(self):
        # Common legal entity patterns
        self.entity_patterns = {
            EntityType.COURT: [
                r"\b(?:Supreme Court|Court of Appeals|District Court|Circuit Court|Superior Court|Municipal Court)\b",
                r"\b(?:U\.?S\.? (?:Supreme )?Court|Federal Court)\b",
                r"\b\w+\s+(?:County|District|Circuit)\s+Court\b",
            ],
            EntityType.JUDGE: [
                r"\bJudge\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
                r"\bJustice\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
                r"\bChief\s+Justice\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            ],
            EntityType.ATTORNEY: [
                r"\b(?:Attorney|Lawyer|Counsel)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*,\s*Esq\.\b",
                r"\b(?:Partner|Associate)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            ],
            EntityType.LAW_FIRM: [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:&|and)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*,?\s*(?:LLP|LLC|P\.?C\.?|Law Firm)\b",
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Law\s+(?:Firm|Group|Associates|Partners)\b",
            ],
        }

    def extract_entities_pattern_based(self, text: str) -> List[LegalEntity]:
        """Extract entities using regex patterns."""
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.group().strip()
                    if len(name) > 3:  # Filter out very short matches
                        entities.append(
                            LegalEntity(
                                name=name,
                                entity_type=entity_type,
                                relevance_score=0.7,  # Pattern-based gets medium relevance
                            )
                        )

        return entities


class MonetaryAmountExtractor:
    """Extracts monetary amounts from legal text."""

    def __init__(self):
        # Patterns for monetary amounts
        self.money_patterns = [
            r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?",
            r"(?:USD|EUR|GBP)\s*[\d,]+(?:\.\d{2})?",
            r"[\d,]+(?:\.\d{2})?\s*(?:dollars|euros|pounds)",
            r"(?:damages|settlement|fine|penalty)\s+of\s+\$?[\d,]+(?:\.\d{2})?",
        ]

    def extract_amounts(self, text: str) -> List[MonetaryAmount]:
        """Extract monetary amounts from text."""
        amounts = []

        for pattern in self.money_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group()
                amount_value = self._parse_amount(amount_str)

                if amount_value > 0:
                    # Get context around the amount
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()

                    amounts.append(
                        MonetaryAmount(
                            amount=amount_value,
                            currency="USD",  # Default to USD
                            formatted_string=amount_str,
                            context=context,
                        )
                    )

        return amounts

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to numerical value."""
        try:
            # Remove currency symbols and normalize
            normalized = re.sub(r"[^\d.,kmb]", "", amount_str.lower())

            # Handle multipliers
            multiplier = 1
            if "k" in normalized or "thousand" in amount_str.lower():
                multiplier = 1000
            elif "m" in normalized or "million" in amount_str.lower():
                multiplier = 1000000
            elif "b" in normalized or "billion" in amount_str.lower():
                multiplier = 1000000000

            # Extract numeric part
            numeric = re.sub(r"[^\d.,]", "", normalized)
            numeric = numeric.replace(",", "")

            if numeric:
                return float(numeric) * multiplier

        except (ValueError, TypeError):
            pass

        return 0.0


class DateExtractor:
    """Extracts legal dates from text."""

    def __init__(self):
        self.date_patterns = [
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            r"\b\d{1,2}-\d{1,2}-\d{4}\b",
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        ]

        self.date_types = [
            "filing date",
            "hearing date",
            "trial date",
            "deadline",
            "settlement date",
            "ruling date",
            "appeal date",
            "discovery deadline",
        ]

    def extract_dates(self, text: str) -> List[LegalDate]:
        """Extract dates with legal context."""
        dates = []

        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group()
                parsed_date = self._parse_date(date_str)

                if parsed_date:
                    # Determine date type from context
                    date_type = self._determine_date_type(
                        text, match.start(), match.end()
                    )

                    # Get context
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()

                    dates.append(
                        LegalDate(
                            date_value=parsed_date,
                            date_type=date_type,
                            context=context,
                            confidence=0.8,
                        )
                    )

        return dates

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string to date object."""
        import dateparser

        try:
            parsed = dateparser.parse(date_str)
            return parsed.date() if parsed else None
        except Exception:
            return None

    def _determine_date_type(self, text: str, start: int, end: int) -> str:
        """Determine type of date from surrounding context."""
        # Look at text around the date
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].lower()

        for date_type in self.date_types:
            if date_type.replace(" ", "") in context.replace(" ", ""):
                return date_type

        return "mentioned date"


class CaseExtractor:
    """Extracts legal case references from text."""

    def __init__(self):
        self.case_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            r"\b(?:Case No\.|Docket No\.)\s*[\d\-]+\b",
            r"\b\d+\s+[A-Z][a-z]+\s+\d+\b",  # Citation pattern
        ]

    def extract_cases(self, text: str) -> List[CaseReference]:
        """Extract case references from text."""
        cases = []

        for pattern in self.case_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                case_text = match.group().strip()

                if " v. " in case_text or " v " in case_text:
                    # This is a case name
                    parties = re.split(r"\s+v\.?\s+", case_text, flags=re.IGNORECASE)
                    if len(parties) >= 2:
                        cases.append(
                            CaseReference(
                                case_name=case_text,
                                parties=parties,
                                year=self._extract_year_from_context(
                                    text, match.start(), match.end()
                                ),
                            )
                        )
                elif "Case No." in case_text or "Docket No." in case_text:
                    # This is a case number
                    case_number = re.sub(
                        r"(?:Case|Docket)\s+No\.?\s*",
                        "",
                        case_text,
                        flags=re.IGNORECASE,
                    )
                    cases.append(
                        CaseReference(
                            case_name=f"Case {case_number}",
                            case_number=case_number.strip(),
                        )
                    )

        return cases

    def _extract_year_from_context(
        self, text: str, start: int, end: int
    ) -> Optional[int]:
        """Extract year from context around case mention."""
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text[context_start:context_end]

        year_match = re.search(r"\b(19|20)\d{2}\b", context)
        if year_match:
            return int(year_match.group())

        return None


class ExtractionAgent:
    """
    Data extraction agent that processes legal articles into structured data.
    Uses both pattern matching and LLM-powered extraction for comprehensive analysis.
    """

    def __init__(self):
        self.entity_extractor = LegalEntityExtractor()
        self.money_extractor = MonetaryAmountExtractor()
        self.date_extractor = DateExtractor()
        self.case_extractor = CaseExtractor()

    async def extract_article_data(
        self, article_id: str, content: str, title: str = ""
    ) -> ExtractedData:
        """
        Extract structured data from a legal article.

        Args:
            article_id: Unique identifier for the article
            content: Article content text
            title: Article title

        Returns:
            ExtractedData object with extracted information
        """
        try:
            # Combine title and content for extraction
            full_text = f"{title}\n\n{content}"

            # Pattern-based extraction
            entities = self.entity_extractor.extract_entities_pattern_based(full_text)
            monetary_amounts = self.money_extractor.extract_amounts(full_text)
            dates = self.date_extractor.extract_dates(full_text)
            cases = self.case_extractor.extract_cases(full_text)

            # LLM-powered extraction for additional data
            llm_data = await self._extract_with_llm(full_text)

            # Merge pattern-based and LLM results
            all_entities = self._merge_entities(entities, llm_data.get("entities", []))
            all_topics = self._extract_legal_topics(llm_data.get("legal_topics", []))
            key_quotes = llm_data.get("key_quotes", [])

            # Determine jurisdiction and court level
            jurisdiction = self._determine_jurisdiction(full_text, all_entities)
            court_level = self._determine_court_level(full_text, all_entities)

            # Calculate quality metrics
            extraction_confidence = self._calculate_extraction_confidence(
                all_entities, cases, dates, monetary_amounts, key_quotes
            )
            completeness_score = self._calculate_completeness_score(
                all_entities, cases, dates, monetary_amounts, all_topics
            )

            extracted_data = ExtractedData(
                article_id=article_id,
                source_url=f"stored://article/{article_id}",  # Reference to stored article
                entities=all_entities,
                cases=cases,
                dates=dates,
                monetary_amounts=monetary_amounts,
                legal_topics=all_topics,
                key_quotes=key_quotes,
                jurisdiction=jurisdiction,
                court_level=court_level,
                practice_areas=llm_data.get("practice_areas", []),
                structured_data=llm_data.get("structured_data", {}),
                extraction_confidence=extraction_confidence,
                completeness_score=completeness_score,
            )

            logger.info(
                f"Extraction completed for article {article_id}: "
                f"confidence={extraction_confidence:.2f}, completeness={completeness_score:.2f}"
            )

            return extracted_data

        except Exception as e:
            logger.error(f"Error extracting data from article {article_id}: {e}")
            raise

    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract additional structured data."""
        try:
            prompt = LegalPrompts.get_extraction_prompt(text)
            response = await llm_client.generate_json(
                prompt, max_tokens=1000, temperature=0.3
            )

            # Validate and normalize LLM response
            return self._normalize_llm_response(response)

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return {}

    def _normalize_llm_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate LLM response data."""
        normalized = {}

        # Extract entities
        entities = []
        for entity_data in response.get("entities", []):
            try:
                entity_type = EntityType(entity_data.get("type", "other"))
                entities.append(
                    LegalEntity(
                        name=entity_data.get("name", ""),
                        entity_type=entity_type,
                        relevance_score=entity_data.get("relevance_score", 0.5),
                    )
                )
            except Exception:
                continue
        normalized["entities"] = entities

        # Extract legal topics
        topics = []
        for topic in response.get("legal_topics", []):
            try:
                legal_topic = LegalTopic(topic.lower().replace(" ", "_"))
                topics.append(legal_topic)
            except ValueError:
                # If not a standard topic, keep as practice area
                continue
        normalized["legal_topics"] = topics

        # Extract other fields
        normalized["key_quotes"] = response.get("key_quotes", [])[
            :5
        ]  # Limit to 5 quotes
        normalized["practice_areas"] = response.get("practice_areas", [])
        normalized["structured_data"] = response.get("structured_data", {})

        return normalized

    def _merge_entities(
        self, pattern_entities: List[LegalEntity], llm_entities: List[LegalEntity]
    ) -> List[LegalEntity]:
        """Merge entities from pattern matching and LLM extraction."""
        all_entities = pattern_entities.copy()

        # Add LLM entities that don't duplicate pattern entities
        for llm_entity in llm_entities:
            is_duplicate = any(
                entity.name.lower() == llm_entity.name.lower()
                for entity in all_entities
            )
            if not is_duplicate:
                all_entities.append(llm_entity)

        # Sort by relevance score
        all_entities.sort(key=lambda e: e.relevance_score, reverse=True)

        # Limit to top 20 entities
        return all_entities[:20]

    def _extract_legal_topics(self, topic_strings: List[str]) -> List[LegalTopic]:
        """Convert topic strings to LegalTopic enums."""
        topics = []
        for topic_str in topic_strings:
            try:
                # Normalize topic string
                normalized = topic_str.lower().replace(" ", "_").replace("-", "_")
                topic = LegalTopic(normalized)
                topics.append(topic)
            except ValueError:
                # If not a standard topic, skip
                continue

        return list(set(topics))  # Remove duplicates

    def _determine_jurisdiction(
        self, text: str, entities: List[LegalEntity]
    ) -> Optional[Jurisdiction]:
        """Determine legal jurisdiction from content."""
        text_lower = text.lower()

        # Check for federal indicators
        federal_indicators = [
            "federal court",
            "supreme court",
            "u.s. court",
            "federal judge",
        ]
        if any(indicator in text_lower for indicator in federal_indicators):
            return Jurisdiction.FEDERAL

        # Check for state indicators
        state_indicators = ["state court", "county court", "superior court"]
        if any(indicator in text_lower for indicator in state_indicators):
            return Jurisdiction.STATE

        # Check for international indicators
        international_indicators = [
            "international court",
            "treaty",
            "international law",
        ]
        if any(indicator in text_lower for indicator in international_indicators):
            return Jurisdiction.INTERNATIONAL

        return None

    def _determine_court_level(
        self, text: str, entities: List[LegalEntity]
    ) -> Optional[CourtLevel]:
        """Determine court level from content."""
        text_lower = text.lower()

        if "supreme court" in text_lower:
            return CourtLevel.SUPREME
        elif any(
            term in text_lower for term in ["court of appeals", "appellate court"]
        ):
            return CourtLevel.APPELLATE
        elif any(
            term in text_lower
            for term in ["district court", "trial court", "superior court"]
        ):
            return CourtLevel.TRIAL
        elif "administrative" in text_lower:
            return CourtLevel.ADMINISTRATIVE

        return None

    def _calculate_extraction_confidence(
        self,
        entities: List[LegalEntity],
        cases: List[CaseReference],
        dates: List[LegalDate],
        amounts: List[MonetaryAmount],
        quotes: List[str],
    ) -> float:
        """Calculate confidence score for extraction."""
        base_score = 0.5

        # Add points for successful extractions
        if len(entities) > 0:
            base_score += 0.1
        if len(cases) > 0:
            base_score += 0.1
        if len(dates) > 0:
            base_score += 0.1
        if len(amounts) > 0:
            base_score += 0.1
        if len(quotes) > 0:
            base_score += 0.1

        # Bonus for high-relevance entities
        high_relevance_entities = [e for e in entities if e.relevance_score > 0.8]
        if len(high_relevance_entities) > 2:
            base_score += 0.1

        return min(1.0, base_score)

    def _calculate_completeness_score(
        self,
        entities: List[LegalEntity],
        cases: List[CaseReference],
        dates: List[LegalDate],
        amounts: List[MonetaryAmount],
        topics: List[LegalTopic],
    ) -> float:
        """Calculate completeness score for extraction."""
        max_categories = 5
        present_categories = 0

        if len(entities) > 0:
            present_categories += 1
        if len(cases) > 0:
            present_categories += 1
        if len(dates) > 0:
            present_categories += 1
        if len(amounts) > 0:
            present_categories += 1
        if len(topics) > 0:
            present_categories += 1

        return present_categories / max_categories

    async def process_unprocessed_articles(self, limit: int = 10) -> List[str]:
        """Process unprocessed articles from the database."""
        try:
            articles = await database_agent.get_unprocessed_articles(limit)
            processed_ids = []

            for article in articles:
                try:
                    extracted_data = await self.extract_article_data(
                        article["id"], article["content"], article["title"]
                    )

                    # Store extraction in database
                    extraction_id = await database_agent.store_extraction(
                        extracted_data
                    )
                    processed_ids.append(extraction_id)

                    logger.info(
                        f"Processed article {article['id']} -> extraction {extraction_id}"
                    )

                except Exception as e:
                    logger.error(f"Failed to process article {article['id']}: {e}")
                    continue

            return processed_ids

        except Exception as e:
            logger.error(f"Error processing unprocessed articles: {e}")
            raise


# Global extraction agent instance
extraction_agent = ExtractionAgent()
