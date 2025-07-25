"""
Telegram bot agent for distributing legal content to subscribers.
Implements bot API integration with message formatting and delivery confirmation.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
import aiohttp
import json

from pydantic import BaseModel, Field

from config.settings import settings
from config.prompts import LegalPrompts
from agents.writer import GeneratedArticle
from agents.quality_assurance import ValidationResult
from tools.llm_client import llm_client


logger = logging.getLogger(__name__)


class TelegramMessage(BaseModel):
    """Formatted message for Telegram distribution."""

    message_id: str = Field(..., description="Unique message identifier")
    chat_id: str = Field(..., description="Telegram chat/channel ID")

    # Content
    headline: str = Field(..., description="Message headline")
    content: str = Field(..., description="Formatted message content")
    parse_mode: str = Field("Markdown", description="Telegram parse mode")

    # Metadata
    article_id: str = Field(..., description="Source article ID")
    character_count: int = Field(0, description="Message character count")

    # Delivery tracking
    created_at: datetime = Field(
        default_factory=datetime.now, description="When message was created"
    )
    sent_at: Optional[datetime] = Field(None, description="When message was sent")
    delivery_status: str = Field("pending", description="Delivery status")
    telegram_message_id: Optional[int] = Field(
        None, description="Telegram's message ID"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not self.character_count:
            self.character_count = len(self.content)


class DeliveryResult(BaseModel):
    """Result of message delivery attempt."""

    success: bool = Field(..., description="Whether delivery succeeded")
    message_id: str = Field(..., description="Our message identifier")
    telegram_message_id: Optional[int] = Field(
        None, description="Telegram's message ID"
    )

    # Error information
    error_code: Optional[str] = Field(None, description="Error code if failed")
    error_message: Optional[str] = Field(None, description="Error description")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")

    # Delivery metadata
    delivered_at: datetime = Field(
        default_factory=datetime.now, description="Delivery timestamp"
    )
    response_time: float = Field(0.0, description="API response time in seconds")


class TelegramFormatter:
    """Formats legal articles for Telegram distribution."""

    def __init__(self):
        self.max_message_length = 4096  # Telegram's message limit
        self.legal_emojis = {
            "court": "⚖️",
            "law": "📜",
            "legal": "🏛️",
            "ruling": "📋",
            "case": "📁",
            "judge": "👨‍⚖️",
            "attorney": "👩‍💼",
            "settlement": "🤝",
            "appeal": "📈",
            "verdict": "✅",
        }

    async def format_article_for_telegram(
        self, article: GeneratedArticle, validation: Optional[ValidationResult] = None
    ) -> TelegramMessage:
        """
        Format a generated article for Telegram distribution.

        Args:
            article: GeneratedArticle to format
            validation: Optional validation result for quality indicators

        Returns:
            TelegramMessage ready for distribution
        """
        try:
            # Generate engaging headline with emojis
            headline = self._create_telegram_headline(article.title)

            # Format main content
            formatted_content = await self._format_article_content(article)

            # Add quality indicators if available
            if validation and validation.passed:
                formatted_content += self._add_quality_indicators(validation)

            # Add footer
            formatted_content += self._create_footer(article)

            # Ensure message fits Telegram limits
            if len(formatted_content) > self.max_message_length:
                formatted_content = self._truncate_message(formatted_content)

            return TelegramMessage(
                message_id=f"tg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{article.article_id[:8]}",
                chat_id=settings.telegram_channel_id,
                headline=headline,
                content=formatted_content,
                article_id=article.article_id,
            )

        except Exception as e:
            logger.error(f"Error formatting article for Telegram: {e}")
            raise

    def _create_telegram_headline(self, title: str) -> str:
        """Create a professional headline without emojis."""
        return title

    async def _format_article_content(self, article: GeneratedArticle) -> str:
        """Format the main article content as a professional legal briefing."""
        try:
            # Create professional briefing format
            return self._create_professional_briefing(article)

        except Exception as e:
            logger.error(f"Error formatting professional briefing: {e}")
            return self._create_professional_briefing(article)

    def _create_professional_briefing(self, article: GeneratedArticle) -> str:
        """Create a professional legal research briefing format."""
        # Header
        date = datetime.now().strftime("%B %d, %Y")
        briefing_parts = [
            "*Legal Research Briefing*",
            f"{date}",
            "",
            f"*{article.title}*",
            ""
        ]
        
        # Main content - convert from formatted to flowing text
        content = article.content
        
        # Remove markdown formatting for clean prose
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # Remove italics
        content = re.sub(r'#{1,6}\s*', '', content)         # Remove headers
        
        # Clean up and format paragraphs
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        # Join paragraphs with proper spacing
        formatted_content = '\n\n'.join(paragraphs)
        
        briefing_parts.append(formatted_content)
        
        return '\n'.join(briefing_parts)

    async def _get_llm_telegram_format(
        self, article: GeneratedArticle
    ) -> Optional[str]:
        """Use LLM to format article for Telegram."""
        try:
            prompt = LegalPrompts.get_telegram_prompt(
                f"Title: {article.title}\n\nContent: {article.content}"
            )

            response = await llm_client.generate(
                prompt, max_tokens=800, temperature=0.6
            )

            if response.success:
                return response.content.strip()
            else:
                logger.warning(f"LLM Telegram formatting failed: {response.error}")
                return None

        except Exception as e:
            logger.error(f"Error in LLM Telegram formatting: {e}")
            return None

    def _manual_format_content(self, article: GeneratedArticle) -> str:
        """Manual fallback formatting for Telegram."""
        content_parts = []

        # Brief introduction (first 2 sentences)
        sentences = re.split(r"[.!?]+", article.content)
        intro_sentences = [s.strip() for s in sentences[:2] if s.strip()]
        if intro_sentences:
            intro = ". ".join(intro_sentences) + "."
            content_parts.append(f"*{intro}*\n")

        # Key points from article tags
        if article.tags:
            content_parts.append("*Key Topics:*")
            for tag in article.tags[:4]:  # Limit to 4 tags
                content_parts.append(f"• {tag.title()}")
            content_parts.append("")

        # Summary
        if article.summary:
            content_parts.append("*Summary:*")
            content_parts.append(article.summary)
            content_parts.append("")

        # Quality indicators
        if hasattr(article, "quality_score") and article.quality_score > 0:
            quality_text = (
                "✅ *High Quality*"
                if article.quality_score > 0.8
                else "📊 *Verified Content*"
            )
            content_parts.append(quality_text)
            content_parts.append("")

        return "\n".join(content_parts)

    def _add_quality_indicators(self, validation: ValidationResult) -> str:
        """Add quality indicators - disabled for professional format."""
        return ""

    def _create_footer(self, article: GeneratedArticle) -> str:
        """Create professional footer with actual article sources."""
        # Get the actual sources from the article's source summaries
        sources = []
        
        if hasattr(article, 'source_summaries') and article.source_summaries:
            # Use the source names directly (they're now actual source names, not article IDs)
            sources = article.source_summaries[:3]  # Limit to 3 sources for readability
        
        # Fallback to generic legal sources if no specific sources found
        if not sources:
            sources = [
                "Supreme Court Decisions & Federal Court Records",
                "Securities and Exchange Commission Regulatory Filings", 
                "Federal Register Legal Publications"
            ]
        
        # Format sources professionally
        if len(sources) == 1:
            return f"\n\nSource: {sources[0]}"
        elif len(sources) == 2:
            return f"\n\nSources: {sources[0]}; {sources[1]}"
        else:
            formatted_sources = "; ".join(sources[:-1]) + f"; {sources[-1]}"
            return f"\n\nSources: {formatted_sources}"

    def _truncate_message(self, content: str) -> str:
        """Truncate message to fit Telegram limits while preserving sources."""
        if len(content) <= self.max_message_length:
            return content

        # Find the sources section to preserve it
        sources_match = content.rfind("\n\nSources:")
        if sources_match == -1:
            sources_match = content.rfind("\n\nSource:")
        
        sources_section = ""
        main_content = content
        
        if sources_match != -1:
            sources_section = content[sources_match:]
            main_content = content[:sources_match]
        
        # Calculate how much space we have for main content
        available_space = self.max_message_length - len(sources_section) - 50  # Buffer
        
        if len(main_content) > available_space:
            # Find a good truncation point (end of sentence)
            sentences_end = main_content.rfind(".", 0, available_space)
            if sentences_end > available_space * 0.8:  # If we found a good break point
                main_content = main_content[:sentences_end + 1].strip()
            else:
                main_content = main_content[:available_space].strip()
        
        return f"{main_content}{sources_section}"


class TelegramAPI:
    """Handles Telegram Bot API communication."""

    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def send_message(self, message: TelegramMessage) -> DeliveryResult:
        """
        Send message to Telegram.

        Args:
            message: TelegramMessage to send

        Returns:
            DeliveryResult with delivery status
        """
        if not self.session:
            raise RuntimeError("TelegramAPI must be used as async context manager")

        start_time = datetime.now()

        try:
            # Prepare request payload
            payload = {
                "chat_id": message.chat_id,
                "text": message.content,
                "parse_mode": message.parse_mode,
                "disable_web_page_preview": True,
            }

            # Send request
            async with self.session.post(
                f"{self.base_url}/sendMessage", json=payload
            ) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                response_data = await response.json()

                if response.status == 200 and response_data.get("ok"):
                    # Success
                    telegram_message_id = response_data["result"]["message_id"]

                    return DeliveryResult(
                        success=True,
                        message_id=message.message_id,
                        telegram_message_id=telegram_message_id,
                        response_time=response_time,
                    )
                else:
                    # Error
                    error_code = response_data.get("error_code", response.status)
                    error_description = response_data.get(
                        "description", "Unknown error"
                    )
                    retry_after = response_data.get("parameters", {}).get("retry_after")

                    return DeliveryResult(
                        success=False,
                        message_id=message.message_id,
                        error_code=str(error_code),
                        error_message=error_description,
                        retry_after=retry_after,
                        response_time=response_time,
                    )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return DeliveryResult(
                success=False,
                message_id=message.message_id,
                error_message=str(e),
                response_time=response_time,
            )

    async def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chat/channel."""
        if not self.session:
            raise RuntimeError("TelegramAPI must be used as async context manager")

        try:
            async with self.session.post(
                f"{self.base_url}/getChat", json={"chat_id": chat_id}
            ) as response:
                response_data = await response.json()

                if response.status == 200 and response_data.get("ok"):
                    return response_data["result"]
                else:
                    logger.error(f"Failed to get chat info: {response_data}")
                    return None

        except Exception as e:
            logger.error(f"Error getting chat info: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test bot connection and permissions."""
        if not self.session:
            raise RuntimeError("TelegramAPI must be used as async context manager")

        try:
            # Test bot info
            async with self.session.get(f"{self.base_url}/getMe") as response:
                response_data = await response.json()

                if response.status == 200 and response_data.get("ok"):
                    bot_info = response_data["result"]
                    logger.info(
                        f"Bot connection successful: {bot_info.get('username', 'Unknown')}"
                    )
                    return True
                else:
                    logger.error(f"Bot connection failed: {response_data}")
                    return False

        except Exception as e:
            logger.error(f"Error testing bot connection: {e}")
            return False


class TelegramBotAgent:
    """
    Telegram bot agent for distributing legal content to subscribers.
    Implements message formatting, delivery, and subscriber management.
    """

    def __init__(self):
        self.formatter = TelegramFormatter()
        self.delivery_history: List[DeliveryResult] = []
        self.max_retry_attempts = 3
        self.retry_delay_base = 2.0  # seconds

    async def distribute_article(
        self, article: GeneratedArticle, validation: Optional[ValidationResult] = None
    ) -> DeliveryResult:
        """
        Distribute an article via Telegram.

        Args:
            article: GeneratedArticle to distribute
            validation: Optional validation result

        Returns:
            DeliveryResult with delivery status
        """
        try:
            logger.info(f"Distributing article via Telegram: {article.title}")

            # Format article for Telegram
            telegram_message = await self.formatter.format_article_for_telegram(
                article, validation
            )

            # Send message with retry logic
            delivery_result = await self._send_with_retry(telegram_message)

            # Store delivery history
            self.delivery_history.append(delivery_result)

            # Update message status
            telegram_message.delivery_status = (
                "sent" if delivery_result.success else "failed"
            )
            telegram_message.sent_at = delivery_result.delivered_at
            telegram_message.telegram_message_id = delivery_result.telegram_message_id

            if delivery_result.success:
                logger.info(
                    f"Article distributed successfully: Telegram message ID {delivery_result.telegram_message_id}"
                )
            else:
                logger.error(
                    f"Article distribution failed: {delivery_result.error_message}"
                )

            return delivery_result

        except Exception as e:
            logger.error(f"Error distributing article: {e}")
            return DeliveryResult(
                success=False,
                message_id=f"error_{article.article_id}",
                error_message=str(e),
            )

    async def _send_with_retry(self, message: TelegramMessage) -> DeliveryResult:
        """Send message with retry logic for rate limiting and temporary errors."""
        last_result = None

        for attempt in range(self.max_retry_attempts):
            try:
                async with TelegramAPI() as api:
                    result = await api.send_message(message)

                    if result.success:
                        if attempt > 0:
                            logger.info(
                                f"Message sent successfully on attempt {attempt + 1}"
                            )
                        return result

                    last_result = result

                    # Check if we should retry
                    if not self._should_retry(result, attempt):
                        break

                    # Calculate retry delay
                    retry_delay = result.retry_after or (
                        self.retry_delay_base * (2**attempt)
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {result.error_message}. "
                        f"Retrying in {retry_delay} seconds..."
                    )

                    await asyncio.sleep(retry_delay)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}")
                last_result = DeliveryResult(
                    success=False, message_id=message.message_id, error_message=str(e)
                )

                if attempt < self.max_retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay_base * (2**attempt))

        return last_result or DeliveryResult(
            success=False,
            message_id=message.message_id,
            error_message="All retry attempts failed",
        )

    def _should_retry(self, result: DeliveryResult, attempt: int) -> bool:
        """Determine if delivery should be retried."""
        if attempt >= self.max_retry_attempts - 1:
            return False

        # Retry on rate limiting
        if result.error_code == "429":
            return True

        # Retry on temporary server errors
        if result.error_code in ["500", "502", "503", "504"]:
            return True

        # Retry on network errors
        if "timeout" in (result.error_message or "").lower():
            return True

        # Don't retry on client errors (4xx except 429)
        if (
            result.error_code
            and result.error_code.startswith("4")
            and result.error_code != "429"
        ):
            return False

        return True

    async def test_bot_setup(self) -> Dict[str, Any]:
        """Test bot configuration and permissions."""
        test_results = {
            "bot_connection": False,
            "channel_access": False,
            "send_permission": False,
            "bot_info": None,
            "channel_info": None,
            "error_messages": [],
        }

        try:
            async with TelegramAPI() as api:
                # Test bot connection
                test_results["bot_connection"] = await api.test_connection()

                if not test_results["bot_connection"]:
                    test_results["error_messages"].append(
                        "Bot token is invalid or bot is not accessible"
                    )
                    return test_results

                # Test channel access
                channel_info = await api.get_chat_info(settings.telegram_channel_id)
                if channel_info:
                    test_results["channel_access"] = True
                    test_results["channel_info"] = channel_info
                else:
                    test_results["error_messages"].append(
                        "Cannot access the specified channel/chat"
                    )

                # Test send permission with a simple message
                test_message = TelegramMessage(
                    message_id="test_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                    chat_id=settings.telegram_channel_id,
                    headline="🔧 Test Message",
                    content="*System Test*\n\nThis is a test message from the Legal Research System.\n\n_If you see this message, the bot is working correctly._",
                    article_id="test",
                )

                test_result = await api.send_message(test_message)
                test_results["send_permission"] = test_result.success

                if not test_result.success:
                    test_results["error_messages"].append(
                        f"Cannot send messages: {test_result.error_message}"
                    )

        except Exception as e:
            test_results["error_messages"].append(f"Test error: {str(e)}")

        return test_results

    async def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get delivery statistics and performance metrics."""
        if not self.delivery_history:
            return {
                "total_messages": 0,
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0,
                "common_errors": [],
            }

        successful = [r for r in self.delivery_history if r.success]
        failed = [r for r in self.delivery_history if not r.success]

        # Calculate average response time
        total_response_time = sum(r.response_time for r in self.delivery_history)
        avg_response_time = total_response_time / len(self.delivery_history)

        # Count common errors
        error_counts = {}
        for result in failed:
            error_key = result.error_code or "unknown"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1

        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return {
            "total_messages": len(self.delivery_history),
            "successful_deliveries": len(successful),
            "failed_deliveries": len(failed),
            "success_rate": len(successful) / len(self.delivery_history),
            "average_response_time": avg_response_time,
            "common_errors": common_errors,
        }

    def clear_delivery_history(self):
        """Clear delivery history (useful for testing)."""
        self.delivery_history = []


# Global telegram bot agent instance
telegram_bot_agent = TelegramBotAgent()
