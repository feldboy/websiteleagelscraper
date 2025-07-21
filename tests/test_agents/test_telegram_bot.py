"""
Tests for the Telegram bot agent.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from agents.telegram_bot import (
    TelegramBotAgent, 
    TelegramFormatter, 
    TelegramAPI,
    TelegramMessage,
    DeliveryResult
)
from agents.writer import GeneratedArticle
from agents.quality_assurance import ValidationResult


class TestTelegramFormatter:
    """Test the TelegramFormatter class."""
    
    @pytest.fixture
    def formatter(self):
        """Create a TelegramFormatter instance."""
        return TelegramFormatter()
    
    @pytest.mark.asyncio
    async def test_format_article_for_telegram(self, formatter, sample_generated_article):
        """Test formatting an article for Telegram."""
        # Mock LLM formatting to return None (use fallback)
        with patch.object(formatter, '_get_llm_telegram_format') as mock_llm:
            mock_llm.return_value = None
            
            telegram_message = await formatter.format_article_for_telegram(sample_generated_article)
            
            assert isinstance(telegram_message, TelegramMessage)
            assert telegram_message.article_id == sample_generated_article.article_id
            assert len(telegram_message.headline) > 0
            assert len(telegram_message.content) > 0
            assert telegram_message.character_count == len(telegram_message.content)
            assert telegram_message.character_count <= formatter.max_message_length
    
    @pytest.mark.asyncio
    async def test_format_article_with_validation(self, formatter, sample_generated_article):
        """Test formatting an article with validation result."""
        validation = ValidationResult(
            validation_id="test-validation",
            article_id=sample_generated_article.article_id,
            passed=True,
            overall_score=8.5,
            confidence=0.9,
            word_count_compliant=True,
            originality_score=0.95,
            legal_quality_score=8.0,
            readability_score=7.5,
            factual_accuracy_score=8.0,
            legal_terminology_adequate=True,
            structure_adequate=True
        )
        
        with patch.object(formatter, '_get_llm_telegram_format') as mock_llm:
            mock_llm.return_value = None
            
            telegram_message = await formatter.format_article_for_telegram(
                sample_generated_article, validation
            )
            
            # Should include quality indicators
            assert "Quality" in telegram_message.content or "✅" in telegram_message.content
    
    def test_create_telegram_headline(self, formatter):
        """Test creating Telegram headlines with emojis."""
        # Test court-related title
        court_title = "Supreme Court Rules on Important Case"
        headline = formatter._create_telegram_headline(court_title)
        assert "⚖️" in headline or "🏛️" in headline
        assert court_title in headline
        
        # Test law-related title
        law_title = "New Legal Regulation Announced"
        headline = formatter._create_telegram_headline(law_title)
        assert any(emoji in headline for emoji in formatter.legal_emojis.values())
    
    def test_manual_format_content(self, formatter, sample_generated_article):
        """Test manual content formatting fallback."""
        formatted = formatter._manual_format_content(sample_generated_article)
        
        assert len(formatted) > 0
        assert "*" in formatted  # Should contain markdown formatting
        assert sample_generated_article.summary in formatted
        
        # Should include tags
        for tag in sample_generated_article.tags[:4]:
            assert tag.title() in formatted
    
    def test_create_footer(self, formatter, sample_generated_article):
        """Test footer creation."""
        footer = formatter._create_footer(sample_generated_article)
        
        assert str(sample_generated_article.word_count) in footer
        assert sample_generated_article.author in footer
        assert "#LegalNews" in footer
        assert "📄" in footer  # Word count emoji
        assert "📅" in footer  # Date emoji
    
    def test_truncate_message(self, formatter):
        """Test message truncation for length limits."""
        # Create a very long message
        long_content = "This is a test sentence. " * 200  # Will exceed limit
        
        truncated = formatter._truncate_message(long_content)
        
        assert len(truncated) <= formatter.max_message_length
        assert "truncated for length" in truncated.lower()


class TestTelegramAPI:
    """Test the TelegramAPI class."""
    
    @pytest.fixture
    def telegram_message(self):
        """Create a test Telegram message."""
        return TelegramMessage(
            message_id="test-msg-123",
            chat_id="test-channel",
            headline="🏛️ Test Legal News",
            content="*Test Message*\n\nThis is a test legal news update.",
            article_id="test-article-123"
        )
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, telegram_message, mock_telegram_response):
        """Test successful message sending."""
        async with TelegramAPI() as api:
            # Mock the aiohttp session
            with patch.object(api, 'session') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_telegram_response
                
                mock_session.post.return_value.__aenter__.return_value = mock_response
                
                result = await api.send_message(telegram_message)
                
                assert result.success
                assert result.telegram_message_id == 123
                assert result.message_id == telegram_message.message_id
                assert result.response_time > 0
    
    @pytest.mark.asyncio
    async def test_send_message_failure(self, telegram_message):
        """Test message sending failure."""
        async with TelegramAPI() as api:
            # Mock failed response
            with patch.object(api, 'session') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 400
                mock_response.json.return_value = {
                    'ok': False,
                    'error_code': 400,
                    'description': 'Bad Request: chat not found'
                }
                
                mock_session.post.return_value.__aenter__.return_value = mock_response
                
                result = await api.send_message(telegram_message)
                
                assert not result.success
                assert result.error_code == "400"
                assert "chat not found" in result.error_message
    
    @pytest.mark.asyncio
    async def test_send_message_rate_limit(self, telegram_message):
        """Test message sending with rate limiting."""
        async with TelegramAPI() as api:
            # Mock rate limit response
            with patch.object(api, 'session') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 429
                mock_response.json.return_value = {
                    'ok': False,
                    'error_code': 429,
                    'description': 'Too Many Requests',
                    'parameters': {'retry_after': 30}
                }
                
                mock_session.post.return_value.__aenter__.return_value = mock_response
                
                result = await api.send_message(telegram_message)
                
                assert not result.success
                assert result.error_code == "429"
                assert result.retry_after == 30
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful bot connection test."""
        async with TelegramAPI() as api:
            with patch.object(api, 'session') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = {
                    'ok': True,
                    'result': {
                        'id': 123456789,
                        'is_bot': True,
                        'first_name': 'Legal Research Bot',
                        'username': 'legal_research_bot'
                    }
                }
                
                mock_session.get.return_value.__aenter__.return_value = mock_response
                
                result = await api.test_connection()
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test failed bot connection test."""
        async with TelegramAPI() as api:
            with patch.object(api, 'session') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 401
                mock_response.json.return_value = {
                    'ok': False,
                    'error_code': 401,
                    'description': 'Unauthorized'
                }
                
                mock_session.get.return_value.__aenter__.return_value = mock_response
                
                result = await api.test_connection()
                
                assert result is False


class TestTelegramBotAgent:
    """Test the TelegramBotAgent class."""
    
    @pytest.fixture
    def bot_agent(self):
        """Create a TelegramBotAgent instance."""
        return TelegramBotAgent()
    
    @pytest.mark.asyncio
    async def test_distribute_article_success(self, bot_agent, sample_generated_article):
        """Test successful article distribution."""
        validation = ValidationResult(
            validation_id="test-validation",
            article_id=sample_generated_article.article_id,
            passed=True,
            overall_score=8.0,
            confidence=0.9,
            word_count_compliant=True,
            originality_score=0.95,
            legal_quality_score=8.0,
            readability_score=7.5,
            factual_accuracy_score=8.0,
            legal_terminology_adequate=True,
            structure_adequate=True
        )
        
        # Mock successful delivery
        with patch.object(bot_agent, '_send_with_retry') as mock_send:
            mock_send.return_value = DeliveryResult(
                success=True,
                message_id="test-msg-123",
                telegram_message_id=456,
                response_time=1.5
            )
            
            result = await bot_agent.distribute_article(sample_generated_article, validation)
            
            assert result.success
            assert result.telegram_message_id == 456
            assert len(bot_agent.delivery_history) == 1
    
    @pytest.mark.asyncio
    async def test_distribute_article_failure(self, bot_agent, sample_generated_article):
        """Test failed article distribution."""
        # Mock failed delivery
        with patch.object(bot_agent, '_send_with_retry') as mock_send:
            mock_send.return_value = DeliveryResult(
                success=False,
                message_id="test-msg-123",
                error_message="Channel not found",
                response_time=1.0
            )
            
            result = await bot_agent.distribute_article(sample_generated_article)
            
            assert not result.success
            assert "Channel not found" in result.error_message
            assert len(bot_agent.delivery_history) == 1
    
    @pytest.mark.asyncio
    async def test_send_with_retry_success_first_attempt(self, bot_agent):
        """Test successful sending on first attempt."""
        telegram_message = TelegramMessage(
            message_id="test-msg-123",
            chat_id="test-channel",
            headline="Test Headline",
            content="Test content",
            article_id="test-article-123"
        )
        
        # Mock TelegramAPI to succeed on first try
        with patch('agents.telegram_bot.TelegramAPI') as mock_api_class:
            mock_api = AsyncMock()
            mock_api_class.return_value.__aenter__.return_value = mock_api
            
            mock_api.send_message.return_value = DeliveryResult(
                success=True,
                message_id=telegram_message.message_id,
                telegram_message_id=123
            )
            
            result = await bot_agent._send_with_retry(telegram_message)
            
            assert result.success
            assert mock_api.send_message.call_count == 1
    
    @pytest.mark.asyncio
    async def test_send_with_retry_retry_logic(self, bot_agent):
        """Test retry logic for failed sends."""
        telegram_message = TelegramMessage(
            message_id="test-msg-123",
            chat_id="test-channel",
            headline="Test Headline",
            content="Test content",
            article_id="test-article-123"
        )
        
        # Mock TelegramAPI to fail twice then succeed
        with patch('agents.telegram_bot.TelegramAPI') as mock_api_class:
            mock_api = AsyncMock()
            mock_api_class.return_value.__aenter__.return_value = mock_api
            
            # First two calls fail with retryable error
            mock_api.send_message.side_effect = [
                DeliveryResult(
                    success=False,
                    message_id=telegram_message.message_id,
                    error_code="429",
                    error_message="Too Many Requests"
                ),
                DeliveryResult(
                    success=False,
                    message_id=telegram_message.message_id,
                    error_code="500",
                    error_message="Internal Server Error"
                ),
                DeliveryResult(
                    success=True,
                    message_id=telegram_message.message_id,
                    telegram_message_id=123
                )
            ]
            
            # Speed up test by reducing retry delay
            bot_agent.retry_delay_base = 0.01
            
            result = await bot_agent._send_with_retry(telegram_message)
            
            assert result.success
            assert mock_api.send_message.call_count == 3
    
    def test_should_retry_logic(self, bot_agent):
        """Test retry decision logic."""
        # Should retry on rate limiting
        rate_limit_result = DeliveryResult(
            success=False,
            message_id="test",
            error_code="429"
        )
        assert bot_agent._should_retry(rate_limit_result, 0)
        
        # Should retry on server errors
        server_error_result = DeliveryResult(
            success=False,
            message_id="test",
            error_code="500"
        )
        assert bot_agent._should_retry(server_error_result, 0)
        
        # Should not retry on client errors (except 429)
        client_error_result = DeliveryResult(
            success=False,
            message_id="test",
            error_code="400"
        )
        assert not bot_agent._should_retry(client_error_result, 0)
        
        # Should not retry when max attempts reached
        assert not bot_agent._should_retry(rate_limit_result, 3)
    
    @pytest.mark.asyncio
    async def test_test_bot_setup(self, bot_agent):
        """Test bot setup testing."""
        # Mock successful bot test
        with patch('agents.telegram_bot.TelegramAPI') as mock_api_class:
            mock_api = AsyncMock()
            mock_api_class.return_value.__aenter__.return_value = mock_api
            
            mock_api.test_connection.return_value = True
            mock_api.get_chat_info.return_value = {
                'id': 'test-channel',
                'type': 'channel',
                'title': 'Test Channel'
            }
            mock_api.send_message.return_value = DeliveryResult(
                success=True,
                message_id="test-msg",
                telegram_message_id=123
            )
            
            result = await bot_agent.test_bot_setup()
            
            assert result['bot_connection']
            assert result['channel_access']
            assert result['send_permission']
            assert len(result['error_messages']) == 0
    
    @pytest.mark.asyncio
    async def test_get_delivery_statistics(self, bot_agent):
        """Test delivery statistics generation."""
        # Add some mock delivery history
        bot_agent.delivery_history = [
            DeliveryResult(success=True, message_id="msg1", response_time=1.0),
            DeliveryResult(success=True, message_id="msg2", response_time=2.0),
            DeliveryResult(success=False, message_id="msg3", error_code="400", response_time=0.5)
        ]
        
        stats = await bot_agent.get_delivery_statistics()
        
        assert stats['total_messages'] == 3
        assert stats['successful_deliveries'] == 2
        assert stats['failed_deliveries'] == 1
        assert stats['success_rate'] == 2/3
        assert stats['average_response_time'] == 1.17  # approximately
        assert len(stats['common_errors']) > 0
    
    def test_clear_delivery_history(self, bot_agent):
        """Test clearing delivery history."""
        # Add some history
        bot_agent.delivery_history = [
            DeliveryResult(success=True, message_id="msg1")
        ]
        
        assert len(bot_agent.delivery_history) == 1
        
        bot_agent.clear_delivery_history()
        
        assert len(bot_agent.delivery_history) == 0