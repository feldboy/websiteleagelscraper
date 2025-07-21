"""
Tests for the LLM client tool.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp

from tools.llm_client import LLMClient, OpenAIProvider, AnthropicProvider, LLMResponse
from config.settings import Settings


class TestOpenAIProvider:
    """Test the OpenAI provider."""

    @pytest.fixture
    def provider(self):
        """Create OpenAI provider instance."""
        return OpenAIProvider("test-api-key")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response_data = {
            "choices": [{"message": {"content": "Test response from OpenAI"}}],
            "usage": {"total_tokens": 50, "prompt_tokens": 20, "completion_tokens": 30},
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data

            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await provider.generate("Test prompt", max_tokens=100)

            assert result.success
            assert result.content == "Test response from OpenAI"
            assert result.tokens_used == 50
            assert result.provider == "openai"
            assert result.cost_estimate > 0

    @pytest.mark.asyncio
    async def test_generate_api_error(self, provider):
        """Test API error handling."""
        mock_response_data = {"error": {"message": "Invalid API key"}}

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.json.return_value = mock_response_data

            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await provider.generate("Test prompt")

            assert not result.success
            assert "Invalid API key" in result.error
            assert result.tokens_used == 0

    def test_calculate_cost(self, provider):
        """Test cost calculation."""
        usage = {"prompt_tokens": 100, "completion_tokens": 50}

        cost = provider._calculate_cost(usage)

        assert cost > 0
        # Cost should be input tokens * input rate + output tokens * output rate
        expected_cost = (100 / 1000) * 0.03 + (50 / 1000) * 0.06
        assert abs(cost - expected_cost) < 0.001


class TestAnthropicProvider:
    """Test the Anthropic provider."""

    @pytest.fixture
    def provider(self):
        """Create Anthropic provider instance."""
        return AnthropicProvider("test-api-key")

    @pytest.mark.asyncio
    async def test_generate_success(self, provider):
        """Test successful generation."""
        mock_response_data = {
            "content": [{"text": "Test response from Claude"}],
            "usage": {"input_tokens": 25, "output_tokens": 35},
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data

            mock_session.post.return_value.__aenter__.return_value = mock_response

            result = await provider.generate("Test prompt")

            assert result.success
            assert result.content == "Test response from Claude"
            assert result.tokens_used == 60  # 25 + 35
            assert result.provider == "anthropic"


class TestLLMClient:
    """Test the main LLM client."""

    @pytest.fixture
    def mock_settings_openai(self):
        """Mock settings for OpenAI."""
        return Settings(
            database_url="sqlite:///:memory:",
            llm_provider="openai",
            openai_api_key="test-openai-key",
            telegram_bot_token="test-bot-token",
            telegram_channel_id="test-channel",
        )

    @pytest.fixture
    def mock_settings_anthropic(self):
        """Mock settings for Anthropic."""
        return Settings(
            database_url="sqlite:///:memory:",
            llm_provider="anthropic",
            anthropic_api_key="test-anthropic-key",
            telegram_bot_token="test-bot-token",
            telegram_channel_id="test-channel",
        )

    def test_initialize_openai_provider(self, mock_settings_openai):
        """Test initializing with OpenAI provider."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()
            assert isinstance(client.provider, OpenAIProvider)

    def test_initialize_anthropic_provider(self, mock_settings_anthropic):
        """Test initializing with Anthropic provider."""
        with patch("tools.llm_client.settings", mock_settings_anthropic):
            client = LLMClient()
            assert isinstance(client.provider, AnthropicProvider)

    def test_initialize_invalid_provider(self):
        """Test initializing with invalid provider."""
        mock_settings = Settings(
            database_url="sqlite:///:memory:",
            llm_provider="invalid",
            telegram_bot_token="test-bot-token",
            telegram_channel_id="test-channel",
        )

        with patch("tools.llm_client.settings", mock_settings):
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                LLMClient()

    @pytest.mark.asyncio
    async def test_generate_with_retry_success(self, mock_settings_openai):
        """Test successful generation with retry logic."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()

            # Mock provider to return success
            mock_response = LLMResponse(
                content="Test response",
                provider="openai",
                model="gpt-4",
                tokens_used=50,
                cost_estimate=0.01,
                response_time=1.5,
                success=True,
            )

            with patch.object(client.provider, "generate") as mock_generate:
                mock_generate.return_value = mock_response

                result = await client.generate("Test prompt")

                assert result.success
                assert result.content == "Test response"
                assert client.usage_tracker.calls_made == 1
                assert client.usage_tracker.total_tokens == 50

    @pytest.mark.asyncio
    async def test_generate_with_retry_failure_then_success(self, mock_settings_openai):
        """Test retry logic with initial failure then success."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()

            # Mock provider to fail first, then succeed
            failed_response = LLMResponse(
                content="",
                provider="openai",
                model="gpt-4",
                tokens_used=0,
                cost_estimate=0.0,
                response_time=1.0,
                success=False,
                error="rate limit exceeded",
            )

            success_response = LLMResponse(
                content="Test response",
                provider="openai",
                model="gpt-4",
                tokens_used=50,
                cost_estimate=0.01,
                response_time=1.5,
                success=True,
            )

            with patch.object(client.provider, "generate") as mock_generate:
                mock_generate.side_effect = [failed_response, success_response]

                # Speed up test by reducing retry delay
                with patch("asyncio.sleep"):
                    result = await client.generate("Test prompt")

                assert result.success
                assert result.content == "Test response"
                assert mock_generate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_json_success(self, mock_settings_openai):
        """Test JSON generation."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()

            # Mock provider to return JSON
            json_content = '{"key": "value", "number": 42}'
            mock_response = LLMResponse(
                content=json_content,
                provider="openai",
                model="gpt-4",
                tokens_used=30,
                cost_estimate=0.005,
                response_time=1.2,
                success=True,
            )

            with patch.object(client.provider, "generate") as mock_generate:
                mock_generate.return_value = mock_response

                result = await client.generate_json("Generate JSON")

                assert result == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_generate_json_with_markdown(self, mock_settings_openai):
        """Test JSON generation with markdown code blocks."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()

            # Mock provider to return JSON wrapped in markdown
            json_content = '```json\n{"key": "value", "number": 42}\n```'
            mock_response = LLMResponse(
                content=json_content,
                provider="openai",
                model="gpt-4",
                tokens_used=30,
                cost_estimate=0.005,
                response_time=1.2,
                success=True,
            )

            with patch.object(client.provider, "generate") as mock_generate:
                mock_generate.return_value = mock_response

                result = await client.generate_json("Generate JSON")

                assert result == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_generate_json_invalid(self, mock_settings_openai):
        """Test JSON generation with invalid JSON."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()

            # Mock provider to return invalid JSON
            mock_response = LLMResponse(
                content="This is not valid JSON",
                provider="openai",
                model="gpt-4",
                tokens_used=20,
                cost_estimate=0.003,
                response_time=1.0,
                success=True,
            )

            with patch.object(client.provider, "generate") as mock_generate:
                mock_generate.return_value = mock_response

                with pytest.raises(ValueError, match="Invalid JSON response"):
                    await client.generate_json("Generate JSON")

    def test_usage_tracking(self, mock_settings_openai):
        """Test usage statistics tracking."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()

            # Add some usage data
            client.usage_tracker.add_usage(100, 0.05)
            client.usage_tracker.add_usage(150, 0.08)
            client.usage_tracker.add_error()

            stats = client.get_usage_stats()

            assert stats["total_tokens"] == 250
            assert stats["total_cost"] == 0.13
            assert stats["calls_made"] == 2
            assert stats["errors"] == 1
            assert stats["avg_tokens_per_call"] == 125.0
            assert stats["error_rate"] == 1 / 3  # 1 error out of 3 total operations

    def test_reset_usage_stats(self, mock_settings_openai):
        """Test resetting usage statistics."""
        with patch("tools.llm_client.settings", mock_settings_openai):
            client = LLMClient()

            # Add some usage data
            client.usage_tracker.add_usage(100, 0.05)
            client.usage_tracker.add_error()

            # Reset
            client.reset_usage_stats()

            stats = client.get_usage_stats()
            assert stats["total_tokens"] == 0
            assert stats["total_cost"] == 0.0
            assert stats["calls_made"] == 0
            assert stats["errors"] == 0
