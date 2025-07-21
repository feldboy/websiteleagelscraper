"""
Centralized LLM client with support for multiple providers and retry logic.
Implements token usage tracking and exponential backoff for API reliability.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum

import aiohttp
from config.settings import settings


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"


@dataclass
class LLMResponse:
    """Response from LLM API call."""

    content: str
    provider: str
    model: str
    tokens_used: int
    cost_estimate: float
    response_time: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TokenUsageTracker:
    """Tracks token usage and costs across LLM calls."""

    total_tokens: int = 0
    total_cost: float = 0.0
    calls_made: int = 0
    errors: int = 0

    def add_usage(self, tokens: int, cost: float) -> None:
        """Add token usage to tracker."""
        self.total_tokens += tokens
        self.total_cost += cost
        self.calls_made += 1

    def add_error(self) -> None:
        """Record an error."""
        self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "calls_made": self.calls_made,
            "errors": self.errors,
            "avg_tokens_per_call": self.total_tokens / max(self.calls_made, 1),
            "error_rate": self.errors / max(self.calls_made + self.errors, 1),
        }


class LLMProviderBase(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self, prompt: str, max_tokens: int = 800, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Generate content using the LLM."""
        pass


class OpenAIProvider(LLMProviderBase):
    """OpenAI API provider implementation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.model = "gpt-4"

        # Token costs per 1K tokens (approximate, as of 2024)
        self.cost_per_1k_tokens = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        }

    async def generate(
        self, prompt: str, max_tokens: int = 800, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Generate content using OpenAI API."""
        start_time = datetime.now()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        content = response_data["choices"][0]["message"]["content"]
                        usage = response_data.get("usage", {})
                        tokens_used = usage.get("total_tokens", 0)

                        # Calculate cost estimate
                        cost_estimate = self._calculate_cost(usage)

                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content=content,
                            provider="openai",
                            model=self.model,
                            tokens_used=tokens_used,
                            cost_estimate=cost_estimate,
                            response_time=response_time,
                            success=True,
                            metadata=response_data,
                        )
                    else:
                        error_msg = response_data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content="",
                            provider="openai",
                            model=self.model,
                            tokens_used=0,
                            cost_estimate=0.0,
                            response_time=response_time,
                            success=False,
                            error=f"API Error: {error_msg}",
                        )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return LLMResponse(
                content="",
                provider="openai",
                model=self.model,
                tokens_used=0,
                cost_estimate=0.0,
                response_time=response_time,
                success=False,
                error=f"Request failed: {str(e)}",
            )

    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost estimate based on token usage."""
        if self.model not in self.cost_per_1k_tokens:
            return 0.0

        costs = self.cost_per_1k_tokens[self.model]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]

        return input_cost + output_cost


class AnthropicProvider(LLMProviderBase):
    """Anthropic Claude API provider implementation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.model = "claude-3-sonnet-20240229"

        # Token costs per 1K tokens (approximate, as of 2024)
        self.cost_per_1k_tokens = {
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        }

    async def generate(
        self, prompt: str, max_tokens: int = 800, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Generate content using Anthropic Claude API."""
        start_time = datetime.now()

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            **kwargs,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        content = response_data["content"][0]["text"]
                        usage = response_data.get("usage", {})
                        tokens_used = usage.get("input_tokens", 0) + usage.get(
                            "output_tokens", 0
                        )

                        # Calculate cost estimate
                        cost_estimate = self._calculate_cost(usage)

                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content=content,
                            provider="anthropic",
                            model=self.model,
                            tokens_used=tokens_used,
                            cost_estimate=cost_estimate,
                            response_time=response_time,
                            success=True,
                            metadata=response_data,
                        )
                    else:
                        error_msg = response_data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content="",
                            provider="anthropic",
                            model=self.model,
                            tokens_used=0,
                            cost_estimate=0.0,
                            response_time=response_time,
                            success=False,
                            error=f"API Error: {error_msg}",
                        )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return LLMResponse(
                content="",
                provider="anthropic",
                model=self.model,
                tokens_used=0,
                cost_estimate=0.0,
                response_time=response_time,
                success=False,
                error=f"Request failed: {str(e)}",
            )

    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost estimate based on token usage."""
        if self.model not in self.cost_per_1k_tokens:
            return 0.0

        costs = self.cost_per_1k_tokens[self.model]
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]

        return input_cost + output_cost


class OpenRouterProvider(LLMProviderBase):
    """OpenRouter API provider implementation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = "anthropic/claude-3.5-sonnet"

        # Token costs per 1K tokens (approximate)
        self.cost_per_1k_tokens = {
            "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
            "openai/gpt-4": {"input": 0.03, "output": 0.06},
            "openai/gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        }

    async def generate(
        self, prompt: str, max_tokens: int = 800, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Generate content using OpenRouter API."""
        start_time = datetime.now()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://legal-research-system.local",
            "X-Title": "Legal Research System",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        content = response_data["choices"][0]["message"]["content"]
                        usage = response_data.get("usage", {})
                        tokens_used = usage.get("total_tokens", 0)

                        # Calculate cost estimate
                        cost_estimate = self._calculate_cost(usage)

                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content=content,
                            provider="openrouter",
                            model=self.model,
                            tokens_used=tokens_used,
                            cost_estimate=cost_estimate,
                            response_time=response_time,
                            success=True,
                            metadata=response_data,
                        )
                    else:
                        error_msg = response_data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content="",
                            provider="openrouter",
                            model=self.model,
                            tokens_used=0,
                            cost_estimate=0.0,
                            response_time=response_time,
                            success=False,
                            error=f"API Error: {error_msg}",
                        )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return LLMResponse(
                content="",
                provider="openrouter",
                model=self.model,
                tokens_used=0,
                cost_estimate=0.0,
                response_time=response_time,
                success=False,
                error=f"Request failed: {str(e)}",
            )

    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost estimate based on token usage."""
        if self.model not in self.cost_per_1k_tokens:
            return 0.0

        costs = self.cost_per_1k_tokens[self.model]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]

        return input_cost + output_cost


class DeepSeekProvider(LLMProviderBase):
    """Deepseek API provider implementation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-chat"

        # Token costs per 1K tokens (approximate)
        self.cost_per_1k_tokens = {
            "deepseek-chat": {"input": 0.0014, "output": 0.0028},
            "deepseek-coder": {"input": 0.0014, "output": 0.0028},
        }

    async def generate(
        self, prompt: str, max_tokens: int = 800, temperature: float = 0.7, **kwargs
    ) -> LLMResponse:
        """Generate content using Deepseek API."""
        start_time = datetime.now()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        content = response_data["choices"][0]["message"]["content"]
                        usage = response_data.get("usage", {})
                        tokens_used = usage.get("total_tokens", 0)

                        # Calculate cost estimate
                        cost_estimate = self._calculate_cost(usage)

                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content=content,
                            provider="deepseek",
                            model=self.model,
                            tokens_used=tokens_used,
                            cost_estimate=cost_estimate,
                            response_time=response_time,
                            success=True,
                            metadata=response_data,
                        )
                    else:
                        error_msg = response_data.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        response_time = (datetime.now() - start_time).total_seconds()

                        return LLMResponse(
                            content="",
                            provider="deepseek",
                            model=self.model,
                            tokens_used=0,
                            cost_estimate=0.0,
                            response_time=response_time,
                            success=False,
                            error=f"API Error: {error_msg}",
                        )

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return LLMResponse(
                content="",
                provider="deepseek",
                model=self.model,
                tokens_used=0,
                cost_estimate=0.0,
                response_time=response_time,
                success=False,
                error=f"Request failed: {str(e)}",
            )

    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost estimate based on token usage."""
        if self.model not in self.cost_per_1k_tokens:
            return 0.0

        costs = self.cost_per_1k_tokens[self.model]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]

        return input_cost + output_cost


class LLMClient:
    """
    Centralized LLM client with retry logic and provider abstraction.
    Supports multiple providers and tracks usage/costs.
    """

    def __init__(self):
        self.provider = self._initialize_provider()
        self.usage_tracker = TokenUsageTracker()
        self.max_retries = 3

    def _initialize_provider(self) -> LLMProviderBase:
        """Initialize the configured LLM provider."""
        if settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAIProvider(settings.openai_api_key)
        elif settings.llm_provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API key is required")
            return AnthropicProvider(settings.anthropic_api_key)
        elif settings.llm_provider == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError("OpenRouter API key is required")
            return OpenRouterProvider(settings.openrouter_api_key)
        elif settings.llm_provider == "deepseek":
            if not settings.deepseek_api_key:
                raise ValueError("Deepseek API key is required")
            return DeepSeekProvider(settings.deepseek_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retry_count: int = 0,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate content with retry logic and usage tracking.

        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            retry_count: Current retry attempt
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content and metadata
        """
        # Use defaults from settings if not provided
        if max_tokens is None:
            max_tokens = settings.max_tokens
        if temperature is None:
            temperature = settings.temperature

        try:
            response = await self.provider.generate(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
            )

            if response.success:
                # Track successful usage
                self.usage_tracker.add_usage(
                    response.tokens_used, response.cost_estimate
                )
                logger.info(
                    f"LLM call successful: {response.tokens_used} tokens, "
                    f"${response.cost_estimate:.4f}, {response.response_time:.2f}s"
                )
            else:
                # Track error
                self.usage_tracker.add_error()
                logger.warning(f"LLM call failed: {response.error}")

                # Retry on certain errors
                if retry_count < self.max_retries and self._should_retry(
                    response.error
                ):
                    wait_time = 2**retry_count  # Exponential backoff
                    logger.info(
                        f"Retrying LLM call in {wait_time}s (attempt {retry_count + 1})"
                    )
                    await asyncio.sleep(wait_time)
                    return await self.generate(
                        prompt, max_tokens, temperature, retry_count + 1, **kwargs
                    )

            return response

        except Exception as e:
            self.usage_tracker.add_error()
            error_msg = f"LLM client error: {str(e)}"
            logger.error(error_msg)

            # Retry on exceptions
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                logger.info(
                    f"Retrying LLM call in {wait_time}s (attempt {retry_count + 1})"
                )
                await asyncio.sleep(wait_time)
                return await self.generate(
                    prompt, max_tokens, temperature, retry_count + 1, **kwargs
                )

            return LLMResponse(
                content="",
                provider=settings.llm_provider,
                model="unknown",
                tokens_used=0,
                cost_estimate=0.0,
                response_time=0.0,
                success=False,
                error=error_msg,
            )

    def _should_retry(self, error: Optional[str]) -> bool:
        """Determine if an error should trigger a retry."""
        if not error:
            return False

        # Retry on rate limiting, temporary server errors
        retry_indicators = [
            "rate limit",
            "timeout",
            "server error",
            "service unavailable",
            "internal error",
            "502",
            "503",
            "504",
        ]

        error_lower = error.lower()
        return any(indicator in error_lower for indicator in retry_indicators)

    async def generate_json(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate JSON content with validation.

        Args:
            prompt: Input prompt (should request JSON output)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response as dictionary

        Raises:
            ValueError: If response is not valid JSON
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."

        response = await self.generate(
            prompt=json_prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

        if not response.success:
            raise ValueError(f"LLM call failed: {response.error}")

        try:
            # Clean up response content (remove markdown code blocks if present)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_tracker.get_stats()

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_tracker = TokenUsageTracker()


# Global LLM client instance - only create if not in test environment
import sys
if 'pytest' not in sys.modules and 'tests' not in sys.argv[0]:
    llm_client = LLMClient()
else:
    # For tests, we'll mock this
    llm_client = None
