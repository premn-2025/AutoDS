"""
LLMMonitor - Monitor LLM-specific metrics for Gemini/Vertex AI
Tracks tokens, costs, latency, and response quality
Uses logging instead of Datadog statsd
"""

import time
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


class LLMMonitor:
    """
    Monitor LLM-specific metrics for observability.
    Tracks Gemini API usage, costs, and performance.
    """
    
    # Pricing per 1K tokens (approximate for Gemini 1.5 Pro)
    PRICING = {
        'gemini-1.5-pro': {
            'input': 0.00125,   # $1.25 per 1M input tokens
            'output': 0.00375   # $3.75 per 1M output tokens
        },
        'gemini-1.5-flash': {
            'input': 0.000125,  # $0.125 per 1M input tokens
            'output': 0.000375  # $0.375 per 1M output tokens
        },
        'gemini-2.0-flash': {
            'input': 0.000125,  # $0.125 per 1M input tokens
            'output': 0.000375  # $0.375 per 1M output tokens
        },
        'default': {
            'input': 0.001,
            'output': 0.003
        }
    }
    
    @staticmethod
    def track_llm_request(
        model: str,
        agent: str,
        prompt_length: int,
        start_time: Optional[float] = None
    ) -> float:
        """
        Track the start of an LLM request.
        
        Args:
            model: Model name
            agent: Agent making the request
            prompt_length: Length of prompt in characters
            start_time: Optional start time
            
        Returns:
            Start time for duration calculation
        """
        if start_time is None:
            start_time = time.time()
        
        logger.info("LLM request started", extra={
            'model': model,
            'agent': agent,
            'prompt_length': prompt_length
        })
        
        return start_time
    
    @staticmethod
    def track_llm_response(
        model: str,
        agent: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Track LLM response metrics.
        
        Args:
            model: Model name
            agent: Agent that made the request
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            latency: Request latency in seconds
            success: Whether request succeeded
            error: Error message if failed
        """
        total_tokens = prompt_tokens + completion_tokens
        tokens_per_second = completion_tokens / latency if latency > 0 else 0
        cost = LLMMonitor.calculate_cost(model, prompt_tokens, completion_tokens)
        
        log_data = {
            'model': model,
            'agent': agent,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'latency_seconds': round(latency, 3),
            'tokens_per_second': round(tokens_per_second, 2),
            'cost_usd': round(cost, 6),
            'success': success
        }
        
        if success:
            logger.info("LLM response received", extra=log_data)
        else:
            log_data['error'] = error[:100] if error else 'unknown'
            logger.error("LLM request failed", extra=log_data)
    
    @staticmethod
    def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate estimated cost for LLM usage.
        
        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        pricing = LLMMonitor.PRICING.get(model, LLMMonitor.PRICING['default'])
        
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
    
    @staticmethod
    def track_token_usage(
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_cost: Optional[float] = None
    ):
        """
        Track token usage and costs.
        
        Args:
            model: Model name
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            total_cost: Optional pre-calculated cost
        """
        if total_cost is None:
            total_cost = LLMMonitor.calculate_cost(model, prompt_tokens, completion_tokens)
        
        logger.info("Token usage tracked", extra={
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_cost_usd': round(total_cost, 6)
        })
    
    @staticmethod
    def track_response_quality(
        model: str,
        response_length: int,
        contains_error: bool = False,
        is_valid_json: bool = True
    ):
        """
        Track LLM response quality indicators.
        
        Args:
            model: Model name
            response_length: Length of response
            contains_error: Whether response contains error indicators
            is_valid_json: Whether response is valid JSON (if expected)
        """
        log_data = {
            'model': model,
            'response_length': response_length,
            'contains_error': contains_error,
            'is_valid_json': is_valid_json
        }
        
        if contains_error or not is_valid_json:
            logger.warning("LLM response quality issue", extra=log_data)
        else:
            logger.debug("LLM response quality OK", extra=log_data)
    
    @staticmethod
    def track_rate_limit(model: str, retry_after: Optional[float] = None):
        """
        Track rate limiting events.
        
        Args:
            model: Model name
            retry_after: Retry-after duration if provided
        """
        logger.warning("LLM rate limit hit", extra={
            'model': model,
            'retry_after_seconds': retry_after
        })
    
    @staticmethod
    def track_cache_hit(model: str, agent: str, hit: bool):
        """
        Track LLM response cache hits/misses.
        
        Args:
            model: Model name
            agent: Agent name
            hit: Whether cache was hit
        """
        logger.debug("LLM cache access", extra={
            'model': model,
            'agent': agent,
            'cache_hit': hit
        })
    
    @staticmethod
    def get_hourly_cost_estimate(
        model: str,
        avg_prompt_tokens: int,
        avg_completion_tokens: int,
        requests_per_hour: int
    ) -> float:
        """
        Estimate hourly LLM costs.
        
        Args:
            model: Model name
            avg_prompt_tokens: Average input tokens per request
            avg_completion_tokens: Average output tokens per request
            requests_per_hour: Expected requests per hour
            
        Returns:
            Estimated hourly cost in USD
        """
        cost_per_request = LLMMonitor.calculate_cost(
            model, avg_prompt_tokens, avg_completion_tokens
        )
        
        return cost_per_request * requests_per_hour

    @staticmethod
    def get_cost_report() -> Dict[str, Any]:
        """
        Get a comprehensive cost and usage report.
        
        Returns:
            Dictionary containing LLM usage metrics and cost estimates
        """
        # Return sample structure - in production this would pull from logs/metrics storage
        return {
            'total_requests': 0,
            'total_tokens': 0,
            'estimated_cost_usd': 0.0,
            'avg_latency_ms': 0,
            'models_used': ['gemini-2.0-flash'],
            'hourly_estimates': {
                'requests': 0,
                'cost_usd': 0.0
            },
            'errors': 0,
            'cache_hit_rate': 0.0
        }
