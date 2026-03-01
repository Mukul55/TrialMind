"""
API rate limiting utilities.

Provides token bucket and sliding window rate limiters for:
- PubMed API (3 req/s without key, 10 req/s with key)
- FDA API (1,000/day without key, 120,000/day with key)
- WHO ICTRP (2 req/s conservative)

Also provides exponential backoff decorators for transient failures.
"""

import asyncio
import time
from collections import deque
from functools import wraps
from loguru import logger
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import logging


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for controlling request rates.

    Allows bursting up to max_burst requests, then enforces
    the specified rate.
    """

    def __init__(self, rate: float, max_burst: int = None):
        """
        rate: Maximum requests per second
        max_burst: Maximum burst size (defaults to 2x rate)
        """
        self.rate = rate
        self.max_tokens = max_burst or max(int(rate * 2), 1)
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a token is available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.rate
            )
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return

            # Need to wait
            wait_time = (1.0 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0.0

    def acquire_sync(self):
        """Synchronous version of acquire."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.rate
        )
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return

        wait_time = (1.0 - self.tokens) / self.rate
        time.sleep(wait_time)
        self.tokens = 0.0


class DailyQuotaLimiter:
    """
    Daily quota limiter for APIs with per-day limits (e.g., FDA API).
    Tracks usage within a rolling 24-hour window.
    """

    def __init__(self, daily_limit: int):
        self.daily_limit = daily_limit
        self.requests = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Check if we're within daily quota."""
        async with self._lock:
            now = time.time()
            day_ago = now - 86400  # 24 hours

            # Remove old requests
            while self.requests and self.requests[0] < day_ago:
                self.requests.popleft()

            if len(self.requests) >= self.daily_limit:
                # Calculate wait time until oldest request expires
                oldest = self.requests[0]
                wait_time = oldest - day_ago
                logger.warning(
                    f"Daily quota ({self.daily_limit}) reached. "
                    f"Waiting {wait_time:.0f}s..."
                )
                await asyncio.sleep(wait_time)

            self.requests.append(now)

    @property
    def remaining_quota(self) -> int:
        """Get remaining requests in current 24-hour window."""
        now = time.time()
        day_ago = now - 86400
        active_requests = sum(1 for r in self.requests if r > day_ago)
        return max(0, self.daily_limit - active_requests)


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 30.0
):
    """
    Decorator factory for retry with exponential backoff.
    Handles transient API failures.

    Usage:
        @with_retry(max_attempts=3)
        async def fetch_data():
            ...
    """
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_sync_retry(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 30.0
):
    """Synchronous version of with_retry decorator."""
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-configured limiters for known APIs
PUBMED_LIMITER_NO_KEY = TokenBucketRateLimiter(rate=3, max_burst=5)
PUBMED_LIMITER_WITH_KEY = TokenBucketRateLimiter(rate=10, max_burst=20)
FDA_DAILY_LIMITER_NO_KEY = DailyQuotaLimiter(daily_limit=1000)
FDA_DAILY_LIMITER_WITH_KEY = DailyQuotaLimiter(daily_limit=120000)
WHO_LIMITER = TokenBucketRateLimiter(rate=2, max_burst=3)
