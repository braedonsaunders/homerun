import asyncio
import random
from functools import wraps
from typing import Awaitable, Callable, Type, Tuple, TypeVar
import httpx

from utils.logger import get_logger

logger = get_logger("retry")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Shared DB retry helpers
# ---------------------------------------------------------------------------

DB_RETRY_ATTEMPTS = 4
DB_RETRY_BASE_DELAY_SECONDS = 0.05
DB_RETRY_MAX_DELAY_SECONDS = 0.4

_DB_RETRYABLE_MARKERS = (
    "deadlock detected",
    "serialization failure",
    "could not serialize access",
    "lock not available",
    "too many clients already",
    "sorry, too many clients already",
    "remaining connection slots are reserved",
    "cannot connect now",
    "connection is closed",
    "underlying connection is closed",
    "connection has been closed",
    "closed the connection unexpectedly",
    "terminating connection",
    "connection reset by peer",
    "broken pipe",
    "connection was closed",
    "connectiondoesnotexist",
    "closed in the middle of operation",
    "another operation",
    "cannot switch to state",
)


def is_retryable_db_error(exc: Exception) -> bool:
    """Check if a DB exception is transient and worth retrying."""
    timeout_types = (TimeoutError, asyncio.TimeoutError)
    if isinstance(exc, timeout_types):
        return False
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, timeout_types):
        return False
    context = getattr(exc, "__context__", None)
    if isinstance(context, timeout_types):
        return False
    message = str(getattr(exc, "orig", exc)).lower()
    return any(marker in message for marker in _DB_RETRYABLE_MARKERS)


def db_retry_delay(
    attempt: int,
    *,
    base_delay: float = DB_RETRY_BASE_DELAY_SECONDS,
    max_delay: float = DB_RETRY_MAX_DELAY_SECONDS,
) -> float:
    """Exponential backoff delay for DB retries."""
    return min(float(base_delay) * (2 ** attempt), float(max_delay))


def is_retryable_db_operation_error(exc: Exception) -> bool:
    return isinstance(exc, asyncio.TimeoutError) or is_retryable_db_error(exc)


async def run_with_db_retries(
    operation: Callable[[int], Awaitable[T]],
    *,
    max_attempts: int = DB_RETRY_ATTEMPTS,
    base_delay: float = DB_RETRY_BASE_DELAY_SECONDS,
    max_delay: float = DB_RETRY_MAX_DELAY_SECONDS,
    is_retryable: Callable[[Exception], bool] = is_retryable_db_operation_error,
) -> T:
    last_error: Exception | None = None
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            return await operation(attempt)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            await asyncio.sleep(db_retry_delay(attempt, base_delay=base_delay, max_delay=max_delay))
    raise RuntimeError("DB retry loop exited without result") from last_error


async def commit_with_retry(
    session,
    *,
    max_attempts: int = DB_RETRY_ATTEMPTS,
    base_delay: float = DB_RETRY_BASE_DELAY_SECONDS,
    max_delay: float = DB_RETRY_MAX_DELAY_SECONDS,
    is_retryable: Callable[[Exception], bool] = is_retryable_db_operation_error,
) -> None:
    async def _commit(_attempt: int) -> None:
        try:
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:
                pass
            raise

    await run_with_db_retries(
        _commit,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        is_retryable=is_retryable,
    )


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ConnectError,
            ConnectionError,
            asyncio.TimeoutError,
        ),
        retryable_status_codes: Tuple[int, ...] = (429, 500, 502, 503, 504),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.retryable_status_codes = retryable_status_codes


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and optional jitter"""
    delay = min(config.base_delay * (config.exponential_base**attempt), config.max_delay)
    if config.jitter:
        delay = delay * (0.5 + random.random())
    return delay


def is_retryable_error(error: Exception, config: RetryConfig) -> bool:
    """Check if an error should be retried"""
    # Check exception type
    if isinstance(error, config.retryable_exceptions):
        return True

    # Check HTTP status code
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in config.retryable_status_codes

    return False


def with_retry(config: RetryConfig = None):
    """Decorator for async functions with retry logic"""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not is_retryable_error(e, config):
                        logger.error(
                            "Non-retryable error",
                            function=func.__name__,
                            error=str(e),
                            error_type=type(e).__name__,
                        )
                        raise

                    if attempt < config.max_attempts - 1:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            "Retrying after error",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=config.max_attempts,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All retry attempts exhausted",
                            function=func.__name__,
                            attempts=config.max_attempts,
                            error=str(e),
                        )

            raise last_error

        return wrapper

    return decorator


class RetryableClient:
    """HTTP client wrapper with automatic retry"""

    def __init__(self, client: httpx.AsyncClient, config: RetryConfig = None):
        self.client = client
        self.config = config or RetryConfig()

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a request with retry logic"""
        last_error = None

        for attempt in range(self.config.max_attempts):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except Exception as e:
                last_error = e

                if not is_retryable_error(e, self.config):
                    raise

                if attempt < self.config.max_attempts - 1:
                    delay = calculate_delay(attempt, self.config)

                    # Special handling for rate limits
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after:
                            delay = max(delay, float(retry_after))

                    logger.warning(
                        "Retrying HTTP request",
                        method=method,
                        url=url,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        raise last_error

    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", url, **kwargs)
