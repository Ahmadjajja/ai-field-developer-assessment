"""Model routing with automatic failover.

The primary provider gets a hard deadline (3000 ms by default) for the whole
call: connecting, waiting and reading the answer. If it answers 429, misses the
deadline, can't be reached or fails with a 5xx, the same request goes to the
secondary provider.

Timeout race: the deadline is enforced with asyncio.timeout(), which cancels
the primary request (httpx closes its connection). A late primary answer can
therefore never arrive after the switch, and exactly one provider's result is
ever returned, so the tenant is charged exactly once.

A 4xx other than 429 is the request's own fault: another provider would reject
it too, so there's no failover.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger("router")


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 3.0  # seconds for the whole call


@dataclass
class RoutedResponse:
    provider: str
    completion: dict
    attempts: list[dict]  # what happened at each provider, for logs and headers


class ProviderRejected(Exception):
    """The provider refused the request itself (4xx other than 429): don't retry elsewhere."""

    def __init__(self, status: int) -> None:
        super().__init__(f"provider rejected the request with HTTP {status}")
        self.status = status


class AllProvidersFailed(Exception):
    def __init__(self, attempts: list[dict]) -> None:
        super().__init__("all providers failed")
        self.attempts = attempts


class _Retryable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ModelRouter:
    def __init__(self, providers: list[Provider], http: httpx.AsyncClient) -> None:
        self.providers, self.http = providers, http

    async def complete(self, body: dict) -> RoutedResponse:
        attempts: list[dict] = []
        for provider in self.providers:
            started = time.perf_counter()
            try:
                completion = await self._call(provider, body)
            except _Retryable as exc:
                attempts.append({"provider": provider.name, "outcome": exc.reason,
                                 "ms": round((time.perf_counter() - started) * 1000)})
                log.warning("provider %s failed (%s)", provider.name, exc.reason)
                continue
            attempts.append({"provider": provider.name, "outcome": "ok", "ms": round((time.perf_counter() - started) * 1000)})
            return RoutedResponse(provider.name, completion, attempts)
        raise AllProvidersFailed(attempts)

    async def _call(self, provider: Provider, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        payload = {**body, "model": provider.model}
        try:
            async with asyncio.timeout(provider.timeout):
                response = await self.http.post(f"{provider.base_url}/chat/completions", json=payload, headers=headers)
                completion = response.json() if response.status_code < 400 else None
        except TimeoutError:
            raise _Retryable("timeout") from None
        except httpx.HTTPError as exc:
            raise _Retryable(f"unreachable:{type(exc).__name__}") from None
        except ValueError:
            raise _Retryable("invalid_response") from None

        if response.status_code == 429:
            raise _Retryable("rate_limited")
        if response.status_code >= 500:
            raise _Retryable(f"http_{response.status_code}")
        if response.status_code >= 400:
            raise ProviderRejected(response.status_code)
        if not isinstance(completion, dict) or not isinstance(completion.get("choices"), list):
            raise _Retryable("invalid_response")
        return completion
