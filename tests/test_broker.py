from __future__ import annotations

from typing import Any

from sandbox.broker import RequestBroker


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        json_payload: Any | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._json_payload = json_payload

    def iter_content(self, chunk_size: int = 8192) -> Any:
        del chunk_size
        yield self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> Any:
        return self._json_payload


class _FakeHTTP:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []
        self.request_response = _FakeResponse(body=b"{}")
        self.get_response = _FakeResponse(json_payload={"results": []})

    def request(self, **kwargs: Any) -> _FakeResponse:
        self.requests.append(dict(kwargs))
        return self.request_response

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.gets.append({"url": url, **dict(kwargs)})
        return self.get_response


def _broker(
    *,
    credentials: dict[str, dict[str, Any]] | None = None,
    public_policy: dict[str, Any] | None = None,
    web_search: dict[str, Any] | None = None,
) -> tuple[RequestBroker, _FakeHTTP]:
    broker = RequestBroker(
        credentials=credentials or {},
        public_policy=public_policy,
        web_search_config=web_search,
    )
    fake = _FakeHTTP()
    broker._http = fake  # noqa: SLF001
    return broker, fake


def test_broker_http_request_integration_injects_bearer_and_executes() -> None:
    broker, fake = _broker(
        credentials={
            "notion": {
                "auth_type": "bearer",
                "token": "secret-token",
                "allowed_hosts": ["api.notion.com"],
                "allowed_methods": ["POST"],
                "allowed_paths": ["/v1/*"],
                "protected_headers": ["Authorization"],
                "default_headers": {"Notion-Version": "2026-03-11"},
                "max_response_bytes": 2048,
            }
        }
    )
    fake.request_response = _FakeResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        body=b'{"ok":true}',
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "POST",
            "url": "https://api.notion.com/v1/pages",
            "headers": {"Content-Type": "application/json"},
            "body": "{}",
        }
    )

    assert result["success"] is True
    sent = fake.requests[0]
    assert sent["headers"]["Authorization"] == "Bearer secret-token"
    assert sent["headers"]["Notion-Version"] == "2026-03-11"


def test_broker_http_request_denies_protected_header_override() -> None:
    broker, _fake = _broker(
        credentials={
            "github": {
                "auth_type": "header",
                "token": "gh-token",
                "header_name": "Authorization",
                "allowed_hosts": ["api.github.com"],
                "allowed_methods": ["GET"],
                "allowed_paths": ["/repos/*"],
                "protected_headers": ["Authorization"],
                "default_headers": {},
                "max_response_bytes": 2048,
            }
        }
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "github",
            "method": "GET",
            "url": "https://api.github.com/repos/o/r",
            "headers": {"Authorization": "Bearer attacker"},
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "protected headers" in str(result["reason"])


def test_broker_http_request_public_policy_disabled_denies() -> None:
    broker, _fake = _broker(public_policy={"enabled": False})

    result = broker.handle(
        {
            "action": "http_request",
            "method": "GET",
            "url": "https://example.com",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "public requests are disabled" in str(result["reason"])


def test_broker_web_fetch_blocks_ssrf_loopback() -> None:
    broker, _fake = _broker()

    result = broker.handle(
        {
            "action": "web_fetch",
            "url": "http://127.0.0.1:8080/private",
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "ssrf protection blocked private address" in str(result["reason"])


def test_broker_web_search_uses_web_search_integration() -> None:
    broker, fake = _broker(
        credentials={
            "_web_search": {
                "auth_type": "header",
                "token": "brave-key",
                "header_name": "X-Subscription-Token",
                "allowed_hosts": ["api.search.brave.com"],
                "allowed_methods": ["GET"],
                "allowed_paths": ["/*"],
                "protected_headers": ["X-Subscription-Token"],
                "default_headers": {},
                "max_response_bytes": 2048,
            }
        },
        web_search={
            "endpoint": "https://api.search.brave.com/res/v1/web/search",
            "format": "brave",
            "max_results": 10,
        },
    )
    fake.get_response = _FakeResponse(
        json_payload={
            "web": {
                "results": [
                    {
                        "title": "T1",
                        "url": "https://example.com/1",
                        "description": "S1",
                    }
                ]
            }
        }
    )

    result = broker.handle(
        {
            "action": "web_search",
            "query": "solid state batteries",
            "max_results": 1,
        }
    )

    assert result["success"] is True
    assert result["results"] == [
        {
            "title": "T1",
            "url": "https://example.com/1",
            "snippet": "S1",
        }
    ]
    assert fake.gets[0]["headers"]["X-Subscription-Token"] == "brave-key"


def test_broker_web_search_denies_when_integration_missing() -> None:
    broker, _fake = _broker(web_search={"endpoint": "https://api.search.brave.com/res/v1/web/search"})

    result = broker.handle({"action": "web_search", "query": "x"})

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "_web_search" in str(result["reason"])


def test_broker_rate_limit_denies_n_plus_one_request() -> None:
    broker, fake = _broker(
        credentials={
            "github": {
                "auth_type": "header",
                "token": "gh-token",
                "header_name": "Authorization",
                "header_prefix": "token ",
                "allowed_hosts": ["api.github.com"],
                "allowed_methods": ["GET"],
                "allowed_paths": ["/repos/*"],
                "protected_headers": ["Authorization"],
                "default_headers": {},
                "max_response_bytes": 2048,
                "rate_limit": {"requests": 1, "per_seconds": 30},
            }
        }
    )
    fake.request_response = _FakeResponse(
        status_code=200,
        headers={"Content-Type": "application/json"},
        body=b"{}",
    )
    payload: dict[str, Any] = {
        "action": "http_request",
        "integration": "github",
        "method": "GET",
        "url": "https://api.github.com/repos/o/r",
        "headers": {},
        "body": None,
    }

    first = broker.handle(payload)
    second = broker.handle(payload)

    assert first["success"] is True
    assert second["success"] is False
    assert second["error"] == "rate_limited"
