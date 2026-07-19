from __future__ import annotations

from typing import Any, Dict

import requests
import tenacity.nap

from mlb_quantitative_engine.api.telegram_api import TelegramApiClient, TelegramApiError


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_send_message_success(monkeypatch) -> None:
    captured = {}

    def fake_post(url: str, json: Dict[str, Any] = None, **kwargs) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"ok": True, "result": {"message_id": 42}})

    monkeypatch.setattr("mlb_quantitative_engine.api.telegram_api.requests.post", fake_post)
    client = TelegramApiClient(bot_token="test-token")

    result = client.send_message("@meucanal", "Olá mundo")

    assert result["ok"] is True
    assert captured["url"] == "https://api.telegram.org/bottest-token/sendMessage"
    assert captured["json"]["chat_id"] == "@meucanal"
    assert captured["json"]["text"] == "Olá mundo"
    assert captured["json"]["parse_mode"] == "HTML"


def test_send_message_raises_on_api_error_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.telegram_api.requests.post",
        lambda *args, **kwargs: _FakeResponse({"ok": False, "description": "chat not found"}),
    )
    client = TelegramApiClient(bot_token="test-token")

    try:
        client.send_message("@canal_inexistente", "oi")
        assert False, "esperava TelegramApiError"
    except TelegramApiError as exc:
        assert "chat not found" in str(exc)


def test_send_message_retries_and_then_raises_on_network_error(monkeypatch) -> None:
    monkeypatch.setattr(tenacity.nap, "sleep", lambda seconds: None)

    call_count = {"count": 0}

    def failing_post(*args: Any, **kwargs: Any) -> None:
        call_count["count"] += 1
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("mlb_quantitative_engine.api.telegram_api.requests.post", failing_post)
    client = TelegramApiClient(bot_token="test-token")

    try:
        client.send_message("@canal", "oi")
        assert False, "esperava TelegramApiError"
    except TelegramApiError:
        pass

    assert call_count["count"] == 3
