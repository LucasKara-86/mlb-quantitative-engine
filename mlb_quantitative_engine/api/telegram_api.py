from __future__ import annotations

"""Cliente para a Telegram Bot API (https://core.telegram.org/bots/api).

Usado para enviar alertas de Value Bet a um canal/grupo. Diferente dos
clientes de API de dados (MLB, Odds), aqui NÃO há cache — cada chamada deve
de fato enviar a mensagem; reaproveitar uma resposta anterior faria sentido
zero (silenciosamente deixaria de enviar alertas novos).
"""

from typing import Any, Dict, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.utils.logger import log


class TelegramApiError(RuntimeError):
    """Erro ao comunicar com a Telegram Bot API após esgotar as tentativas de retry."""


class TelegramApiClient:
    """Cliente HTTP para a Telegram Bot API."""

    def __init__(self, bot_token: Optional[str] = None) -> None:
        self.bot_token = bot_token or settings.telegram_bot_token
        if not self.bot_token:
            log.warning("TELEGRAM_BOT_TOKEN não configurado — envios ao Telegram vão falhar")

    def _base_url(self) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}"

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _request(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return response

    def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """Envia uma mensagem de texto para um chat/canal. Retorna o payload de resposta da API.

        Levanta TelegramApiError tanto em falha de rede quanto quando a API responde
        com sucesso HTTP mas `ok: false` no corpo (ex.: canal errado, bot sem permissão).
        """
        url = f"{self._base_url()}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}

        try:
            response = self._request(url, payload)
        except requests.RequestException as exc:
            log.error(f"Falha ao enviar mensagem ao Telegram: {exc}")
            raise TelegramApiError(f"Falha ao enviar mensagem: {exc}") from exc

        data = response.json()
        if not data.get("ok"):
            log.error(f"Telegram retornou erro: {data}")
            raise TelegramApiError(f"Telegram retornou erro: {data.get('description')}")
        return data
