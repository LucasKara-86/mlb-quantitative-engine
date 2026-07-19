from __future__ import annotations

"""Normaliza odds brutas da The Odds API em estruturas prontas para comparação com projeções.

Raciocínio:
- Casas diferentes frequentemente oferecem linhas de total (`point`) distintas
  para o mesmo jogo (ex.: 8.5 em uma casa, 9.5 em outra). Comparar apenas o
  preço sem considerar a linha seria comparar mercados diferentes. Por isso
  agrupamos as odds de total por `point` e escolhemos a melhor odd (maior
  preço decimal) dentro de cada linha, não a melhor odd entre linhas
  diferentes.
- Para decidir qual linha de total é "a" linha do mercado (linha de
  consenso), usamos a que tem mais casas oferecendo — uma proxy razoável de
  liquidez/consenso, já que a The Odds API não expõe volume de apostas. Em
  caso de empate (comum quando poucas casas cobrem cada linha), desempatamos
  pela linha mais próxima da MEDIANA de todas as linhas cotadas — evita que
  uma linha isolada e não-representativa (ex.: uma única casa ofertando um
  total muito destoante do resto do mercado) vire "consenso" só por ordem de
  chegada nos dados brutos. Bug real encontrado em produção: sem esse
  desempate, uma linha de Game Total de 11.0 (implausível — o normal em MLB
  é 6.5–10.5) venceu por empate arbitrário de contagem de casas.
- "Melhor odd" (best price) sempre significa o maior preço decimal
  disponível para aquele lado (over/under, ou cada time no moneyline) —
  maior preço decimal = melhor retorno para quem aposta.
"""

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypeVar

from mlb_quantitative_engine.api.odds_api import OddsApiClient
from mlb_quantitative_engine.utils.logger import log

_QuoteT = TypeVar("_QuoteT")


def _pick_consensus_quote(quotes: Sequence[_QuoteT]) -> Optional[_QuoteT]:
    """Escolhe a cotação de maior `bookmaker_count`; empates vão para a mais próxima
    da mediana de todas as `point` cotadas (ver docstring do módulo)."""
    if not quotes:
        return None
    median_point = statistics.median(quote.point for quote in quotes)
    return max(quotes, key=lambda quote: (quote.bookmaker_count, -abs(quote.point - median_point)))


def implied_probability(decimal_odds: float) -> float:
    """Converte uma odd decimal na probabilidade implícita pelo mercado (sem remover o vig)."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


@dataclass(frozen=True)
class TotalsQuote:
    """Melhor odd de Game Total (Over/Under) encontrada entre as casas, para uma linha específica."""

    point: float
    bookmaker_count: int
    over_price: float
    over_bookmaker: str
    under_price: float
    under_bookmaker: str


@dataclass(frozen=True)
class MoneylineQuote:
    """Melhor odd de moneyline (vencedor do jogo) para cada lado."""

    home_price: Optional[float]
    home_bookmaker: Optional[str]
    away_price: Optional[float]
    away_bookmaker: Optional[str]


@dataclass(frozen=True)
class GameOdds:
    """Odds normalizadas de um jogo específico, agregadas de todas as casas retornadas."""

    home_team: str
    away_team: str
    commence_time: str
    totals: List[TotalsQuote]
    moneyline: MoneylineQuote
    event_id: Optional[str] = None

    @property
    def consensus_total(self) -> Optional[TotalsQuote]:
        """Linha de total com mais casas oferecendo (proxy de consenso do mercado);
        empates são resolvidos pela linha mais próxima da mediana (ver _pick_consensus_quote)."""
        return _pick_consensus_quote(self.totals)


@dataclass(frozen=True)
class TeamTotalsQuote:
    """Melhor odd de Team Total (Over/Under) para um time específico, numa linha específica."""

    team: str
    point: float
    bookmaker_count: int
    over_price: float
    over_bookmaker: str
    under_price: float
    under_bookmaker: str


@dataclass(frozen=True)
class GameTeamTotals:
    """Team Totals normalizados dos dois times de um jogo (None quando indisponível para o time)."""

    home: Optional[TeamTotalsQuote]
    away: Optional[TeamTotalsQuote]


class OddsService:
    """Busca e normaliza odds de MLB da The Odds API."""

    def __init__(self, api_client: Optional[OddsApiClient] = None) -> None:
        self.api_client = api_client or OddsApiClient()

    def get_all_game_odds(self) -> List[GameOdds]:
        """Retorna as odds normalizadas de todos os jogos de MLB disponíveis no momento."""
        raw_events = self.api_client.get_mlb_odds()
        return [self._parse_event(event) for event in raw_events]

    def find_game_odds(self, games: List[GameOdds], home_team: str, away_team: str) -> Optional[GameOdds]:
        """Localiza as odds de um jogo específico pelos nomes dos times (case-insensitive)."""
        for game in games:
            if game.home_team.lower() == home_team.lower() and game.away_team.lower() == away_team.lower():
                return game
        log.warning(f"Odds não encontradas para {away_team} @ {home_team}")
        return None

    def get_team_totals(self, event_id: str, home_team: str, away_team: str) -> GameTeamTotals:
        """Busca as odds de Team Total (Over/Under por time) para um evento específico.

        Custa 1 requisição adicional à cota da chave (endpoint por evento) — diferente
        de get_all_game_odds(), que cobre o slate do dia inteiro numa única chamada.
        Use com moderação (uma chamada por jogo que você realmente quer avaliar).
        """
        raw = self.api_client.get_event_odds(event_id, markets="team_totals")
        home_by_point: Dict[float, Dict[str, Any]] = {}
        away_by_point: Dict[float, Dict[str, Any]] = {}

        for bookmaker in raw.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title", bookmaker.get("key", "unknown"))
            for market in bookmaker.get("markets", []):
                if market.get("key") != "team_totals":
                    continue
                for outcome in market.get("outcomes", []):
                    team = outcome.get("description")
                    if team == home_team:
                        self._ingest_totals_outcome(outcome, bookmaker_name, home_by_point)
                    elif team == away_team:
                        self._ingest_totals_outcome(outcome, bookmaker_name, away_by_point)

        return GameTeamTotals(
            home=self._best_team_totals_quote(home_team, home_by_point),
            away=self._best_team_totals_quote(away_team, away_by_point),
        )

    @staticmethod
    def _parse_event(event: Dict[str, Any]) -> GameOdds:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        totals_by_point: Dict[float, Dict[str, Any]] = {}
        best_home_price, best_home_bookmaker = None, None
        best_away_price, best_away_bookmaker = None, None

        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title", bookmaker.get("key", "unknown"))
            for market in bookmaker.get("markets", []):
                if market.get("key") == "totals":
                    OddsService._ingest_totals_market(market, bookmaker_name, totals_by_point)
                elif market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        name, price = outcome.get("name"), outcome.get("price")
                        if name == home_team and (best_home_price is None or price > best_home_price):
                            best_home_price, best_home_bookmaker = price, bookmaker_name
                        elif name == away_team and (best_away_price is None or price > best_away_price):
                            best_away_price, best_away_bookmaker = price, bookmaker_name

        totals = [
            TotalsQuote(
                point=point,
                bookmaker_count=data["bookmaker_count"],
                over_price=data["over_price"],
                over_bookmaker=data["over_bookmaker"],
                under_price=data["under_price"],
                under_bookmaker=data["under_bookmaker"],
            )
            for point, data in totals_by_point.items()
            if data["over_price"] is not None and data["under_price"] is not None
        ]

        return GameOdds(
            home_team=home_team,
            away_team=away_team,
            commence_time=event.get("commence_time", ""),
            totals=totals,
            moneyline=MoneylineQuote(
                home_price=best_home_price,
                home_bookmaker=best_home_bookmaker,
                away_price=best_away_price,
                away_bookmaker=best_away_bookmaker,
            ),
            event_id=event.get("id"),
        )

    @staticmethod
    def _ingest_totals_market(
        market: Dict[str, Any], bookmaker_name: str, totals_by_point: Dict[float, Dict[str, Any]]
    ) -> None:
        for outcome in market.get("outcomes", []):
            OddsService._ingest_totals_outcome(outcome, bookmaker_name, totals_by_point)

    @staticmethod
    def _ingest_totals_outcome(
        outcome: Dict[str, Any], bookmaker_name: str, buckets: Dict[float, Dict[str, Any]]
    ) -> None:
        point = outcome.get("point")
        name, price = outcome.get("name"), outcome.get("price")
        if point is None or price is None:
            return

        bucket = buckets.setdefault(
            point,
            {
                "bookmaker_count": 0,
                "over_price": None,
                "over_bookmaker": None,
                "under_price": None,
                "under_bookmaker": None,
                "_books_seen": set(),
            },
        )
        if bookmaker_name not in bucket["_books_seen"]:
            bucket["_books_seen"].add(bookmaker_name)
            bucket["bookmaker_count"] += 1

        if name == "Over" and (bucket["over_price"] is None or price > bucket["over_price"]):
            bucket["over_price"] = price
            bucket["over_bookmaker"] = bookmaker_name
        elif name == "Under" and (bucket["under_price"] is None or price > bucket["under_price"]):
            bucket["under_price"] = price
            bucket["under_bookmaker"] = bookmaker_name

    @staticmethod
    def _best_team_totals_quote(team: str, buckets: Dict[float, Dict[str, Any]]) -> Optional[TeamTotalsQuote]:
        candidates = [
            TeamTotalsQuote(
                team=team,
                point=point,
                bookmaker_count=data["bookmaker_count"],
                over_price=data["over_price"],
                over_bookmaker=data["over_bookmaker"],
                under_price=data["under_price"],
                under_bookmaker=data["under_bookmaker"],
            )
            for point, data in buckets.items()
            if data["over_price"] is not None and data["under_price"] is not None
        ]
        return _pick_consensus_quote(candidates)
