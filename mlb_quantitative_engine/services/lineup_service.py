from __future__ import annotations

"""Descoberta da lineup titular (ou provável) de um time para um jogo específico.

Duas fontes, dependendo de quão perto o jogo está de começar:

1. Lineup oficial (alta confiança): o boxscore da MLB Stats API expõe
   `teams.{home|away}.battingOrder` — populado tipicamente poucas horas antes
   do primeiro arremesso. Quando disponível, o boxscore também já traz
   `seasonStats.batting` embutido para cada jogador, o que evita 9 chamadas de
   API extras (uma por rebatedor) — usamos esse atalho.

2. Lineup provável (confiança baixa, fallback): quando o jogo está longe
   demais no tempo, `battingOrder` vem vazio e os `seasonStats` do boxscore
   vêm zerados (placeholder). Nesse caso caímos para o roster ativo do time
   (`/teams/{id}/roster`), que não traz estatísticas — cada jogador precisa
   ser buscado individualmente via OffenseService.

Fontes externas de lineup provável (Rotowire, Rotogrinders, BaseballPress) e
comparação entre fontes ficam fora do escopo desta etapa.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.utils.logger import log

TeamSide = str  # "home" ou "away"


@dataclass(frozen=True)
class LineupEntry:
    """Um jogador na lineup (titular ou provável)."""

    player_id: int
    batting_order: int  # 1-9 (ordem de rebatida); posição arbitrária quando "probable_roster"
    raw_batting_stats: Optional[Dict[str, Any]] = None  # seasonStats.batting, quando já disponível


@dataclass(frozen=True)
class LineupSnapshot:
    """Lineup normalizada de um time para um jogo, com indicador de confiança."""

    team_side: TeamSide
    source: str  # "official" ou "probable_roster"
    confidence_score: int  # 0-100
    entries: List[LineupEntry] = field(default_factory=list)

    @property
    def player_ids(self) -> List[int]:
        return [entry.player_id for entry in self.entries]

    @property
    def has_embedded_stats(self) -> bool:
        return self.source == "official" and all(e.raw_batting_stats is not None for e in self.entries)


class LineupService:
    """Descobre a lineup titular (ou provável) de um time a partir da MLB Stats API."""

    OFFICIAL_CONFIDENCE = 90
    PROBABLE_ROSTER_CONFIDENCE = 40

    def __init__(self, api_client: Optional[MLBApiClient] = None) -> None:
        self.api_client = api_client or MLBApiClient()

    def get_batting_order(self, game_pk: int, team_side: TeamSide) -> LineupSnapshot:
        """Retorna a lineup oficial se já publicada; caso contrário, cai para o roster ativo."""
        if team_side not in ("home", "away"):
            raise ValueError("team_side deve ser 'home' ou 'away'")

        boxscore = self.api_client.get_boxscore(game_pk)
        team_data = boxscore.get("teams", {}).get(team_side, {})
        raw_batting_order = team_data.get("battingOrder", [])

        if raw_batting_order:
            return self._official_lineup(team_data, team_side, raw_batting_order)

        log.warning(
            f"Lineup oficial indisponível para o jogo {game_pk} ({team_side}); "
            "usando roster ativo como lineup provável"
        )
        return self._probable_lineup_from_roster(team_data, team_side)

    def _official_lineup(
        self, team_data: Dict[str, Any], team_side: TeamSide, raw_batting_order: List[int]
    ) -> LineupSnapshot:
        players = team_data.get("players", {})
        players_by_id = {info["person"]["id"]: info for info in players.values()}

        entries = []
        for index, player_id in enumerate(raw_batting_order):
            info = players_by_id.get(int(player_id), {})
            batting_stats = info.get("seasonStats", {}).get("batting")
            entries.append(
                LineupEntry(player_id=int(player_id), batting_order=index + 1, raw_batting_stats=batting_stats)
            )

        return LineupSnapshot(
            team_side=team_side, source="official", confidence_score=self.OFFICIAL_CONFIDENCE, entries=entries
        )

    def _probable_lineup_from_roster(self, team_data: Dict[str, Any], team_side: TeamSide) -> LineupSnapshot:
        team_id = team_data.get("team", {}).get("id")
        if team_id is None:
            log.error(f"Não foi possível determinar o team_id para montar a lineup provável ({team_side})")
            return LineupSnapshot(team_side=team_side, source="probable_roster", confidence_score=0, entries=[])

        roster = self.api_client.get_team_roster(team_id, roster_type="active")
        position_players = [
            entry for entry in roster if entry.get("position", {}).get("abbreviation") != "P"
        ]

        entries = [
            LineupEntry(player_id=int(entry["person"]["id"]), batting_order=index + 1)
            for index, entry in enumerate(position_players)
        ]
        return LineupSnapshot(
            team_side=team_side,
            source="probable_roster",
            confidence_score=self.PROBABLE_ROSTER_CONFIDENCE,
            entries=entries,
        )
