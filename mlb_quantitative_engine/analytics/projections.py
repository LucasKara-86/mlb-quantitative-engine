from __future__ import annotations

"""Motor de projeção de corridas esperadas — primeira camada do modelo híbrido
previsto na especificação (Poisson -> Monte Carlo -> Regressão -> Ensemble).

Raciocínio estatístico:
- Cada time recebe um "fator ofensivo" = wRC+ do time / 100. wRC+ já é
  normalizado pela liga (100 = média), então um time com wRC+ 110 é tratado
  como ~10% mais produtivo ofensivamente que a média da liga.
- O time adversário recebe um "fator de prevenção de corridas" a partir de um
  FIP EFETIVO que mistura o titular e o bullpen, ponderado pelas innings que
  cada um deve cobrir (ver _effective_pitching_fip). Usa-se FIP (em vez de
  ERA bruta) porque isola o que o arremessador realmente controla, e por
  construção já fica na mesma escala da ERA (ver fip_constant em sabermetrics.py).
- Corridas esperadas do time = Corridas Médias da Liga * Fator Ofensivo *
  Fator de Pitching Adversário * Park Factor * Fator Climático. Isso é uma
  extensão contínua do método Log5 de Bill James para combinar duas forças
  opostas (ataque vs. arremesso) em uma taxa esperada — abordagem padrão em
  modelos públicos de projeção de corridas.
- Park factor e fator climático: park factor já é real (park_factor_service);
  clima segue neutro (1.0) até weather_service existir.

Integração do bullpen (ver _effective_pitching_fip):
- Um titular cobre, em média, STARTER_EXPECTED_INNINGS innings; o bullpen
  cobre o restante até completar as 9 innings do jogo. O FIP efetivo do time
  é a média do FIP do titular e do FIP do bullpen, ponderada por essas innings.
- O FIP do bullpen é penalizado (piora) proporcionalmente ao Bullpen Fatigue
  Index e à quantidade de arremessadores prováveis indisponíveis — ambos
  índices PRÓPRIOS deste projeto (bullpen_service.py), não métricas
  sabermétricas publicadas. A penalidade é um multiplicador transparente e
  calibrável (BULLPEN_FATIGUE_IMPACT, BULLPEN_UNAVAILABLE_IMPACT), não uma
  fórmula com validação empírica própria — deve ser revisitada quando o
  módulo de calibração existir.
- Quando não há dados de bullpen disponíveis (bullpen_fip=None), o modelo
  usa apenas o FIP do titular, preservando o comportamento anterior.

Encolhimento Bayesiano de amostra pequena (ver sabermetrics.shrink_toward_league_average):
- wRC+ do time e FIP do titular vêm de amostras de temporada que variam muito de
  tamanho (um titular com 15 IP vs. outro com 120 IP; uma lineup toda com poucos PA
  no início da temporada). Tratar os dois como igualmente confiáveis faz o motor
  reagir a ruído de amostra pequena como se fosse sinal real — foi exatamente esse
  tipo de erro que produziu, num caso real de produção, uma projeção de 7.76 corridas
  para um time que terminou zerado. Antes de entrar na fórmula de corridas esperadas,
  wrc_plus e starter_fip são encolhidos em direção à média da liga (100 e
  league_avg_era, respectivamente), proporcionalmente ao tamanho da amostra
  (`plate_appearances`/`starter_innings_pitched`, opcionais — quando ausentes, o
  encolhimento é pulado e o valor bruto é usado, preservando compatibilidade).
- PA_STABILIZATION_POINT (250) e IP_STABILIZATION_POINT (70) são aproximações da
  literatura pública de sabermetria sobre onde essas métricas começam a ficar
  confiáveis — não calibradas com dados deste projeto ainda (ver analytics/calibration.py).

Limitação conhecida: mesmo com o encolhimento acima, este é um modelo de médias —
a incerteza que SOBRA na média (mesmo após encolher) e a sobre-dispersão de
corridas por jogo são tratadas nas camadas seguintes (analytics/poisson.py com
Binomial Negativa, analytics/monte_carlo.py com incerteza de parâmetro).
"""

from dataclasses import dataclass
from typing import Optional

from mlb_quantitative_engine.analytics.sabermetrics import LeagueConstants, shrink_toward_league_average
from mlb_quantitative_engine.models.game_projection import GameProjection


@dataclass(frozen=True)
class TeamOffenseInput:
    """Força ofensiva agregada de um time para uma partida.

    `plate_appearances`, quando informado, ativa o encolhimento bayesiano de
    `wrc_plus` em direção à média da liga (ver docstring do módulo)."""

    wrc_plus: float
    plate_appearances: Optional[int] = None


@dataclass(frozen=True)
class OpposingPitcherInput:
    """Qualidade da equipe de arremesso adversária: titular e, opcionalmente, o bullpen de apoio.

    `starter_innings_pitched`, quando informado, ativa o encolhimento bayesiano de
    `starter_fip` em direção à média da liga (ver docstring do módulo) — aplicado só
    ao titular, já que o FIP do bullpen agregado já mistura vários arremessadores."""

    starter_fip: float
    bullpen_fip: Optional[float] = None
    bullpen_fatigue_index: float = 0.0
    bullpen_unavailable_count: int = 0
    starter_innings_pitched: Optional[float] = None


class ProjectionEngine:
    """Projeta corridas esperadas de cada time combinando ataque e pitching adversário."""

    MIN_PROJECTED_RUNS: float = 0.1

    # Innings médias cobertas pelo titular na MLB moderna; o restante até 9 fica com o bullpen.
    STARTER_EXPECTED_INNINGS: float = 5.33
    GAME_INNINGS: float = 9.0

    # Penalidades sobre o FIP do bullpen (índice próprio — ver bullpen_service.py e docstring do módulo).
    BULLPEN_FATIGUE_IMPACT: float = 0.15
    BULLPEN_UNAVAILABLE_IMPACT: float = 0.03

    # Pontos de estabilização do encolhimento bayesiano (ver docstring do módulo e
    # sabermetrics.shrink_toward_league_average) — aproximações da literatura pública.
    PA_STABILIZATION_POINT: float = 250.0
    IP_STABILIZATION_POINT: float = 70.0

    def __init__(
        self,
        constants: LeagueConstants = LeagueConstants(),
        starter_expected_innings: Optional[float] = None,
        bullpen_fatigue_impact: Optional[float] = None,
        bullpen_unavailable_impact: Optional[float] = None,
        pa_stabilization_point: Optional[float] = None,
        ip_stabilization_point: Optional[float] = None,
    ) -> None:
        self.constants = constants
        # Overrides opcionais das constantes de classe acima (mesmos nomes, sombreando só
        # nesta instância) -- permite a orquestração (report_generator.py) injetar os
        # valores de config.settings sem duplicar a lógica interna, que já referencia
        # self.STARTER_EXPECTED_INNINGS etc. Acesso via classe (ProjectionEngine.X)
        # continua devolvendo o default de literatura, inalterado.
        if starter_expected_innings is not None:
            self.STARTER_EXPECTED_INNINGS = starter_expected_innings
        if bullpen_fatigue_impact is not None:
            self.BULLPEN_FATIGUE_IMPACT = bullpen_fatigue_impact
        if bullpen_unavailable_impact is not None:
            self.BULLPEN_UNAVAILABLE_IMPACT = bullpen_unavailable_impact
        if pa_stabilization_point is not None:
            self.PA_STABILIZATION_POINT = pa_stabilization_point
        if ip_stabilization_point is not None:
            self.IP_STABILIZATION_POINT = ip_stabilization_point

    def _shrunk_wrc_plus(self, offense: TeamOffenseInput) -> float:
        if offense.plate_appearances is None:
            return offense.wrc_plus
        return shrink_toward_league_average(
            offense.wrc_plus, 100.0, offense.plate_appearances, self.PA_STABILIZATION_POINT
        )

    def _shrunk_starter_fip(self, pitcher: OpposingPitcherInput) -> float:
        if pitcher.starter_innings_pitched is None:
            return pitcher.starter_fip
        return shrink_toward_league_average(
            pitcher.starter_fip, self.constants.league_avg_era, pitcher.starter_innings_pitched,
            self.IP_STABILIZATION_POINT,
        )

    def _effective_pitching_fip(self, pitcher: OpposingPitcherInput) -> float:
        """Combina titular e bullpen num FIP único, ponderado pelas innings de cada um."""
        starter_fip = self._shrunk_starter_fip(pitcher)
        if pitcher.bullpen_fip is None:
            return starter_fip

        bullpen_innings = self.GAME_INNINGS - self.STARTER_EXPECTED_INNINGS
        fatigue_penalty = (pitcher.bullpen_fatigue_index / 100.0) * self.BULLPEN_FATIGUE_IMPACT
        unavailable_penalty = pitcher.bullpen_unavailable_count * self.BULLPEN_UNAVAILABLE_IMPACT
        effective_bullpen_fip = pitcher.bullpen_fip * (1.0 + fatigue_penalty + unavailable_penalty)

        blended = (
            starter_fip * self.STARTER_EXPECTED_INNINGS + effective_bullpen_fip * bullpen_innings
        ) / self.GAME_INNINGS
        return blended

    def _team_expected_runs(
        self,
        offense: TeamOffenseInput,
        opposing_pitcher: OpposingPitcherInput,
        park_factor: float,
        weather_factor: float,
    ) -> float:
        offense_factor = self._shrunk_wrc_plus(offense) / 100.0
        effective_fip = self._effective_pitching_fip(opposing_pitcher)
        pitching_factor = effective_fip / self.constants.league_avg_era
        expected = (
            self.constants.league_avg_runs_per_game
            * offense_factor
            * pitching_factor
            * park_factor
            * weather_factor
        )
        return max(expected, self.MIN_PROJECTED_RUNS)

    def project_game(
        self,
        home_team: str,
        away_team: str,
        home_offense: TeamOffenseInput,
        away_offense: TeamOffenseInput,
        home_starting_pitcher: OpposingPitcherInput,
        away_starting_pitcher: OpposingPitcherInput,
        park_factor: float = 1.0,
        weather_factor: float = 1.0,
    ) -> GameProjection:
        """Projeta corridas esperadas para os dois times de uma partida.

        `home_starting_pitcher` representa a equipe de arremesso da casa (titular +
        bullpen), que enfrenta `away_offense` (o time visitante bate contra ela) — e
        vice-versa.
        """
        projected_away_runs = self._team_expected_runs(
            away_offense, home_starting_pitcher, park_factor, weather_factor
        )
        projected_home_runs = self._team_expected_runs(
            home_offense, away_starting_pitcher, park_factor, weather_factor
        )
        return GameProjection(
            home_team=home_team,
            away_team=away_team,
            projected_home_runs=round(projected_home_runs, 2),
            projected_away_runs=round(projected_away_runs, 2),
            projected_total_runs=round(projected_home_runs + projected_away_runs, 2),
            park_factor=park_factor,
            weather_factor=weather_factor,
        )
