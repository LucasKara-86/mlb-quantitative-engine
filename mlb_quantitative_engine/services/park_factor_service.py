from __future__ import annotations

"""Park factors por estádio da MLB.

A MLB Stats API não expõe park factors calculados — são um produto derivado
de sabermetria (ex.: Baseball Savant, FanGraphs), não um dado bruto do
schedule/boxscore. Os valores abaixo são aproximações razoáveis baseadas em
características publicamente conhecidas de cada parque (altitude, dimensões,
orientação, tipo de superfície) e devem ser substituídos por um provedor de
park factors oficial quando uma integração de dados dedicada for construída
— mesmo padrão de "placeholder documentado" já usado em LeagueConstants
(analytics/sabermetrics.py).

Convenção: fator 1.00 = neutro (produção igual à média da liga). Valores
acima de 1.00 favorecem o ataque; abaixo de 1.00 favorecem o arremesso.

Os nomes de venue usados como chave foram obtidos diretamente da MLB Stats
API (/teams) para a temporada corrente, não digitados de memória — vários
estádios mudaram de nome por patrocínio (ex.: "Rate Field", "Daikin Park",
"UNIQLO Field at Dodger Stadium") e um time (Athletics) joga em um estádio
temporário ("Sutter Health Park") com histórico de dados MLB muito limitado.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from mlb_quantitative_engine.utils.logger import log


@dataclass(frozen=True)
class ParkFactors:
    """Fatores de ajuste de um estádio específico sobre a produção ofensiva."""

    venue: str
    run_factor: float
    hr_factor: float
    single_factor: float
    double_factor: float
    triple_factor: float
    lhb_factor: float  # fator para rebatedores canhotos
    rhb_factor: float  # fator para rebatedores destros
    altitude_ft: int
    turf_type: str  # "grass" ou "turf"
    foul_territory: str  # "small", "average" ou "large"
    left_field_ft: int
    center_field_ft: int
    right_field_ft: int


NEUTRAL_PARK_FACTORS = ParkFactors(
    venue="Neutro (estádio desconhecido)",
    run_factor=1.00, hr_factor=1.00, single_factor=1.00, double_factor=1.00, triple_factor=1.00,
    lhb_factor=1.00, rhb_factor=1.00, altitude_ft=500, turf_type="grass", foul_territory="average",
    left_field_ft=330, center_field_ft=400, right_field_ft=330,
)


_PARK_FACTORS: Dict[str, ParkFactors] = {
    "Chase Field": ParkFactors(
        "Chase Field", run_factor=1.04, hr_factor=1.05, single_factor=1.02, double_factor=1.03,
        triple_factor=1.10, lhb_factor=1.03, rhb_factor=1.05, altitude_ft=1100, turf_type="grass",
        foul_territory="average", left_field_ft=330, center_field_ft=407, right_field_ft=334,
    ),
    "Sutter Health Park": ParkFactors(
        # Estádio temporário do Athletics (parque de Triple-A); histórico de dados MLB muito
        # limitado, valores tratados com incerteza maior que os demais.
        "Sutter Health Park", run_factor=1.08, hr_factor=1.10, single_factor=1.03, double_factor=1.05,
        triple_factor=0.90, lhb_factor=1.05, rhb_factor=1.10, altitude_ft=30, turf_type="grass",
        foul_territory="small", left_field_ft=330, center_field_ft=403, right_field_ft=325,
    ),
    "Truist Park": ParkFactors(
        "Truist Park", run_factor=1.02, hr_factor=1.03, single_factor=1.00, double_factor=1.02,
        triple_factor=1.05, lhb_factor=1.02, rhb_factor=1.03, altitude_ft=1050, turf_type="grass",
        foul_territory="average", left_field_ft=335, center_field_ft=400, right_field_ft=325,
    ),
    "Oriole Park at Camden Yards": ParkFactors(
        "Oriole Park at Camden Yards", run_factor=0.97, hr_factor=0.90, single_factor=1.00, double_factor=0.98,
        triple_factor=1.05, lhb_factor=0.85, rhb_factor=1.00, altitude_ft=20, turf_type="grass",
        foul_territory="average", left_field_ft=333, center_field_ft=400, right_field_ft=318,
    ),
    "Fenway Park": ParkFactors(
        "Fenway Park", run_factor=1.05, hr_factor=1.00, single_factor=1.00, double_factor=1.25,
        triple_factor=0.75, lhb_factor=0.95, rhb_factor=1.05, altitude_ft=20, turf_type="grass",
        foul_territory="small", left_field_ft=310, center_field_ft=390, right_field_ft=302,
    ),
    "Wrigley Field": ParkFactors(
        # Fortemente dependente do vento; valores refletem a média da temporada.
        "Wrigley Field", run_factor=1.02, hr_factor=1.03, single_factor=1.00, double_factor=1.00,
        triple_factor=1.05, lhb_factor=1.00, rhb_factor=1.02, altitude_ft=600, turf_type="grass",
        foul_territory="average", left_field_ft=355, center_field_ft=400, right_field_ft=353,
    ),
    "Rate Field": ParkFactors(
        "Rate Field", run_factor=1.05, hr_factor=1.10, single_factor=1.00, double_factor=1.02,
        triple_factor=0.95, lhb_factor=1.05, rhb_factor=1.08, altitude_ft=595, turf_type="grass",
        foul_territory="average", left_field_ft=330, center_field_ft=400, right_field_ft=335,
    ),
    "Great American Ball Park": ParkFactors(
        "Great American Ball Park", run_factor=1.08, hr_factor=1.15, single_factor=1.00, double_factor=1.00,
        triple_factor=0.90, lhb_factor=1.10, rhb_factor=1.12, altitude_ft=550, turf_type="grass",
        foul_territory="small", left_field_ft=328, center_field_ft=404, right_field_ft=325,
    ),
    "Progressive Field": ParkFactors(
        "Progressive Field", run_factor=0.99, hr_factor=0.98, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=0.98, rhb_factor=0.99, altitude_ft=650, turf_type="grass",
        foul_territory="average", left_field_ft=325, center_field_ft=405, right_field_ft=325,
    ),
    "Coors Field": ParkFactors(
        # Altitude extrema (~1.6km) reduz o atrito do ar: bolas voam mais, breaking balls
        # quebram menos. O park factor mais alto da MLB por uma margem considerável.
        "Coors Field", run_factor=1.17, hr_factor=1.15, single_factor=1.08, double_factor=1.15,
        triple_factor=1.60, lhb_factor=1.15, rhb_factor=1.15, altitude_ft=5200, turf_type="grass",
        foul_territory="average", left_field_ft=347, center_field_ft=415, right_field_ft=350,
    ),
    "Comerica Park": ParkFactors(
        "Comerica Park", run_factor=0.97, hr_factor=0.90, single_factor=1.00, double_factor=1.02,
        triple_factor=1.10, lhb_factor=0.92, rhb_factor=0.95, altitude_ft=600, turf_type="grass",
        foul_territory="average", left_field_ft=345, center_field_ft=420, right_field_ft=330,
    ),
    "Daikin Park": ParkFactors(
        "Daikin Park", run_factor=1.04, hr_factor=1.08, single_factor=0.98, double_factor=1.02,
        triple_factor=0.85, lhb_factor=1.05, rhb_factor=1.10, altitude_ft=50, turf_type="grass",
        foul_territory="small", left_field_ft=315, center_field_ft=409, right_field_ft=326,
    ),
    "Kauffman Stadium": ParkFactors(
        "Kauffman Stadium", run_factor=0.96, hr_factor=0.90, single_factor=1.02, double_factor=1.00,
        triple_factor=1.15, lhb_factor=0.93, rhb_factor=0.95, altitude_ft=750, turf_type="grass",
        foul_territory="average", left_field_ft=330, center_field_ft=410, right_field_ft=330,
    ),
    "Angel Stadium": ParkFactors(
        "Angel Stadium", run_factor=1.00, hr_factor=0.98, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=0.98, rhb_factor=1.00, altitude_ft=160, turf_type="grass",
        foul_territory="average", left_field_ft=330, center_field_ft=396, right_field_ft=330,
    ),
    "UNIQLO Field at Dodger Stadium": ParkFactors(
        "UNIQLO Field at Dodger Stadium", run_factor=0.96, hr_factor=0.95, single_factor=1.00, double_factor=0.98,
        triple_factor=0.95, lhb_factor=0.93, rhb_factor=0.97, altitude_ft=340, turf_type="grass",
        foul_territory="large", left_field_ft=330, center_field_ft=395, right_field_ft=330,
    ),
    "loanDepot park": ParkFactors(
        "loanDepot park", run_factor=0.93, hr_factor=0.88, single_factor=1.00, double_factor=0.98,
        triple_factor=1.05, lhb_factor=0.90, rhb_factor=0.90, altitude_ft=10, turf_type="grass",
        foul_territory="large", left_field_ft=344, center_field_ft=400, right_field_ft=335,
    ),
    "American Family Field": ParkFactors(
        "American Family Field", run_factor=1.01, hr_factor=1.02, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=1.00, rhb_factor=1.02, altitude_ft=635, turf_type="grass",
        foul_territory="average", left_field_ft=344, center_field_ft=400, right_field_ft=345,
    ),
    "Target Field": ParkFactors(
        "Target Field", run_factor=0.98, hr_factor=0.97, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=0.97, rhb_factor=0.98, altitude_ft=815, turf_type="grass",
        foul_territory="average", left_field_ft=339, center_field_ft=404, right_field_ft=328,
    ),
    "Citi Field": ParkFactors(
        "Citi Field", run_factor=0.97, hr_factor=0.93, single_factor=1.00, double_factor=1.00,
        triple_factor=1.05, lhb_factor=0.93, rhb_factor=0.97, altitude_ft=20, turf_type="grass",
        foul_territory="average", left_field_ft=335, center_field_ft=408, right_field_ft=330,
    ),
    "Yankee Stadium": ParkFactors(
        # Right field curto favorece fortemente home runs de rebatedores canhotos.
        "Yankee Stadium", run_factor=1.06, hr_factor=1.15, single_factor=0.98, double_factor=1.00,
        triple_factor=0.85, lhb_factor=1.20, rhb_factor=1.02, altitude_ft=55, turf_type="grass",
        foul_territory="small", left_field_ft=318, center_field_ft=408, right_field_ft=314,
    ),
    "Citizens Bank Park": ParkFactors(
        "Citizens Bank Park", run_factor=1.08, hr_factor=1.12, single_factor=1.00, double_factor=1.02,
        triple_factor=0.95, lhb_factor=1.08, rhb_factor=1.10, altitude_ft=20, turf_type="grass",
        foul_territory="small", left_field_ft=329, center_field_ft=401, right_field_ft=330,
    ),
    "PNC Park": ParkFactors(
        "PNC Park", run_factor=0.95, hr_factor=0.88, single_factor=1.02, double_factor=1.00,
        triple_factor=1.10, lhb_factor=1.00, rhb_factor=0.85, altitude_ft=730, turf_type="grass",
        foul_territory="average", left_field_ft=325, center_field_ft=399, right_field_ft=320,
    ),
    "Petco Park": ParkFactors(
        "Petco Park", run_factor=0.94, hr_factor=0.90, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=0.92, rhb_factor=0.93, altitude_ft=50, turf_type="grass",
        foul_territory="average", left_field_ft=336, center_field_ft=396, right_field_ft=322,
    ),
    "Oracle Park": ParkFactors(
        # Camada marinha (marine layer) e right-center profundo suprimem HR fortemente,
        # sobretudo para destros. O park factor mais baixo da MLB.
        "Oracle Park", run_factor=0.92, hr_factor=0.82, single_factor=1.02, double_factor=1.00,
        triple_factor=1.10, lhb_factor=0.85, rhb_factor=0.90, altitude_ft=10, turf_type="grass",
        foul_territory="large", left_field_ft=339, center_field_ft=399, right_field_ft=309,
    ),
    "T-Mobile Park": ParkFactors(
        "T-Mobile Park", run_factor=0.93, hr_factor=0.90, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=0.90, rhb_factor=0.92, altitude_ft=10, turf_type="grass",
        foul_territory="average", left_field_ft=331, center_field_ft=401, right_field_ft=326,
    ),
    "Busch Stadium": ParkFactors(
        "Busch Stadium", run_factor=0.95, hr_factor=0.90, single_factor=1.02, double_factor=1.00,
        triple_factor=1.05, lhb_factor=0.93, rhb_factor=0.95, altitude_ft=465, turf_type="grass",
        foul_territory="average", left_field_ft=336, center_field_ft=400, right_field_ft=335,
    ),
    "Tropicana Field": ParkFactors(
        "Tropicana Field", run_factor=0.97, hr_factor=0.95, single_factor=1.00, double_factor=1.00,
        triple_factor=0.90, lhb_factor=0.95, rhb_factor=0.97, altitude_ft=45, turf_type="turf",
        foul_territory="large", left_field_ft=315, center_field_ft=404, right_field_ft=322,
    ),
    "Globe Life Field": ParkFactors(
        "Globe Life Field", run_factor=1.00, hr_factor=1.00, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=1.00, rhb_factor=1.00, altitude_ft=550, turf_type="grass",
        foul_territory="average", left_field_ft=329, center_field_ft=407, right_field_ft=326,
    ),
    "Rogers Centre": ParkFactors(
        "Rogers Centre", run_factor=1.01, hr_factor=1.02, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=1.00, rhb_factor=1.02, altitude_ft=300, turf_type="turf",
        foul_territory="average", left_field_ft=328, center_field_ft=400, right_field_ft=328,
    ),
    "Nationals Park": ParkFactors(
        "Nationals Park", run_factor=1.00, hr_factor=1.00, single_factor=1.00, double_factor=1.00,
        triple_factor=1.00, lhb_factor=1.00, rhb_factor=1.00, altitude_ft=25, turf_type="grass",
        foul_territory="average", left_field_ft=336, center_field_ft=402, right_field_ft=335,
    ),
}


class ParkFactorService:
    """Consulta os fatores de ajuste de um estádio a partir do nome do venue (MLB Stats API)."""

    def get_park_factors(self, venue: Optional[str]) -> ParkFactors:
        """Retorna os park factors do estádio, ou um fallback neutro se desconhecido."""
        if not venue:
            log.warning("Nome do estádio não informado; usando park factor neutro")
            return NEUTRAL_PARK_FACTORS

        factors = _PARK_FACTORS.get(venue)
        if factors is None:
            log.warning(f"Park factor não cadastrado para '{venue}'; usando fator neutro")
            return NEUTRAL_PARK_FACTORS
        return factors
