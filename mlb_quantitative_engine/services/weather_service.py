from __future__ import annotations

"""Converte previsão do tempo bruta (Open-Meteo) num fator multiplicativo sobre a
projeção de corridas esperadas (ver analytics/projections.py, parâmetro `weather_factor`,
neutro em 1.0 até este módulo existir).

Raciocínio estatístico:
- Temperatura é o efeito climático mais robusto e mais citado na literatura pública de
  sabermetria: ar mais quente é menos denso, a bola viaja mais longe (mais HR/XBH).
  Estimativas públicas (ex.: análises de FanGraphs/Baseball Prospectus sobre "temperature
  and offense") apontam algo da ordem de +2% a +3% de produção ofensiva em dias muito
  quentes (~90°F) vs. frios (~50°F), comparado a uma referência de ~70°F.
  `TEMPERATURE_SENSITIVITY` é calibrado pra reproduzir essa ordem de grandeza -- é um
  valor INICIAL de literatura pública, não uma medição própria deste projeto, e deve
  ser recalibrado pelo harness de calibração (analytics/calibration.py) assim que
  houver amostra suficiente de jogos ao ar livre com clima variado.
- Vento: a direção do vento (a favor ou contra o campo) importa mais para o resultado
  do que a velocidade pura -- mas calcular isso exige saber a orientação de home plate
  de cada estádio (de onde sopra "para dentro" vs. "para fora"), dado que não temos
  hoje. Por isso a velocidade/direção do vento são registradas em `WeatherConditions`
  (para auditoria e uma etapa futura), mas NÃO entram no fator aplicado à projeção --
  melhor não ajustar do que ajustar com premissa errada sobre a direção do efeito.
- Estádios com teto fixo ou retrátil recebem fator sempre neutro (1.0): o clima externo
  não afeta o jogo dentro de um domo fechado, e a MLB tipicamente fecha o teto quando as
  condições seriam adversas -- justamente para neutralizar esse efeito.
- O fator final é limitado a [0.90, 1.10]: mesmo em condições extremas, não deixamos uma
  previsão de tempo (que também tem incerteza própria) dominar a projeção sozinha.
- Falhas (API fora do ar, estádio desconhecido, sem horário do jogo) sempre caem para
  fator neutro (1.0) e nunca impedem a projeção de rodar -- clima é um ajuste
  incremental, mesmo padrão de degradação graciosa já usado em bullpen/park factor.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from mlb_quantitative_engine.api.weather_api import WeatherApiClient, WeatherApiError
from mlb_quantitative_engine.utils.logger import log

TEMPERATURE_BASELINE_F: float = 70.0
TEMPERATURE_SENSITIVITY: float = 0.0015  # fração de ajuste no fator por °F de distância da referência
MIN_WEATHER_FACTOR: float = 0.90
MAX_WEATHER_FACTOR: float = 1.10


@dataclass(frozen=True)
class VenueLocation:
    """Coordenadas de um estádio e se o teto (fixo ou retrátil) neutraliza o clima externo."""

    latitude: float
    longitude: float
    climate_controlled: bool


@dataclass(frozen=True)
class WeatherConditions:
    """Condições climáticas usadas (ou não) no ajuste, expostas para transparência/auditoria."""

    venue: str
    temperature_f: Optional[float]
    wind_speed_mph: Optional[float]
    wind_direction_deg: Optional[float]
    climate_controlled: bool
    factor: float


# Coordenadas públicas dos 30 estádios da MLB (mesmos nomes de venue usados em
# park_factor_service.py, obtidos da MLB Stats API). `climate_controlled=True` para
# domo fixo (Tropicana Field) ou teto retrátil (os demais listados como True).
_VENUE_LOCATIONS: Dict[str, VenueLocation] = {
    "Chase Field": VenueLocation(33.4455, -112.0667, climate_controlled=True),
    "Sutter Health Park": VenueLocation(38.5805, -121.5133, climate_controlled=False),
    "Truist Park": VenueLocation(33.8908, -84.4678, climate_controlled=False),
    "Oriole Park at Camden Yards": VenueLocation(39.2838, -76.6217, climate_controlled=False),
    "Fenway Park": VenueLocation(42.3467, -71.0972, climate_controlled=False),
    "Wrigley Field": VenueLocation(41.9484, -87.6553, climate_controlled=False),
    "Rate Field": VenueLocation(41.8299, -87.6338, climate_controlled=False),
    "Great American Ball Park": VenueLocation(39.0979, -84.5082, climate_controlled=False),
    "Progressive Field": VenueLocation(41.4962, -81.6852, climate_controlled=False),
    "Coors Field": VenueLocation(39.7559, -104.9942, climate_controlled=False),
    "Comerica Park": VenueLocation(42.3390, -83.0485, climate_controlled=False),
    "Daikin Park": VenueLocation(29.7573, -95.3555, climate_controlled=True),
    "Kauffman Stadium": VenueLocation(39.0517, -94.4803, climate_controlled=False),
    "Angel Stadium": VenueLocation(33.8003, -117.8827, climate_controlled=False),
    "UNIQLO Field at Dodger Stadium": VenueLocation(34.0739, -118.2400, climate_controlled=False),
    "loanDepot park": VenueLocation(25.7781, -80.2196, climate_controlled=True),
    "American Family Field": VenueLocation(43.0280, -87.9712, climate_controlled=True),
    "Target Field": VenueLocation(44.9817, -93.2776, climate_controlled=False),
    "Citi Field": VenueLocation(40.7571, -73.8458, climate_controlled=False),
    "Yankee Stadium": VenueLocation(40.8296, -73.9262, climate_controlled=False),
    "Citizens Bank Park": VenueLocation(39.9061, -75.1665, climate_controlled=False),
    "PNC Park": VenueLocation(40.4469, -80.0057, climate_controlled=False),
    "Petco Park": VenueLocation(32.7073, -117.1566, climate_controlled=False),
    "Oracle Park": VenueLocation(37.7786, -122.3893, climate_controlled=False),
    "T-Mobile Park": VenueLocation(47.5914, -122.3325, climate_controlled=True),
    "Busch Stadium": VenueLocation(38.6226, -90.1928, climate_controlled=False),
    "Tropicana Field": VenueLocation(27.7683, -82.6534, climate_controlled=True),
    "Globe Life Field": VenueLocation(32.7473, -97.0842, climate_controlled=True),
    "Rogers Centre": VenueLocation(43.6414, -79.3894, climate_controlled=True),
    "Nationals Park": VenueLocation(38.8730, -77.0074, climate_controlled=False),
}


def calculate_temperature_factor(temperature_f: float) -> float:
    """Ajuste multiplicativo da produção esperada de corridas pela temperatura,
    ancorado numa referência de 70°F. Limitado a [MIN_WEATHER_FACTOR, MAX_WEATHER_FACTOR]
    pra previsões extremas (ou erradas) não dominarem a projeção sozinhas."""
    raw = 1.0 + TEMPERATURE_SENSITIVITY * (temperature_f - TEMPERATURE_BASELINE_F)
    return max(MIN_WEATHER_FACTOR, min(raw, MAX_WEATHER_FACTOR))


class WeatherService:
    """Busca a previsão do tempo pro horário de um jogo e converte num weather_factor."""

    def __init__(self, api_client: Optional[WeatherApiClient] = None) -> None:
        self.api_client = api_client or WeatherApiClient()

    def get_weather_conditions(self, venue: Optional[str], game_datetime: Optional[str]) -> WeatherConditions:
        """Retorna as condições e o fator climático para o jogo. Cai para fator neutro
        (1.0) quando o estádio é climatizado, é desconhecido, o horário do jogo não é
        conhecido, ou a previsão falha."""
        if not venue or venue not in _VENUE_LOCATIONS:
            if venue:
                log.warning(f"Estádio '{venue}' sem coordenadas cadastradas; usando fator climático neutro")
            return WeatherConditions(venue or "desconhecido", None, None, None, False, 1.0)

        location = _VENUE_LOCATIONS[venue]
        if location.climate_controlled:
            return WeatherConditions(venue, None, None, None, True, 1.0)

        if not game_datetime:
            return WeatherConditions(venue, None, None, None, False, 1.0)

        try:
            forecast = self.api_client.get_hourly_forecast(location.latitude, location.longitude)
        except WeatherApiError as exc:
            log.warning(f"Previsão do tempo indisponível para {venue}: {exc}")
            return WeatherConditions(venue, None, None, None, False, 1.0)

        hour_data = self._closest_hour(forecast, game_datetime)
        if hour_data is None:
            return WeatherConditions(venue, None, None, None, False, 1.0)

        temperature_f, wind_speed, wind_direction = hour_data
        factor = calculate_temperature_factor(temperature_f) if temperature_f is not None else 1.0
        return WeatherConditions(venue, temperature_f, wind_speed, wind_direction, False, factor)

    @staticmethod
    def _closest_hour(
        forecast: dict, game_datetime: str
    ) -> Optional[Tuple[Optional[float], Optional[float], Optional[float]]]:
        """Encontra, na previsão horária bruta, o horário mais próximo do início do jogo."""
        hourly = forecast.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return None
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        directions = hourly.get("wind_direction_10m", [])

        target = datetime.fromisoformat(game_datetime.replace("Z", "+00:00")).replace(tzinfo=None)
        best_idx, best_diff = None, None
        for idx, time_str in enumerate(times):
            candidate = datetime.fromisoformat(time_str)
            diff = abs((candidate - target).total_seconds())
            if best_diff is None or diff < best_diff:
                best_idx, best_diff = idx, diff

        if best_idx is None:
            return None
        temp = temps[best_idx] if best_idx < len(temps) else None
        wind = winds[best_idx] if best_idx < len(winds) else None
        direction = directions[best_idx] if best_idx < len(directions) else None
        return temp, wind, direction
