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
- Vento: a DIREÇÃO importa mais que a velocidade pura -- vento soprando "campo afora"
  (de trás do batedor em direção ao campo externo) empurra a bola e aumenta HR/produção;
  soprando "campo adentro" (do campo externo em direção ao home) segura a bola. Para
  saber qual é o caso, cada estádio tem uma orientação `cf_bearing_deg` (a direção da
  bússola de home plate para o centro do campo). Combinando essa orientação com a
  direção de onde o vento sopra (Open-Meteo, convenção meteorológica: grau = direção de
  ORIGEM do vento), projetamos a componente do vento ao longo do eixo home->centro:
  componente positiva = vento a favor (afora), negativa = contra (adentro). O ajuste é
  proporcional a essa componente em mph (`WIND_SENSITIVITY`), com um sub-limite próprio
  para o vento sozinho nunca dominar. As orientações `cf_bearing_deg` são aproximações
  documentadas (mesmo padrão de placeholder dos park factors) -- a maioria dos estádios
  da MLB aponta home->centro para o quadrante NE/ENE (Regra 1.04 do regulamento), com
  outliers conhecidos; devem ser refinadas junto com a calibração.
- Estádios com teto fixo ou retrátil recebem fator sempre neutro (1.0): o clima externo
  não afeta o jogo dentro de um domo fechado, e a MLB tipicamente fecha o teto quando as
  condições seriam adversas -- justamente para neutralizar esse efeito.
- O fator final (temperatura * vento) é limitado a [MIN_WEATHER_FACTOR, MAX_WEATHER_FACTOR]:
  mesmo com calor extremo E vento forte a favor, uma previsão de tempo (que também tem
  incerteza própria) não domina a projeção sozinha.
- Falhas (API fora do ar, estádio desconhecido, sem horário do jogo) sempre caem para
  fator neutro (1.0) e nunca impedem a projeção de rodar -- clima é um ajuste
  incremental, mesmo padrão de degradação graciosa já usado em bullpen/park factor.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

from mlb_quantitative_engine.api.weather_api import WeatherApiClient, WeatherApiError
from mlb_quantitative_engine.utils.logger import log

TEMPERATURE_BASELINE_F: float = 70.0
TEMPERATURE_SENSITIVITY: float = 0.0015  # fração de ajuste no fator por °F de distância da referência
WIND_SENSITIVITY: float = 0.005  # fração de ajuste por mph de vento na componente home->centro
MIN_WIND_FACTOR: float = 0.94  # sub-limite: o vento sozinho não passa de ±6%
MAX_WIND_FACTOR: float = 1.06
MIN_WEATHER_FACTOR: float = 0.88
MAX_WEATHER_FACTOR: float = 1.12


@dataclass(frozen=True)
class VenueLocation:
    """Coordenadas de um estádio, orientação (home plate -> centro do campo, em graus de
    bússola) e se o teto (fixo ou retrátil) neutraliza o clima externo.

    `cf_bearing_deg` é uma aproximação documentada (ver docstring do módulo). Estádios
    climatizados carregam 0.0 por convenção -- o valor é irrelevante porque o vento nunca
    é aplicado quando `climate_controlled=True`."""

    latitude: float
    longitude: float
    cf_bearing_deg: float
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
# `cf_bearing_deg` = direção da bússola de home plate para o centro do campo
# (aproximação documentada; 0.0 quando climatizado, pois o vento não é aplicado).
_VENUE_LOCATIONS: Dict[str, VenueLocation] = {
    "Chase Field": VenueLocation(33.4455, -112.0667, 0.0, climate_controlled=True),
    "Sutter Health Park": VenueLocation(38.5805, -121.5133, 60.0, climate_controlled=False),
    "Truist Park": VenueLocation(33.8908, -84.4678, 55.0, climate_controlled=False),
    "Oriole Park at Camden Yards": VenueLocation(39.2838, -76.6217, 65.0, climate_controlled=False),
    "Fenway Park": VenueLocation(42.3467, -71.0972, 45.0, climate_controlled=False),
    "Wrigley Field": VenueLocation(41.9484, -87.6553, 30.0, climate_controlled=False),
    "Rate Field": VenueLocation(41.8299, -87.6338, 70.0, climate_controlled=False),
    "Great American Ball Park": VenueLocation(39.0979, -84.5082, 80.0, climate_controlled=False),
    "Progressive Field": VenueLocation(41.4962, -81.6852, 60.0, climate_controlled=False),
    "Coors Field": VenueLocation(39.7559, -104.9942, 0.0, climate_controlled=False),
    "Comerica Park": VenueLocation(42.3390, -83.0485, 50.0, climate_controlled=False),
    "Daikin Park": VenueLocation(29.7573, -95.3555, 0.0, climate_controlled=True),
    "Kauffman Stadium": VenueLocation(39.0517, -94.4803, 55.0, climate_controlled=False),
    "Angel Stadium": VenueLocation(33.8003, -117.8827, 40.0, climate_controlled=False),
    "UNIQLO Field at Dodger Stadium": VenueLocation(34.0739, -118.2400, 40.0, climate_controlled=False),
    "loanDepot park": VenueLocation(25.7781, -80.2196, 0.0, climate_controlled=True),
    "American Family Field": VenueLocation(43.0280, -87.9712, 0.0, climate_controlled=True),
    "Target Field": VenueLocation(44.9817, -93.2776, 75.0, climate_controlled=False),
    "Citi Field": VenueLocation(40.7571, -73.8458, 30.0, climate_controlled=False),
    "Yankee Stadium": VenueLocation(40.8296, -73.9262, 25.0, climate_controlled=False),
    "Citizens Bank Park": VenueLocation(39.9061, -75.1665, 30.0, climate_controlled=False),
    "PNC Park": VenueLocation(40.4469, -80.0057, 60.0, climate_controlled=False),
    "Petco Park": VenueLocation(32.7073, -117.1566, 40.0, climate_controlled=False),
    "Oracle Park": VenueLocation(37.7786, -122.3893, 90.0, climate_controlled=False),
    "T-Mobile Park": VenueLocation(47.5914, -122.3325, 0.0, climate_controlled=True),
    "Busch Stadium": VenueLocation(38.6226, -90.1928, 60.0, climate_controlled=False),
    "Tropicana Field": VenueLocation(27.7683, -82.6534, 0.0, climate_controlled=True),
    "Globe Life Field": VenueLocation(32.7473, -97.0842, 0.0, climate_controlled=True),
    "Rogers Centre": VenueLocation(43.6414, -79.3894, 0.0, climate_controlled=True),
    "Nationals Park": VenueLocation(38.8730, -77.0074, 30.0, climate_controlled=False),
}


def calculate_temperature_factor(temperature_f: float) -> float:
    """Ajuste multiplicativo da produção esperada de corridas pela temperatura,
    ancorado numa referência de 70°F. Limitado a [MIN_WEATHER_FACTOR, MAX_WEATHER_FACTOR]
    pra previsões extremas (ou erradas) não dominarem a projeção sozinhas."""
    raw = 1.0 + TEMPERATURE_SENSITIVITY * (temperature_f - TEMPERATURE_BASELINE_F)
    return max(MIN_WEATHER_FACTOR, min(raw, MAX_WEATHER_FACTOR))


def outbound_wind_component(wind_speed_mph: float, wind_from_deg: float, cf_bearing_deg: float) -> float:
    """Componente do vento (mph) ao longo do eixo home plate -> centro do campo.
    Positivo = vento soprando "campo afora" (a favor do ataque); negativo = "campo
    adentro" (contra).

    `wind_from_deg` é a direção de ORIGEM do vento (convenção meteorológica do Open-Meteo);
    a direção PARA ONDE o vento sopra é `wind_from_deg + 180`. Projetamos essa direção no
    eixo home->centro (`cf_bearing_deg`) via cosseno do ângulo entre elas."""
    toward_deg = (wind_from_deg + 180.0) % 360.0
    angle = math.radians(toward_deg - cf_bearing_deg)
    return wind_speed_mph * math.cos(angle)


def calculate_wind_factor(wind_speed_mph: float, wind_from_deg: float, cf_bearing_deg: float) -> float:
    """Ajuste multiplicativo pela componente do vento no eixo home->centro do campo.
    Limitado a [MIN_WIND_FACTOR, MAX_WIND_FACTOR] pra o vento sozinho não dominar."""
    component = outbound_wind_component(wind_speed_mph, wind_from_deg, cf_bearing_deg)
    raw = 1.0 + WIND_SENSITIVITY * component
    return max(MIN_WIND_FACTOR, min(raw, MAX_WIND_FACTOR))


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
        temp_factor = calculate_temperature_factor(temperature_f) if temperature_f is not None else 1.0
        wind_factor = (
            calculate_wind_factor(wind_speed, wind_direction, location.cf_bearing_deg)
            if wind_speed is not None and wind_direction is not None
            else 1.0
        )
        combined = max(MIN_WEATHER_FACTOR, min(temp_factor * wind_factor, MAX_WEATHER_FACTOR))
        return WeatherConditions(venue, temperature_f, wind_speed, wind_direction, False, combined)

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
