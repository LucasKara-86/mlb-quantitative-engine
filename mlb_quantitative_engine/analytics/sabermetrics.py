from __future__ import annotations

"""Fórmulas sabermétricas fundamentais para rebatedores e arremessadores.

Este módulo é puramente matemático: recebe estatísticas brutas (contagem) e
devolve métricas derivadas. Não depende de nenhuma API — a origem dos dados
brutos é responsabilidade das camadas api/ e services/, construídas em
etapas futuras.

Fórmulas seguem as definições públicas padrão de sabermetria (Bill James,
FanGraphs). As constantes de liga (pesos de wOBA, constante de FIP) mudam a
cada temporada; os valores padrão em LeagueConstants são uma aproximação
razoável e devem ser substituídos por valores calibrados quando a etapa de
calibração automática existir.
"""

from dataclasses import dataclass
from typing import Optional, Sequence


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide protegendo contra denominador zero (comum com amostras pequenas)."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def shrink_toward_league_average(
    observed: float, league_average: float, sample_size: float, stabilization_point: float
) -> float:
    """Encolhimento Bayesiano simples (Empirical Bayes / regressão à média): puxa uma
    métrica observada em direção à média da liga, proporcionalmente a quão pequena é a
    amostra perto do "ponto de estabilização" da métrica (PA para métricas de rebatida,
    IP para métricas de arremesso — ver pesquisa pública de confiabilidade de estatísticas
    de beisebol, ex. Russell Carleton).

    Fórmula: shrunk = observed * n/(n+k) + league_average * k/(n+k)
    - n >> k (amostra grande): shrunk ~= observed (confiamos no dado real).
    - n << k (amostra pequena/início de temporada): shrunk ~= league_average (não há
      dado suficiente pra confiar na métrica observada — ela pode ser ruído).

    Sem isso, um titular com 15 innings pitched e um FIP de sorte de 1.50, ou uma
    lineup com poucos PA e um wRC+ de 180 por puro acaso de amostra pequena, entram no
    motor de projeção como se fossem tão confiáveis quanto um jogador com temporada
    inteira de dados — inflando artificialmente a "confiança" do modelo.
    """
    if sample_size <= 0:
        return league_average
    weight = sample_size / (sample_size + stabilization_point)
    return observed * weight + league_average * (1.0 - weight)


@dataclass(frozen=True)
class LeagueConstants:
    """Constantes sabermétricas dependentes de temporada (equivalente à aba "Guts" da FanGraphs).

    Os valores padrão refletem uma aproximação da temporada de 2023 e servem
    como fallback enquanto a calibração automática por temporada não existe.
    """

    w_bb: float = 0.696
    w_hbp: float = 0.726
    w_1b: float = 0.883
    w_2b: float = 1.244
    w_3b: float = 1.569
    w_hr: float = 2.007
    woba_scale: float = 1.242
    league_woba: float = 0.318
    league_r_per_pa: float = 0.114
    fip_constant: float = 3.10
    league_hr_per_fb: float = 0.105
    league_avg_runs_per_game: float = 4.30
    league_avg_era: float = 4.00


# ---------------------------------------------------------------------------
# Rebatedores (batting)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BattingStatLine:
    """Estatísticas brutas de contagem de um rebatedor em um período (jogo, split ou temporada)."""

    ab: int
    h: int
    doubles: int = 0
    triples: int = 0
    hr: int = 0
    bb: int = 0
    ibb: int = 0
    hbp: int = 0
    sf: int = 0
    sh: int = 0
    k: int = 0

    @property
    def singles(self) -> int:
        return self.h - self.doubles - self.triples - self.hr

    @property
    def total_bases(self) -> int:
        return self.singles + 2 * self.doubles + 3 * self.triples + 4 * self.hr

    @property
    def plate_appearances(self) -> int:
        return self.ab + self.bb + self.hbp + self.sf + self.sh


@dataclass(frozen=True)
class BattingMetrics:
    """Métricas sabermétricas derivadas de um BattingStatLine."""

    avg: float
    obp: float
    slg: float
    ops: float
    iso: float
    babip: float
    woba: float
    wrc_plus: float
    runs_created: float
    plate_appearances: int = 0


def calculate_avg(stats: BattingStatLine) -> float:
    return _safe_divide(stats.h, stats.ab)


def calculate_obp(stats: BattingStatLine) -> float:
    numerator = stats.h + stats.bb + stats.hbp
    denominator = stats.ab + stats.bb + stats.hbp + stats.sf
    return _safe_divide(numerator, denominator)


def calculate_slg(stats: BattingStatLine) -> float:
    return _safe_divide(stats.total_bases, stats.ab)


def calculate_ops(stats: BattingStatLine) -> float:
    return calculate_obp(stats) + calculate_slg(stats)


def calculate_iso(stats: BattingStatLine) -> float:
    return calculate_slg(stats) - calculate_avg(stats)


def calculate_babip(stats: BattingStatLine) -> float:
    """Batting Average on Balls In Play: exclui HR e conta SF como oportunidades de out."""
    numerator = stats.h - stats.hr
    denominator = stats.ab - stats.k - stats.hr + stats.sf
    return _safe_divide(numerator, denominator)


def calculate_woba(stats: BattingStatLine, constants: LeagueConstants = LeagueConstants()) -> float:
    """Weighted On-Base Average: pondera cada forma de chegar à base pelo seu valor real em corridas."""
    numerator = (
        constants.w_bb * (stats.bb - stats.ibb)
        + constants.w_hbp * stats.hbp
        + constants.w_1b * stats.singles
        + constants.w_2b * stats.doubles
        + constants.w_3b * stats.triples
        + constants.w_hr * stats.hr
    )
    denominator = stats.ab + stats.bb - stats.ibb + stats.sf + stats.hbp
    return _safe_divide(numerator, denominator)


def calculate_wrc_plus(stats: BattingStatLine, constants: LeagueConstants = LeagueConstants()) -> float:
    """Weighted Runs Created Plus, normalizado para 100 = média da liga.

    Nota: esta é a versão SEM ajuste de park factor (park factor = 1.0 implícito).
    O ajuste por estádio será aplicado quando park_factor_service existir.
    """
    woba = calculate_woba(stats, constants)
    wraa_per_pa = (woba - constants.league_woba) / constants.woba_scale
    wrc_per_pa = wraa_per_pa + constants.league_r_per_pa
    return _safe_divide(wrc_per_pa, constants.league_r_per_pa) * 100


def calculate_runs_created(stats: BattingStatLine) -> float:
    """Runs Created — fórmula básica de Bill James: RC = (H + BB) * TB / (AB + BB)."""
    numerator = (stats.h + stats.bb) * stats.total_bases
    denominator = stats.ab + stats.bb
    return _safe_divide(numerator, denominator)


def compute_batting_metrics(
    stats: BattingStatLine, constants: LeagueConstants = LeagueConstants()
) -> BattingMetrics:
    """Calcula o conjunto completo de métricas de rebatedor a partir de um BattingStatLine."""
    return BattingMetrics(
        avg=round(calculate_avg(stats), 4),
        obp=round(calculate_obp(stats), 4),
        slg=round(calculate_slg(stats), 4),
        ops=round(calculate_ops(stats), 4),
        iso=round(calculate_iso(stats), 4),
        babip=round(calculate_babip(stats), 4),
        woba=round(calculate_woba(stats, constants), 4),
        wrc_plus=round(calculate_wrc_plus(stats, constants), 1),
        runs_created=round(calculate_runs_created(stats), 2),
        plate_appearances=stats.plate_appearances,
    )


def aggregate_batting_stat_lines(stat_lines: Sequence[BattingStatLine]) -> BattingStatLine:
    """Soma as estatísticas de contagem de vários rebatedores (ex.: uma lineup) em uma só linha.

    Correto estatisticamente para depois derivar taxas de time (wOBA, wRC+, etc.):
    é preciso somar as contagens primeiro e só então calcular a taxa sobre o
    agregado. Fazer a média das taxas individuais dos jogadores produziria um
    valor incorreto sempre que eles tiverem volumes de AB diferentes.
    """
    return BattingStatLine(
        ab=sum(s.ab for s in stat_lines),
        h=sum(s.h for s in stat_lines),
        doubles=sum(s.doubles for s in stat_lines),
        triples=sum(s.triples for s in stat_lines),
        hr=sum(s.hr for s in stat_lines),
        bb=sum(s.bb for s in stat_lines),
        ibb=sum(s.ibb for s in stat_lines),
        hbp=sum(s.hbp for s in stat_lines),
        sf=sum(s.sf for s in stat_lines),
        sh=sum(s.sh for s in stat_lines),
        k=sum(s.k for s in stat_lines),
    )


# ---------------------------------------------------------------------------
# Arremessadores (pitching)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PitchingStatLine:
    """Estatísticas brutas de um arremessador em um período.

    `outs` (IP * 3) é usado em vez de innings decimais para evitar o erro
    clássico de tratar "6.1 innings" como 6.1 em ponto flutuante — no boxscore
    da MLB, a casa decimal representa outs (0, 1 ou 2), não décimos de inning.
    """

    outs: int
    h: int
    er: int
    r: int = 0
    hr: int = 0
    bb: int = 0
    ibb: int = 0
    hbp: int = 0
    k: int = 0
    batters_faced: int = 0
    ground_balls: int = 0
    fly_balls: int = 0
    line_drives: int = 0

    @property
    def innings_pitched(self) -> float:
        return self.outs / 3


@dataclass(frozen=True)
class PitchingMetrics:
    """Métricas sabermétricas derivadas de um PitchingStatLine."""

    era: float
    whip: float
    k_percent: float
    bb_percent: float
    k_minus_bb_percent: float
    hr_per_9: float
    fip: float
    xfip: Optional[float]
    lob_percent: float
    gb_percent: Optional[float]
    fb_percent: Optional[float]
    innings_pitched: float = 0.0


def calculate_era(stats: PitchingStatLine) -> float:
    return _safe_divide(stats.er, stats.innings_pitched) * 9


def calculate_whip(stats: PitchingStatLine) -> float:
    return _safe_divide(stats.bb + stats.h, stats.innings_pitched)


def calculate_k_percent(stats: PitchingStatLine) -> float:
    return _safe_divide(stats.k, stats.batters_faced)


def calculate_bb_percent(stats: PitchingStatLine) -> float:
    return _safe_divide(stats.bb, stats.batters_faced)


def calculate_hr_per_9(stats: PitchingStatLine) -> float:
    return _safe_divide(stats.hr, stats.innings_pitched) * 9


def calculate_fip(stats: PitchingStatLine, constants: LeagueConstants = LeagueConstants()) -> float:
    """Fielding Independent Pitching: usa apenas eventos que o arremessador controla (HR, BB, HBP, K)."""
    numerator = 13 * stats.hr + 3 * (stats.bb + stats.hbp) - 2 * stats.k
    return _safe_divide(numerator, stats.innings_pitched) + constants.fip_constant


def calculate_xfip(stats: PitchingStatLine, constants: LeagueConstants = LeagueConstants()) -> Optional[float]:
    """Expected FIP: substitui HR reais por HR esperados (fly balls * taxa média de HR/FB da liga).

    Normaliza contra variância de sorte/parque na taxa de bolas voadas que viram HR.
    Retorna None quando não há dados de batted-ball (fly_balls indisponível) — sem essa
    guarda, fly_balls=0 seria interpretado como "zero HR esperados", produzindo um xFIP
    artificialmente baixo em vez de refletir a ausência do dado.
    """
    total_batted_balls = stats.ground_balls + stats.fly_balls + stats.line_drives
    if total_batted_balls == 0:
        return None
    expected_hr = stats.fly_balls * constants.league_hr_per_fb
    numerator = 13 * expected_hr + 3 * (stats.bb + stats.hbp) - 2 * stats.k
    return _safe_divide(numerator, stats.innings_pitched) + constants.fip_constant


def calculate_lob_percent(stats: PitchingStatLine) -> float:
    """Left On Base %: percentual de corredores embasados que não marcaram corrida."""
    numerator = stats.h + stats.bb + stats.hbp - stats.r
    denominator = stats.h + stats.bb + stats.hbp - (1.4 * stats.hr)
    return _safe_divide(numerator, denominator)


def calculate_gb_percent(stats: PitchingStatLine) -> Optional[float]:
    total_batted_balls = stats.ground_balls + stats.fly_balls + stats.line_drives
    if total_batted_balls == 0:
        return None
    return stats.ground_balls / total_batted_balls


def calculate_fb_percent(stats: PitchingStatLine) -> Optional[float]:
    total_batted_balls = stats.ground_balls + stats.fly_balls + stats.line_drives
    if total_batted_balls == 0:
        return None
    return stats.fly_balls / total_batted_balls


def compute_pitching_metrics(
    stats: PitchingStatLine, constants: LeagueConstants = LeagueConstants()
) -> PitchingMetrics:
    """Calcula o conjunto completo de métricas de arremessador a partir de um PitchingStatLine."""
    k_pct = calculate_k_percent(stats)
    bb_pct = calculate_bb_percent(stats)
    gb_pct = calculate_gb_percent(stats)
    fb_pct = calculate_fb_percent(stats)
    xfip = calculate_xfip(stats, constants)
    return PitchingMetrics(
        era=round(calculate_era(stats), 2),
        whip=round(calculate_whip(stats), 3),
        k_percent=round(k_pct, 4),
        bb_percent=round(bb_pct, 4),
        k_minus_bb_percent=round(k_pct - bb_pct, 4),
        hr_per_9=round(calculate_hr_per_9(stats), 2),
        fip=round(calculate_fip(stats, constants), 2),
        xfip=round(xfip, 2) if xfip is not None else None,
        lob_percent=round(calculate_lob_percent(stats), 4),
        gb_percent=round(gb_pct, 4) if gb_pct is not None else None,
        fb_percent=round(fb_pct, 4) if fb_pct is not None else None,
        innings_pitched=round(stats.innings_pitched, 1),
    )


def aggregate_pitching_stat_lines(stat_lines: Sequence[PitchingStatLine]) -> PitchingStatLine:
    """Soma as estatísticas de contagem de vários arremessadores (ex.: um bullpen) em uma só linha.

    Mesma lógica de aggregate_batting_stat_lines: somar as contagens primeiro e só
    então derivar taxas (ERA, WHIP, FIP) do agregado — a média das ERAs individuais
    dos arremessadores seria matematicamente incorreta quando eles têm volumes de
    innings pitched diferentes.
    """
    return PitchingStatLine(
        outs=sum(s.outs for s in stat_lines),
        h=sum(s.h for s in stat_lines),
        er=sum(s.er for s in stat_lines),
        r=sum(s.r for s in stat_lines),
        hr=sum(s.hr for s in stat_lines),
        bb=sum(s.bb for s in stat_lines),
        ibb=sum(s.ibb for s in stat_lines),
        hbp=sum(s.hbp for s in stat_lines),
        k=sum(s.k for s in stat_lines),
        batters_faced=sum(s.batters_faced for s in stat_lines),
        ground_balls=sum(s.ground_balls for s in stat_lines),
        fly_balls=sum(s.fly_balls for s in stat_lines),
        line_drives=sum(s.line_drives for s in stat_lines),
    )
