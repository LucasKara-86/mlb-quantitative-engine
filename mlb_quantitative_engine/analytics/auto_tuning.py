from __future__ import annotations

"""Motor de detecção de oportunidades de melhoria: recebe calibração + backtest já
calculados e devolve propostas de ajuste de parâmetro -- nunca lógica/fórmula, só os
números registrados em `tunable_params.json` (ver config.py).

Raciocínio de design:
- Puramente matemático (não depende de services/ nem de api/, mesma convenção do resto de
  analytics/): recebe `ReliabilityBucket`s (analytics/calibration.py) e métricas de
  `BacktestResult` (analytics/backtesting.py) já prontas -- quem monta esses dados a partir
  do banco é services/auto_tuning_service.py.
- Cada regra só age com amostra mínima (`MIN_SAMPLE_THRESHOLD`), para não reagir a ruído de
  poucas dezenas de apostas -- o intervalo de confiança de uma taxa de acerto com amostra
  pequena é enorme (ver docstring de analytics/calibration.py).
- Cada parâmetro tem limites e um passo máximo por execução (`PARAMETER_BOUNDS`) -- nunca
  sai do intervalo, e nunca dá um salto grande de uma vez só (evita overfitting a um dia
  de dados ruidosos; o efeito de cada mudança só aparece com o tempo).
- `propose_adjustments` também recebe quais parâmetros estão em cooldown (mudados
  recentemente, aguardando efeito observável) e um teto de mudanças por execução --
  candidatos que passariam na regra mas ficam bloqueados por um desses dois motivos
  entram em `deferred`, não em `accepted`, e aparecem no relatório do Telegram como
  "candidata, aguardando".

Regras implementadas nesta primeira versão (ver Passo 3 do plano):
- R1/R2 -- viés sistemático de calibração: reproduz o incidente documentado em
  analytics/calibration.py (8 de 9 apostas entre 64-89% de probabilidade prevista
  perderam no mesmo dia) de forma genérica -- se as faixas de probabilidade prevista
  >=60% (onde o critério de Value Bet realmente admite apostas) estão, em conjunto,
  sistematicamente confiantes demais (`overconfidence` alto, modelo super confiante) ou
  de menos (`overconfidence` muito negativo), ajusta `overdispersion` (alavanca
  primária) e, se já no limite, `mean_uncertainty_pct` (analytics/poisson.py e
  analytics/monte_carlo.py).
- R4 -- ROI geral do backtest negativo com amostra suficiente: eleva `min_edge` e, se já
  no teto, `min_confidence` (torna a admissão de apostas mais rígida). ROI negativo só
  num segmento específico (ex.: só Team-Total Under) vira achado reportado
  (`find_negative_roi_segments`), não ação automática -- o sistema não tem hoje limiares
  por mercado, e criar esse eixo de config é uma mudança maior para decisão futura, não
  para aplicar sozinho sobre uma amostra ainda pequena por segmento.

Fora de escopo nesta versão (não implementado, não fingido):
- Revalidação dos filtros de qualidade de time (low/high winpct threshold): exigiria
  persistir o win% do time NO MOMENTO da aposta (não existe hoje no schema de ValueBet),
  então não há como recalcular a fronteira original com o histórico já gravado.
- CLV (Closing Line Value): analytics/backtesting.py já calcula, mas este projeto ainda
  não persiste a odd de fechamento (`closing_price` fica sempre None em
  services/backtest_report_service.py) -- não há dado real para a regra agir.
- Kelly (`kelly_fraction_multiplier`), teto de stake (`max_stake_fraction`) e as
  constantes de projeção/bullpen (`starter_expected_innings` etc.): ficam no whitelist de
  `tunable_params.json` (ajustáveis manualmente ou por uma regra futura), mas nenhuma
  regra desta versão os altera -- não há hoje um sinal estatístico direto e não-ambíguo
  ligando essas constantes ao resultado observado sem uma segmentação de dados que o
  projeto ainda não coleta.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from mlb_quantitative_engine.analytics.backtesting import BacktestResult
from mlb_quantitative_engine.analytics.calibration import ReliabilityBucket

MIN_SAMPLE_THRESHOLD: int = 20
OVERCONFIDENCE_TOLERANCE: float = 0.10  # 10 pontos percentuais (prevista - real)
NEGATIVE_ROI_TOLERANCE: float = -0.05  # -5% de ROI geral
DEFAULT_MAX_CHANGES_PER_RUN: int = 2
NEGATIVE_ROI_SEGMENT_MIN_SAMPLE: int = 20


@dataclass(frozen=True)
class ParameterBounds:
    """Limites e passo máximo de ajuste por execução para um parâmetro ajustável."""

    minimum: float
    maximum: float
    step: float


PARAMETER_BOUNDS: Dict[str, ParameterBounds] = {
    "overdispersion": ParameterBounds(minimum=1.1, maximum=2.0, step=0.05),
    "mean_uncertainty_pct": ParameterBounds(minimum=0.05, maximum=0.30, step=0.01),
    "min_edge": ParameterBounds(minimum=0.02, maximum=0.10, step=0.005),
    "min_expected_value": ParameterBounds(minimum=0.02, maximum=0.12, step=0.005),
    "min_confidence": ParameterBounds(minimum=50.0, maximum=90.0, step=2.0),
    "low_winpct_over_threshold": ParameterBounds(minimum=0.35, maximum=0.50, step=0.01),
    "high_winpct_under_threshold": ParameterBounds(minimum=0.50, maximum=0.65, step=0.01),
    "kelly_fraction_multiplier": ParameterBounds(minimum=0.10, maximum=0.50, step=0.02),
    "max_stake_fraction": ParameterBounds(minimum=0.005, maximum=0.05, step=0.0025),
}


@dataclass(frozen=True)
class ParameterAdjustment:
    """Uma proposta de mudança de um parâmetro, com o motivo e o tamanho da amostra que a sustenta."""

    parameter_name: str
    old_value: float
    new_value: float
    rationale: str
    sample_size: int


@dataclass(frozen=True)
class AutoTuningOutcome:
    """Resultado de uma rodada de detecção: o que deve ser aplicado agora (`accepted`,
    respeitando cooldown e teto por execução) e o que passaria na regra mas fica represado
    (`deferred`) -- ambos entram no relatório do Telegram."""

    accepted: List[ParameterAdjustment]
    deferred: List[ParameterAdjustment]


def _clip(value: float, bounds: ParameterBounds) -> float:
    return max(bounds.minimum, min(bounds.maximum, value))


def _propose_calibration_adjustment(
    reliability: Sequence[ReliabilityBucket],
    current_overdispersion: float,
    current_mean_uncertainty_pct: float,
) -> Optional[ParameterAdjustment]:
    """R1/R2: viés sistemático de calibração nas faixas de probabilidade prevista >=60%
    (onde o critério de Value Bet realmente admite apostas), ponderado pelo tamanho de
    cada faixa."""
    relevant = [b for b in reliability if b.predicted_probability_mean >= 0.60 and b.count >= MIN_SAMPLE_THRESHOLD]
    if not relevant:
        return None

    total_n = sum(b.count for b in relevant)
    weighted_overconfidence = sum(b.overconfidence * b.count for b in relevant) / total_n

    if weighted_overconfidence > OVERCONFIDENCE_TOLERANCE:
        direction, verb = 1, "confiante demais"
    elif weighted_overconfidence < -OVERCONFIDENCE_TOLERANCE:
        direction, verb = -1, "conservador demais"
    else:
        return None

    bounds = PARAMETER_BOUNDS["overdispersion"]
    at_bound = (direction > 0 and current_overdispersion >= bounds.maximum) or (
        direction < 0 and current_overdispersion <= bounds.minimum
    )
    if not at_bound:
        new_value = _clip(current_overdispersion + direction * bounds.step, bounds)
        if new_value != current_overdispersion:
            return ParameterAdjustment(
                "overdispersion", current_overdispersion, new_value,
                f"Modelo {verb} nas faixas de probabilidade prevista >=60% (overconfidence "
                f"médio ponderado {weighted_overconfidence:+.1%}, n={total_n}) -- ajusta a "
                f"sobre-dispersão da Binomial Negativa para {'alargar' if direction > 0 else 'estreitar'} "
                f"a distribuição de corridas.",
                total_n,
            )

    bounds_u = PARAMETER_BOUNDS["mean_uncertainty_pct"]
    candidate_uncertainty = current_mean_uncertainty_pct + direction * bounds_u.step
    new_value_u = _clip(candidate_uncertainty, bounds_u)
    if new_value_u != current_mean_uncertainty_pct:
        return ParameterAdjustment(
            "mean_uncertainty_pct", current_mean_uncertainty_pct, new_value_u,
            f"Sobre-dispersão já no limite; modelo {verb} nas faixas >=60% (overconfidence "
            f"médio ponderado {weighted_overconfidence:+.1%}, n={total_n}) -- ajusta a incerteza "
            f"da média projetada no Monte Carlo pela mesma razão.",
            total_n,
        )
    return None


def _propose_admission_tightening(
    overall_roi: Optional[float],
    overall_n: int,
    current_min_edge: float,
    current_min_confidence: float,
) -> Optional[ParameterAdjustment]:
    """R4: ROI geral do backtest negativo com amostra suficiente -- eleva o piso de
    admissão de uma aposta (edge, depois confiança), nunca ambos na mesma execução."""
    if overall_roi is None or overall_n < MIN_SAMPLE_THRESHOLD or overall_roi >= NEGATIVE_ROI_TOLERANCE:
        return None

    bounds = PARAMETER_BOUNDS["min_edge"]
    if current_min_edge < bounds.maximum:
        new_value = _clip(current_min_edge + bounds.step, bounds)
        return ParameterAdjustment(
            "min_edge", current_min_edge, new_value,
            f"ROI geral negativo (ROI={overall_roi:+.1%}, n={overall_n}) -- eleva o piso de "
            f"edge exigido para admitir uma aposta.",
            overall_n,
        )

    bounds_c = PARAMETER_BOUNDS["min_confidence"]
    if current_min_confidence < bounds_c.maximum:
        new_value = _clip(current_min_confidence + bounds_c.step, bounds_c)
        return ParameterAdjustment(
            "min_confidence", current_min_confidence, new_value,
            f"ROI geral negativo (ROI={overall_roi:+.1%}, n={overall_n}) -- min_edge já no "
            f"teto; eleva o piso de confiança exigido.",
            overall_n,
        )
    return None


def propose_adjustments(
    *,
    reliability: Sequence[ReliabilityBucket],
    overall_roi: Optional[float],
    overall_n: int,
    current_values: Dict[str, float],
    parameters_in_cooldown: Sequence[str] = (),
    max_changes_per_run: int = DEFAULT_MAX_CHANGES_PER_RUN,
) -> AutoTuningOutcome:
    """Ponto de entrada: gera candidatos a partir das regras acima, filtra pelos que
    estão em cooldown ou acima do teto por execução, devolve (`accepted`, `deferred`).

    `current_values` precisa ter, no mínimo, as chaves referenciadas pelas regras ativas:
    "overdispersion", "mean_uncertainty_pct", "min_edge", "min_confidence".
    """
    candidates: List[ParameterAdjustment] = []

    calibration_adjustment = _propose_calibration_adjustment(
        reliability, current_values["overdispersion"], current_values["mean_uncertainty_pct"],
    )
    if calibration_adjustment is not None:
        candidates.append(calibration_adjustment)

    admission_adjustment = _propose_admission_tightening(
        overall_roi, overall_n, current_values["min_edge"], current_values["min_confidence"],
    )
    if admission_adjustment is not None:
        candidates.append(admission_adjustment)

    accepted: List[ParameterAdjustment] = []
    deferred: List[ParameterAdjustment] = []
    for candidate in candidates:
        if candidate.parameter_name in parameters_in_cooldown or len(accepted) >= max_changes_per_run:
            deferred.append(candidate)
        else:
            accepted.append(candidate)

    return AutoTuningOutcome(accepted=accepted, deferred=deferred)


def find_negative_roi_segments(
    by_market: Dict[str, BacktestResult],
    min_sample: int = NEGATIVE_ROI_SEGMENT_MIN_SAMPLE,
    roi_threshold: float = NEGATIVE_ROI_TOLERANCE,
) -> List[str]:
    """Segmentos (família de mercado) com ROI abaixo de `roi_threshold` e amostra
    suficiente -- achado reportado no Telegram, nunca aplicado automaticamente (ver
    docstring do módulo: o sistema não tem limiares por mercado hoje)."""
    findings = []
    for market, result in sorted(by_market.items()):
        if result.total_bets >= min_sample and result.roi < roi_threshold:
            findings.append(
                f"{market}: ROI {result.roi:+.1%} em {result.total_bets} apostas (hit rate {result.hit_rate:.1%})"
            )
    return findings
