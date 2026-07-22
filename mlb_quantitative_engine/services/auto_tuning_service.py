from __future__ import annotations

"""Orquestra o ciclo completo de auto-tuning: monta calibração + backtest acumulados,
pede propostas a analytics/auto_tuning.py, e aplica cada uma só se sobreviver ao gate de
segurança -- suíte de testes completa. Nunca edita lógica/fórmula em `.py`, só o arquivo
de parâmetros (`tunable_params.json`).

Gate de segurança, por proposta aceita (não em cooldown, dentro do teto por execução):
1. Exige árvore git limpa ANTES de tentar qualquer coisa (senão a sessão inteira vira
   `skipped_reason`, nada é escrito) -- nunca mistura uma mudança de parâmetro com
   trabalho manual não commitado do usuário.
2. Reescreve `tunable_params.json` em memória e no disco.
3. Roda a suíte de testes completa (`run_tests_fn`).
   - Falhou -> restaura o JSON anterior, NÃO commita, registra a tentativa como revertida
     em `ParameterChangeLog`.
   - Passou -> commita (`commit_fn`) e, se `push=True`, `git push origin main` -- é o que
     faz o CI (.github/workflows/tests.yml) rodar de fato e serve de backup fora desta
     máquina. Registra o sucesso em `ParameterChangeLog` com o SHA do commit.
4. Cada proposta aceita é aplicada e testada isoladamente (uma de cada vez), para que uma
   falha numa não bloqueie a outra -- e para que o commit de cada mudança fique atômico e
   revertível individualmente (`git revert <sha>`).

`run_tests_fn`/`git_is_clean_fn`/`commit_fn`/`tunable_params_path` são todos injetáveis
(mesmo padrão de `apply_fn` em reports/daily_planner.py) -- em produção usam os defaults
reais (pytest + git de verdade sobre o repositório real); os testes deste módulo
(tests/test_auto_tuning_service.py) injetam substitutos para nunca tocar em git real nem
no `tunable_params.json` real durante a suíte.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from mlb_quantitative_engine.analytics.auto_tuning import (
    AutoTuningOutcome,
    ParameterAdjustment,
    find_negative_roi_segments,
    propose_adjustments,
)
from mlb_quantitative_engine.config import TUNABLE_PARAMS_PATH, settings
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.services.backtest_report_service import BacktestReport, build_backtest_report
from mlb_quantitative_engine.services.calibration_report_service import CalibrationReport, build_calibration_report
from mlb_quantitative_engine.utils.logger import log

COOLDOWN_DAYS: int = 5
_ACTIVE_RULE_PARAMETERS: Sequence[str] = ("overdispersion", "mean_uncertainty_pct", "min_edge", "min_confidence")
_PROJECT_ROOT: Path = TUNABLE_PARAMS_PATH.parents[1]

RunTestsFn = Callable[[], bool]
GitCleanFn = Callable[[], bool]
CommitFn = Callable[[ParameterAdjustment], Optional[str]]


@dataclass(frozen=True)
class AppliedChange:
    """Uma proposta já processada pelo gate de segurança: `applied=True` quando os testes
    passaram e o commit foi feito; `applied=False` quando os testes falharam e o valor
    anterior foi restaurado (nada commitado)."""

    adjustment: ParameterAdjustment
    applied: bool
    git_commit_sha: Optional[str]
    test_failure_summary: Optional[str] = None


@dataclass(frozen=True)
class AutoTuningRunResult:
    """Resumo de uma execução completa -- o que services/telegram_notifier.py precisa
    para montar o relatório diário."""

    changes: List[AppliedChange]
    deferred: List[ParameterAdjustment]
    negative_roi_findings: List[str]
    calibration: CalibrationReport
    backtest: BacktestReport
    skipped_reason: Optional[str] = None


def _read_tunable_params(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_tunable_params(values: dict, path: Path) -> None:
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bump_model_version(values: dict) -> dict:
    current = str(values.get("model_version", "v1"))
    try:
        number = int(current.lstrip("v")) + 1
    except ValueError:
        number = 2
    values = dict(values)
    values["model_version"] = f"v{number}"
    values["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return values


def _default_git_is_clean() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=_PROJECT_ROOT, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == ""


def _default_run_tests() -> bool:
    result = subprocess.run(["python", "-m", "pytest", "-q"], cwd=_PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(f"auto_tuning_service: gate de testes falhou:\n{result.stdout[-4000:]}\n{result.stderr[-1000:]}")
    return result.returncode == 0


def _default_commit(adjustment: ParameterAdjustment, push: bool) -> Optional[str]:
    subprocess.run(["git", "add", str(TUNABLE_PARAMS_PATH)], cwd=_PROJECT_ROOT, capture_output=True, text=True)
    message = (
        f"auto-tuning: {adjustment.parameter_name} {adjustment.old_value} -> {adjustment.new_value}\n\n"
        f"{adjustment.rationale}\n\nAmostra: {adjustment.sample_size} apostas resolvidas.\n\n"
        f"Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
    )
    commit_result = subprocess.run(["git", "commit", "-m", message], cwd=_PROJECT_ROOT, capture_output=True, text=True)
    if commit_result.returncode != 0:
        log.error(f"auto_tuning_service: falha ao commitar {adjustment.parameter_name}: {commit_result.stderr}")
        return None
    sha_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, capture_output=True, text=True)
    sha = sha_result.stdout.strip() or None
    if push:
        push_result = subprocess.run(["git", "push", "origin", "main"], cwd=_PROJECT_ROOT, capture_output=True, text=True)
        if push_result.returncode != 0:
            log.warning(f"auto_tuning_service: commit local ok, mas push falhou: {push_result.stderr}")
    return sha


def _parameters_in_cooldown(repository: Repository, now: datetime) -> List[str]:
    in_cooldown = []
    for parameter_name in _ACTIVE_RULE_PARAMETERS:
        last_change = repository.last_applied_parameter_change(parameter_name)
        if last_change is not None:
            change_time = last_change.created_at
            if change_time.tzinfo is None:
                change_time = change_time.replace(tzinfo=timezone.utc)
            if now - change_time < timedelta(days=COOLDOWN_DAYS):
                in_cooldown.append(parameter_name)
    return in_cooldown


def run_auto_tuning(
    repository: Optional[Repository] = None,
    now: Optional[datetime] = None,
    run_tests_fn: Optional[RunTestsFn] = None,
    git_is_clean_fn: Optional[GitCleanFn] = None,
    commit_fn: Optional[CommitFn] = None,
    tunable_params_path: Optional[Path] = None,
    push: bool = True,
) -> AutoTuningRunResult:
    """Ponto de entrada chamado por reports/daily_analysis_runner.py."""
    repo = repository or Repository()
    now = now or datetime.now(timezone.utc)
    run_tests = run_tests_fn or _default_run_tests
    git_is_clean = git_is_clean_fn or _default_git_is_clean
    commit = commit_fn or (lambda adjustment: _default_commit(adjustment, push=push))
    params_path = tunable_params_path or TUNABLE_PARAMS_PATH

    calibration = build_calibration_report(repo)
    backtest = build_backtest_report(repo)
    negative_roi_findings = find_negative_roi_segments(backtest.by_market)

    # Lê o "valor atual" do MESMO arquivo que será escrito (não do singleton `settings`,
    # que só reflete o estado do processo no momento em que foi importado e nunca é
    # atualizado depois -- todo o resto do código faz `from config import settings`, uma
    # cópia do valor congelada na hora do import, não uma referência viva) -- em produção
    # os dois coincidem (params_path == TUNABLE_PARAMS_PATH), mas isso também torna o
    # serviço testável com um arquivo isolado (ver tests/test_auto_tuning_service.py).
    on_disk = _read_tunable_params(params_path)
    current_values = {
        "overdispersion": on_disk.get("overdispersion", settings.overdispersion),
        "mean_uncertainty_pct": on_disk.get("mean_uncertainty_pct", settings.mean_uncertainty_pct),
        "min_edge": on_disk.get("min_edge", settings.min_edge),
        "min_confidence": on_disk.get("min_confidence", settings.min_confidence),
    }
    outcome: AutoTuningOutcome = propose_adjustments(
        reliability=calibration.reliability,
        overall_roi=backtest.overall.roi if backtest.overall.total_bets > 0 else None,
        overall_n=backtest.overall.total_bets,
        current_values=current_values,
        parameters_in_cooldown=_parameters_in_cooldown(repo, now),
    )

    if not outcome.accepted:
        return AutoTuningRunResult(
            changes=[], deferred=outcome.deferred, negative_roi_findings=negative_roi_findings,
            calibration=calibration, backtest=backtest,
        )

    if not git_is_clean():
        log.warning("auto_tuning_service: árvore git suja -- nenhum ajuste será aplicado hoje")
        return AutoTuningRunResult(
            changes=[], deferred=outcome.deferred + outcome.accepted, negative_roi_findings=negative_roi_findings,
            calibration=calibration, backtest=backtest,
            skipped_reason="árvore git com mudanças não commitadas -- nenhum ajuste foi tentado",
        )

    changes: List[AppliedChange] = []
    for adjustment in outcome.accepted:
        previous_values = _read_tunable_params(params_path)
        new_values = _bump_model_version(previous_values)
        new_values[adjustment.parameter_name] = adjustment.new_value
        _write_tunable_params(new_values, params_path)

        tests_passed = run_tests()
        sha: Optional[str] = None
        if tests_passed:
            sha = commit(adjustment)
        applied = tests_passed and sha is not None
        if not applied:
            _write_tunable_params(previous_values, params_path)

        repo.record_parameter_change(
            parameter_name=adjustment.parameter_name, old_value=adjustment.old_value,
            new_value=adjustment.new_value, rationale=adjustment.rationale,
            sample_size=adjustment.sample_size, applied=applied, git_commit_sha=sha,
        )
        changes.append(
            AppliedChange(
                adjustment=adjustment, applied=applied, git_commit_sha=sha,
                test_failure_summary=None if tests_passed else "suíte de testes falhou; valor anterior restaurado",
            )
        )

    return AutoTuningRunResult(
        changes=changes, deferred=outcome.deferred, negative_roi_findings=negative_roi_findings,
        calibration=calibration, backtest=backtest,
    )
