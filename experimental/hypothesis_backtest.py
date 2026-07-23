from __future__ import annotations

"""Harness de backtest para as hipóteses da pesquisa exploratória (Etapa 2), com o rigor
da Etapa 3 embutido: holdout out-of-sample walk-forward, correção de múltiplas comparações
(Benjamini-Hochberg / FDR), gate de breakeven (52,5% medido) e gate de amostra mínima.

ISOLADO DE PRODUÇÃO. Este módulo:
  - abre o banco de produção somente-leitura (sqlite mode=ro) e NUNCA escreve;
  - não importa nada de `mlb_quantitative_engine` (evita qualquer efeito colateral);
  - não envia Telegram, não mexe em tunable_params.json nem no crontab.

Filosofia de saída (restrição da Etapa 4): mostrar a DISTRIBUIÇÃO inteira de cada bucket
(quantis + histograma ASCII + contagem W/L), nunca só a média. Uma média esconde a cauda
que quebrou as projeções.

Fontes de dados por hipótese estão declaradas em HYPOTHESES[].source. Quando a fonte não
está plugada (ex.: Statcast via pybaseball, ausente), a feature retorna np.nan e a hipótese
é marcada NÃO TESTÁVEL — em vez de inventar número (restrição da Etapa 4).
"""

import argparse
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

# Breakeven REAL medido no histórico (odd decimal média 1,904 -> 1/1,904).
BREAKEVEN = 0.525
EDGE_MARGIN = 0.024  # exige-se edge >= breakeven + margem (o vig) para não ser ruído
FDR_Q = 0.10
POWER_Z = 0.84       # z para 80% de poder
DEFAULT_DB = Path(__file__).resolve().parent.parent / "mlb_quantitative_engine" / "database" / "database.db"


class DataUnavailable(RuntimeError):
    """Fonte de dados de uma feature não está plugada neste ambiente."""


# ---------------------------------------------------------------------------
# Estatística de rigor (Etapa 3)
# ---------------------------------------------------------------------------
def min_sample_size(p0: float, effect_pts: float, alpha: float, power_z: float = POWER_Z) -> int:
    """n mínimo (aprox. normal) para detectar um deslocamento `effect_pts` acima de p0."""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    p1 = p0 + effect_pts
    num = (z_alpha + power_z) ** 2 * p0 * (1 - p0)
    return math.ceil(num / (effect_pts ** 2)) if effect_pts > 0 else 10**9


def describe_distribution(name: str, values: np.ndarray, unit: str = "") -> str:
    """Distribuição INTEIRA (quantis + histograma ASCII), não a média sozinha."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return f"  {name}: (vazio)"
    qs = np.quantile(v, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    lines = [f"  {name}  n={v.size}{(' ' + unit) if unit else ''}",
             "    quantis  min={:+.2f} p10={:+.2f} q25={:+.2f} med={:+.2f} q75={:+.2f} p90={:+.2f} max={:+.2f}".format(*qs)]
    counts, edges = np.histogram(v, bins=min(10, max(3, v.size)))
    peak = counts.max() or 1
    for i, c in enumerate(counts):
        bar = "#" * int(round(20 * c / peak))
        lines.append(f"    [{edges[i]:+6.2f},{edges[i+1]:+6.2f}) {c:3d} {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registro de hipóteses
# ---------------------------------------------------------------------------
@dataclass
class Hypothesis:
    key: str
    title: str
    side_expected: str                 # direção prevista (para pré-registro; evita HARKing)
    source: str                        # fonte EXATA do dado
    expected_effect_pts: float         # efeito hipotetizado em pts de hit-rate (p/ n_min)
    extractor: Callable[[pd.DataFrame], pd.Series]  # -> feature numérica por aposta
    available: bool = True             # False => fonte não plugada => NÃO TESTÁVEL
    note: str = ""


# --- Extractors testáveis HOJE (schedule/feed/odds já plugados) --------------
def feat_is_dh_game2(df: pd.DataFrame) -> pd.Series:
    # H11 — derivável do schedule (doubleHeader in Y/S e gameNumber==2). No banco local
    # aproximamos por: mesmo par de times + mesma data com >1 game_pk, o de maior pk.
    df = df.copy()
    grp = df.groupby(["game_date", "home_team", "away_team"])["game_pk"]
    df["_rank"] = grp.rank(method="dense")
    df["_max"] = grp.transform("max")
    return ((df["game_pk"] == df["_max"]) & (grp.transform("count") > 1)).astype(float)


def feat_market_is_under(df: pd.DataFrame) -> pd.Series:
    # Não é hipótese nova por si — é o CONTROLE de resultado negativo (ver Etapa 4).
    return df["market"].str.endswith("_under").astype(float)


def feat_line_movement(df: pd.DataFrame) -> pd.Series:
    # H14 — exige 2 snapshots de linha (abertura + fechamento). O banco atual grava só 1
    # (coluna de abertura inexistente) => não disponível até instrumentar.
    if "opening_point" not in df.columns:
        raise DataUnavailable("H14: linha de abertura não é gravada (só 1 snapshot por jogo hoje)")
    return df["point"] - df["opening_point"]


def feat_umpire_run_index(df: pd.DataFrame) -> pd.Series:
    # H1 — precisa do backfill walk-forward das tendências do umpire. O ID do umpire vem
    # do feed (grátis), mas as tendências históricas não estão no banco de produção.
    if "umpire_run_index" not in df.columns:
        raise DataUnavailable("H1: índice de run environment do umpire exige backfill (ver build_umpire_index)")
    return df["umpire_run_index"]


def _statcast_stub(col: str, hyp: str) -> Callable[[pd.DataFrame], pd.Series]:
    def _f(df: pd.DataFrame) -> pd.Series:
        raise DataUnavailable(f"{hyp}: requer Statcast via pybaseball (ausente). Fonte: statcast(start,end).")
    return _f


HYPOTHESES: List[Hypothesis] = [
    Hypothesis("H14", "Movimento de linha abertura->fechamento", "seguir o movimento",
               "The Odds API (2 snapshots/dia)", 5.0, feat_line_movement, available=False,
               note="instrumentar 2 snapshots; limpo de leakage (closing < first pitch)"),
    Hypothesis("H1", "Umpire run environment (walk-forward)", "zona larga -> under",
               "MLB Stats API officials + boxscore (grátis)", 4.0, feat_umpire_run_index, available=False,
               note="ID do umpire já no feed; tendências precisam de backfill walk-forward"),
    Hypothesis("H11", "Segundo jogo de doubleheader", "DH jogo 2 -> under",
               "MLB schedule (doubleHeader/gameNumber)", 4.0, feat_is_dh_game2, available=True),
    Hypothesis("H9", "Resíduo xwOBA-wOBA (regressão da ofensa)", "over-performance -> under",
               "Statcast (pybaseball)", 3.0, _statcast_stub("xwoba_resid", "H9"), available=False),
    Hypothesis("H10", "Delta CSW% do SP", "CSW subindo -> under",
               "Statcast (pybaseball)", 3.0, _statcast_stub("csw_delta", "H10"), available=False),
    # CONTROLE (não é descoberta): expõe a assimetria over/under já observada.
    Hypothesis("CTRL_UNDER", "[controle] mercado é UNDER", "under -> pior hit rate",
               "banco (market)", 6.0, feat_market_is_under, available=True,
               note="resultado negativo conhecido; serve de sanity-check do harness"),
]


# ---------------------------------------------------------------------------
# Carga de dados
# ---------------------------------------------------------------------------
def load_local_resolved(db_path: Path) -> pd.DataFrame:
    """Lê o banco de PRODUÇÃO somente-leitura: apostas resolvidas + contexto de jogo."""
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query(
            """
            SELECT vb.id, vb.market, vb.price, vb.point, vb.projection_probability,
                   vb.edge, vb.expected_value, vb.outcome, vb.alert_sent,
                   g.game_pk, g.game_date, g.home_team, g.away_team, g.game_datetime
            FROM value_bets vb
            JOIN projections p ON p.id = vb.projection_id
            JOIN games g ON g.id = p.game_id
            WHERE vb.outcome IN ('win','loss')
            """,
            con,
        )
    finally:
        con.close()
    df["won"] = (df["outcome"] == "win").astype(int)
    return df


def load_statcast_seasons(seasons: List[int]) -> pd.DataFrame:
    """>=2 temporadas via pybaseball/Statcast + MLB Stats API. Não plugado neste ambiente."""
    try:
        import pybaseball  # noqa: F401
    except ImportError as exc:
        raise DataUnavailable(
            "pybaseball ausente. `pip install pybaseball` e reexecutar. "
            "Fonte histórica pretendida: pybaseball.statcast(start, end) para batted-ball/CSW; "
            "MLB Stats API (schedule+feed) para umpire/DH/day-night; The Odds API historical "
            "(endpoint pago) OU coleta prospectiva para movimento de linha."
        ) from exc
    raise DataUnavailable("Loader Statcast é um esqueleto: implementar join season->bet->resultado antes de usar.")


# ---------------------------------------------------------------------------
# Motor de backtest com holdout OOS
# ---------------------------------------------------------------------------
def evaluate(df_test: pd.DataFrame, hyp: Hypothesis) -> Dict:
    """Avalia UMA hipótese no conjunto de teste (OOS). Mostra a distribuição inteira."""
    if not hyp.available:
        return {"key": hyp.key, "status": "NAO_TESTAVEL", "reason": hyp.note or "fonte ausente"}
    try:
        feature = hyp.extractor(df_test)
    except DataUnavailable as exc:
        return {"key": hyp.key, "status": "NAO_TESTAVEL", "reason": str(exc)}

    d = df_test.assign(feature=feature.values).dropna(subset=["feature"])
    # Bucketização: binária -> 2 grupos; contínua -> tercis.
    if d["feature"].nunique() <= 2:
        d["bucket"] = np.where(d["feature"] > d["feature"].median(), "hi", "lo")
    else:
        d["bucket"] = pd.qcut(d["feature"], 3, labels=["baixo", "médio", "alto"], duplicates="drop")

    report = {"key": hyp.key, "title": hyp.title, "status": "OK", "buckets": [], "dist": []}
    best_edge, best_n, best_p = 0.0, 0, 1.0
    for b, g in d.groupby("bucket", observed=True):
        n = len(g)
        w = int(g["won"].sum())
        hit = w / n if n else float("nan")
        p = binomtest(w, n, BREAKEVEN, alternative="two-sided").pvalue if n else 1.0
        report["buckets"].append({"bucket": str(b), "n": n, "w": w, "l": n - w,
                                  "hit": hit, "edge_vs_be": hit - BREAKEVEN, "p": p})
        report["dist"].append(describe_distribution(f"edge por aposta [{b}]",
                                                    g["won"].values - BREAKEVEN))
        if abs(hit - BREAKEVEN) > abs(best_edge):
            best_edge, best_n, best_p = hit - BREAKEVEN, n, p
    n_min = min_sample_size(BREAKEVEN, hyp.expected_effect_pts / 100.0, alpha=FDR_Q)
    report.update({"best_edge": best_edge, "best_n": best_n, "raw_p": best_p, "n_min": n_min})
    return report


def run(source: str, db_path: Path, seasons: List[int], holdout: Optional[str]) -> None:
    print("=" * 78)
    print(f"BACKTEST DE HIPÓTESES — fonte={source}  breakeven={BREAKEVEN:.3f}  FDR q={FDR_Q}")
    print("=" * 78)

    if source == "local":
        df = load_local_resolved(db_path)
    elif source == "statcast":
        try:
            df = load_statcast_seasons(seasons)
        except DataUnavailable as exc:
            print(f"\n[FONTE INDISPONÍVEL] {exc}\n")
            print("Nada a testar até a fonte histórica ser plugada. Encerrando sem inventar dados.")
            return
    else:
        raise SystemExit(f"fonte desconhecida: {source}")

    # Split walk-forward OOS: treina/olha até holdout, testa DEPOIS.
    if holdout:
        df_test = df[df["game_date"] >= holdout].copy()
        split_msg = f"holdout OOS: testando em game_date >= {holdout}"
    else:
        cut = df["game_date"].sort_values().unique()
        cut = cut[len(cut) // 2] if len(cut) > 1 else cut[0]
        df_test = df[df["game_date"] >= cut].copy()
        split_msg = f"holdout OOS automático: testando em game_date >= {cut} (metade final)"

    print(f"\nAmostra total resolvida: {len(df)} | {split_msg} | teste OOS: {len(df_test)}\n")

    results = [evaluate(df_test, h) for h in HYPOTHESES]

    # Correção de múltiplas comparações (Benjamini-Hochberg) só sobre as testáveis.
    testable = [r for r in results if r["status"] == "OK"]
    if testable:
        pvals = [r["raw_p"] for r in testable]
        rej, qvals, _, _ = multipletests(pvals, alpha=FDR_Q, method="fdr_bh")
        for r, q, rj in zip(testable, qvals, rej):
            r["q"] = q
            r["fdr_reject"] = bool(rj)

    for r in results:
        print("-" * 78)
        if r["status"] == "NAO_TESTAVEL":
            print(f"[{r['key']}] NÃO TESTÁVEL — {r['reason']}")
            continue
        print(f"[{r['key']}] {r['title']}")
        for b in r["buckets"]:
            flag = "  <-- ruído (|edge|<vig)" if abs(b["edge_vs_be"]) < EDGE_MARGIN else ""
            print(f"   bucket={b['bucket']:>6}  n={b['n']:3d}  W{b['w']:3d}/L{b['l']:3d}  "
                  f"hit={b['hit']*100:5.1f}%  edge_vs_BE={b['edge_vs_be']*100:+5.1f}pt  p={b['p']:.3f}{flag}")
        for line in r["dist"]:
            print(line)
        verdict = _verdict(r)
        print(f"   n_min(poder 80%, FDR)={r['n_min']}  best_n={r['best_n']}  "
              f"q_BH={r.get('q', float('nan')):.3f}  => {verdict}")

    print("\n" + "=" * 78)
    print("LEGENDA DO VEREDITO: PROMOVER exige n>=n_min, |edge|>=vig e q<=0.10 no OOS.")
    print("Com o histórico atual (5 dias) o esperado é INSUFICIENTE em tudo — por design.")
    print("=" * 78)


def _verdict(r: Dict) -> str:
    if r["best_n"] < r["n_min"]:
        return f"INSUFICIENTE (n={r['best_n']} < n_min={r['n_min']})"
    if abs(r["best_edge"]) < EDGE_MARGIN:
        return "REJEITADA (edge < vig: ruído)"
    if not r.get("fdr_reject", False):
        return f"REJEITADA (q={r.get('q', 1):.3f} > {FDR_Q})"
    return "PROMOVER (sobreviveu a n_min + vig + FDR)"


def build_umpire_index() -> None:
    """Esqueleto do backfill de H1 (walk-forward, sem leakage):
    para cada temporada histórica, iterar o schedule; para cada jogo, ler
    liveData.boxscore.officials -> Home Plate + total de corridas do boxscore;
    o índice de um umpire numa data D = média de runs dos jogos que ele apitou
    ESTRITAMENTE antes de D / média da liga até D, com encolhimento p/ n baixo.
    Fonte: MLB Stats API (grátis). NÃO usar stats que incluam o próprio jogo previsto.
    """
    raise NotImplementedError("backfill de umpire: implementar contra MLB Stats API antes de testar H1")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest isolado de hipóteses (não toca produção).")
    ap.add_argument("--source", choices=["local", "statcast"], default="local")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--seasons", type=int, nargs="*", default=[2024, 2025])
    ap.add_argument("--holdout", type=str, default=None, help="game_date de início do teste OOS (YYYY-MM-DD)")
    args = ap.parse_args()
    run(args.source, args.db, args.seasons, args.holdout)


if __name__ == "__main__":
    main()
