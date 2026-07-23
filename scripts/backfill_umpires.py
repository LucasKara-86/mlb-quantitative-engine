from __future__ import annotations

"""Backfill de umpires de home plate + ambiente de corridas por jogo (2024, 2025 e 2026
até ontem). ACUMULAÇÃO DE BASE — não conecta nada ao modelo (ver ITEM 3).

Isolado: grava num banco PRÓPRIO (scripts/data/umpire_games.db por padrão), nunca no
banco de produção. Não importa `mlb_quantitative_engine`. Só lê a MLB Stats API (grátis).

Fonte por jogo: endpoint `game/{pk}/boxscore` (v1) — é a versão standalone e leve do
`liveData.boxscore` do feed/live citado no pedido; carrega o MESMO bloco `officials`
(Home Plate) e as corridas por time (`teams.{home,away}.teamStats.batting.runs`). Usar o
boxscore em vez do feed/live completo poupa banda num backfill de ~7 mil jogos.

Robustez (roda desacompanhado):
  - rate limit configurável entre requisições (--sleep, padrão 0.35s);
  - retry com backoff exponencial + jitter em falha de rede / 429 / 5xx;
  - checkpoint retomável: a própria tabela `backfill_log` registra cada gamePk como
    ok/skipped; ao reiniciar, pula o que já foi feito (retoma do jogo 1.400, não do zero);
  - Ctrl-C faz commit do progresso e sai limpo.

IMPORTANTE (walk-forward, para quando o índice for derivado depois): qualquer métrica de
umpire terá que usar SÓ jogos com game_date ANTERIOR à data da aposta. Este script apenas
acumula a base bruta; não computa nenhum índice.

Uso:
    python scripts/backfill_umpires.py                      # 2024..2026(ontem), banco próprio
    python scripts/backfill_umpires.py --start-season 2025  # retomar/continuar
    python scripts/backfill_umpires.py --sleep 0.5          # mais conservador
"""

import argparse
import random
import signal
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

API_V1 = "https://statsapi.mlb.com/api/v1"
DEFAULT_DB = Path(__file__).resolve().parent / "data" / "umpire_games.db"
# Temporada regular + pós-temporada; exclui spring (S), all-star (A), exhibition (E).
DEFAULT_GAME_TYPES = ("R", "F", "D", "L", "W")
FINAL_PREFIXES = ("Final", "Game Over", "Completed Early")

_stop = False


def _handle_sigint(_sig, _frame) -> None:
    global _stop
    _stop = True
    print("\n[sinal] encerrando após o jogo atual — progresso será salvo...", flush=True)


# ---------------------------------------------------------------------------
# HTTP com backoff
# ---------------------------------------------------------------------------
def http_get(url: str, params: Optional[dict], sleep: float, max_attempts: int = 5) -> dict:
    """GET com retry/backoff exponencial + jitter. Respeita `sleep` entre chamadas bem-sucedidas."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, params=params, timeout=25)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.RequestException(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            time.sleep(sleep)  # rate limit respeitoso após sucesso
            return data
        except (requests.RequestException, ValueError) as exc:
            if attempt >= max_attempts:
                raise
            backoff = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.0)
            print(f"  [retry {attempt}/{max_attempts}] {url} :: {exc} -> aguardando {backoff:.1f}s", flush=True)
            time.sleep(backoff)


# ---------------------------------------------------------------------------
# Banco próprio (checkpoint)
# ---------------------------------------------------------------------------
def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS umpire_games (
            ump_id      INTEGER,
            ump_name    TEXT,
            game_pk     INTEGER UNIQUE NOT NULL,
            game_date   TEXT NOT NULL,
            total_runs  INTEGER,
            park_id     INTEGER
        )""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_ug_ump ON umpire_games(ump_id)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_ug_date ON umpire_games(game_date)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS backfill_log (
            game_pk    INTEGER PRIMARY KEY,
            status     TEXT NOT NULL,          -- ok | skipped
            note       TEXT,
            updated_at TEXT NOT NULL
        )""")
    con.commit()
    return con


def already_done(con: sqlite3.Connection) -> set:
    return {r[0] for r in con.execute("SELECT game_pk FROM backfill_log WHERE status IN ('ok','skipped')")}


def mark(con: sqlite3.Connection, game_pk: int, status: str, note: str = "") -> None:
    con.execute(
        "INSERT INTO backfill_log(game_pk,status,note,updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(game_pk) DO UPDATE SET status=excluded.status, note=excluded.note, updated_at=excluded.updated_at",
        (game_pk, status, note, datetime.now(timezone.utc).isoformat()),
    )


# ---------------------------------------------------------------------------
# Worklist a partir do schedule (por mês, para não puxar temporada inteira num payload)
# ---------------------------------------------------------------------------
def month_ranges(start: date, end: date) -> Iterable[Tuple[str, str]]:
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        lo = max(cur, start)
        hi = min(nxt - timedelta(days=1), end)
        yield lo.isoformat(), hi.isoformat()
        cur = nxt


def fetch_worklist(start: date, end: date, game_types: Tuple[str, ...], sleep: float) -> List[dict]:
    """Lista de jogos (gamePk, officialDate, venue_id, status) no intervalo, filtrada por gameType."""
    work: List[dict] = []
    for lo, hi in month_ranges(start, end):
        payload = http_get(API_V1 + "/schedule",
                           {"sportId": 1, "startDate": lo, "endDate": hi, "hydrate": "venue"}, sleep)
        for day in payload.get("dates", []):
            for g in day.get("games", []):
                if g.get("gameType") not in game_types:
                    continue
                work.append({
                    "game_pk": g["gamePk"],
                    "game_date": g.get("officialDate") or day.get("date"),
                    "venue_id": g.get("venue", {}).get("id"),
                    "status": g.get("status", {}).get("detailedState", ""),
                })
        print(f"  schedule {lo}..{hi}: acumulado {len(work)} jogos", flush=True)
    return work


# ---------------------------------------------------------------------------
# Extração por jogo
# ---------------------------------------------------------------------------
def _is_final(status: str) -> bool:
    return any((status or "").startswith(p) for p in FINAL_PREFIXES)


def extract_game(box: dict) -> Optional[dict]:
    """Do boxscore: HP umpire + corridas totais. None se faltar umpire ou placar."""
    hp = None
    for off in box.get("officials", []):
        if off.get("officialType") == "Home Plate":
            hp = off.get("official", {})
            break
    if not hp or hp.get("id") is None:
        return None
    home = box.get("teams", {}).get("home", {}).get("teamStats", {}).get("batting", {}).get("runs")
    away = box.get("teams", {}).get("away", {}).get("teamStats", {}).get("batting", {}).get("runs")
    if home is None or away is None:
        return None
    return {"ump_id": int(hp["id"]), "ump_name": hp.get("fullName"), "total_runs": int(home) + int(away)}


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
def run(db_path: Path, start_season: int, end_season: int, until: date,
        game_types: Tuple[str, ...], sleep: float, since: Optional[date] = None) -> None:
    con = open_db(db_path)
    signal.signal(signal.SIGINT, _handle_sigint)

    start = max(date(start_season, 1, 1), since) if since else date(start_season, 1, 1)
    end = min(date(end_season, 12, 31), until)
    print(f"Backfill umpires -> {db_path}")
    print(f"Intervalo: {start} .. {end} | gameTypes={game_types} | sleep={sleep}s")

    print("Montando worklist a partir do schedule...")
    work = fetch_worklist(start, end, game_types, sleep)
    done = already_done(con)
    pending = [w for w in work if w["game_pk"] not in done]
    print(f"Total no intervalo: {len(work)} | já processados: {len(done)} | pendentes: {len(pending)}\n")

    processed = 0
    for w in pending:
        if _stop:
            break
        pk, status = w["game_pk"], w["status"]
        if not _is_final(status):
            mark(con, pk, "skipped", f"não-final: {status}")
            con.commit()
            continue
        try:
            box = http_get(f"{API_V1}/game/{pk}/boxscore", None, sleep)
        except Exception as exc:  # noqa: BLE001 - erro fica sem marca -> reprocessa no próximo run
            print(f"  [erro] gamePk {pk}: {exc} (será retentado numa próxima execução)", flush=True)
            continue

        data = extract_game(box)
        if data is None:
            mark(con, pk, "skipped", "sem umpire HP ou sem placar")
            con.commit()
            continue

        con.execute(
            "INSERT OR IGNORE INTO umpire_games(ump_id,ump_name,game_pk,game_date,total_runs,park_id) "
            "VALUES(?,?,?,?,?,?)",
            (data["ump_id"], data["ump_name"], pk, w["game_date"], data["total_runs"], w["venue_id"]),
        )
        mark(con, pk, "ok")
        con.commit()

        processed += 1
        if processed % 100 == 0:
            print(f"  ...{processed}/{len(pending)} processados (último: {w['game_date']} pk={pk})", flush=True)

    con.commit()
    total = con.execute("SELECT COUNT(*) FROM umpire_games").fetchone()[0]
    umps = con.execute("SELECT COUNT(DISTINCT ump_id) FROM umpire_games").fetchone()[0]
    con.close()
    state = "INTERROMPIDO (retomável)" if _stop else "CONCLUÍDO"
    print(f"\n{state}. Novos neste run: {processed}. Base total: {total} jogos, {umps} umpires distintos.")


def main() -> None:
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1))
    ap = argparse.ArgumentParser(description="Backfill isolado de umpires (não conecta ao modelo).")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--start-season", type=int, default=2024)
    ap.add_argument("--end-season", type=int, default=yesterday.year)
    ap.add_argument("--until", type=str, default=yesterday.isoformat(), help="data limite YYYY-MM-DD (padrão: ontem)")
    ap.add_argument("--since", type=str, default=None, help="data inicial YYYY-MM-DD (sobrepõe o 1º de janeiro; útil p/ testes)")
    ap.add_argument("--game-types", type=str, default=",".join(DEFAULT_GAME_TYPES))
    ap.add_argument("--sleep", type=float, default=0.35, help="segundos entre requisições")
    args = ap.parse_args()
    try:
        until = datetime.strptime(args.until, "%Y-%m-%d").date()
        since = datetime.strptime(args.since, "%Y-%m-%d").date() if args.since else None
    except ValueError:
        sys.exit(f"data inválida em --until/--since: {args.until} / {args.since}")
    run(args.db, args.start_season, args.end_season, until,
        tuple(t.strip() for t in args.game_types.split(",") if t.strip()), args.sleep, since)


if __name__ == "__main__":
    main()
