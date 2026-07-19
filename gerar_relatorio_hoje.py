import time
from mlb_quantitative_engine.reports.report_generator import ReportGenerator, rows_to_dataframe
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.api.mlb_api import MLBApiClient
import os

date = "2026-07-18"
db_path = os.path.join(os.getcwd(), "mlb_quantitative_engine", "database", "database.db")
repo = Repository(db_path=db_path)
client = MLBApiClient()
generator = ReportGenerator(api_client=client, repository=repo, season=2026)

games = client.get_games_for_date(date)
print(f"Jogos encontrados: {len(games)}", flush=True)
all_odds = generator._fetch_all_odds()
print(f"Odds buscadas: {len(all_odds)} jogos com mercado", flush=True)

rows = []
for i, game in enumerate(games, 1):
    t0 = time.time()
    row = generator._build_row(game, all_odds)
    rows.append(row)
    print(f"[{i}/{len(games)}] {game.away_team} @ {game.home_team} -> "
          f"total={row.projected_total_runs} rec={row.value_bet_recommendation} "
          f"skip={row.skip_reason} ({round(time.time()-t0,1)}s)", flush=True)

df = rows_to_dataframe(rows)
df.to_csv("relatorio_2026-07-18.csv", index=False, encoding="utf-8-sig")
print(f"OK, linhas: {len(df)}", flush=True)
print(f"Creditos odds usados: {generator.odds_service.api_client.last_requests_used}, "
      f"restantes: {generator.odds_service.api_client.last_requests_remaining}", flush=True)
