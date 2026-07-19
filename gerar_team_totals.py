import pandas as pd

from mlb_quantitative_engine.analytics.poisson import probability_over, probability_under

df = pd.read_csv("relatorio_2026-07-18.csv")

def compute_team_totals(row):
    if pd.isna(row["projected_total_runs"]) or pd.isna(row["market_total_line"]):
        return pd.Series({
            "home_team_total_line": None, "home_team_probability_over": None, "home_team_probability_under": None,
            "away_team_total_line": None, "away_team_probability_over": None, "away_team_probability_under": None,
        })

    total_projected = row["projected_total_runs"]
    home_share = row["projected_home_runs"] / total_projected
    away_share = row["projected_away_runs"] / total_projected

    # linha de time implícita: linha real do jogo (mercado) distribuída pela participação
    # de cada time na NOSSA projeção total, arredondada para o meio-ponto mais próximo
    # (convenção usual de linhas de total: terminam em .5)
    home_line = round(row["market_total_line"] * home_share * 2) / 2
    away_line = round(row["market_total_line"] * away_share * 2) / 2

    return pd.Series({
        "home_team_total_line": home_line,
        "home_team_probability_over": round(probability_over(row["projected_home_runs"], home_line), 4),
        "home_team_probability_under": round(probability_under(row["projected_home_runs"], home_line), 4),
        "away_team_total_line": away_line,
        "away_team_probability_over": round(probability_over(row["projected_away_runs"], away_line), 4),
        "away_team_probability_under": round(probability_under(row["projected_away_runs"], away_line), 4),
    })

team_totals = df.apply(compute_team_totals, axis=1)
result = pd.concat([df, team_totals], axis=1)
result.to_csv("relatorio_2026-07-18_com_team_totals.csv", index=False, encoding="utf-8-sig")

cols = [
    "home_team", "away_team", "projected_home_runs", "home_team_total_line", "home_team_probability_over",
    "projected_away_runs", "away_team_total_line", "away_team_probability_over",
]
pd.set_option("display.width", 220)
print(result[cols].to_string())
