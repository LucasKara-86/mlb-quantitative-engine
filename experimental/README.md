# experimental/ — pesquisa de sinais fora da lista negra

**Isolado do fluxo de produção.** Nada aqui é importado por `mlb_quantitative_engine/`.
O backtest abre o banco de produção **somente-leitura** (`mode=ro`) e nunca escreve nele,
nunca chama Telegram, nunca toca em `tunable_params.json` nem no crontab.

Rodar não altera nenhuma tip. É pesquisa, não produção.

## Arquivos
- `hypothesis_backtest.py` — harness de backtest com holdout out-of-sample (walk-forward),
  correção de múltiplas comparações (Benjamini-Hochberg / FDR), gate de breakeven e de
  amostra mínima (ver Etapa 3). Mostra a **distribuição inteira** de cada bucket, não médias.

## Uso
```bash
# Modo local: roda sobre o histórico do banco de produção (hoje: ~5 dias — insuficiente,
# serve para exercitar o pipeline e ver os gates reprovarem por amostra).
.venv/bin/python -m experimental.hypothesis_backtest --source local

# Modo histórico: exige >=2 temporadas via pybaseball/Statcast + MLB Stats API.
# pybaseball ainda NÃO está instalado (pip install pybaseball). Sem ele, as features
# de Statcast retornam "não disponível" e o backtest as marca como NÃO TESTÁVEL.
.venv/bin/python -m experimental.hypothesis_backtest --source statcast --seasons 2024 2025 \
    --holdout 2025-08-01
```

## Critério de aprovação (não negociável)
Uma hipótese só é promovida para "candidata a produção" se, no **holdout OOS**:
1. `n >= n_min` (poder 80% para o efeito declarado, já com correção FDR) — ver tabela na Etapa 3;
2. edge ponto-estimado `>= breakeven + 0.024` (o vig real, medido = 52,5%);
3. `q_value <= 0.10` (Benjamini-Hochberg sobre TODAS as hipóteses testadas juntas).

Enquanto qualquer um falhar, o veredito é **INSUFICIENTE / REJEITADA** — nunca "promissora".
