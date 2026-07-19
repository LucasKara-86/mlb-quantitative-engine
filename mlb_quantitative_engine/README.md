# MLB Quantitative Betting Engine

Sistema quantitativo profissional para projetar o número esperado de corridas
de cada equipe da MLB usando sabermetria e modelos probabilísticos, e para
identificar apostas de valor (Value Bets) comparando a projeção com as odds
disponíveis no mercado.

O objetivo do projeto **não** é automatizar apostas — é construir um motor de
análise quantitativa com rastreabilidade completa dos dados, precisão
estatística e arquitetura pronta para evoluir.

## Arquitetura

Camadas completamente separadas:

- `api/` — adaptadores para fontes externas (MLB Stats API, Odds API, clima, notícias, lineups, umpire)
- `services/` — regras de domínio que combinam e normalizam dados das APIs (lineup, bullpen, park factor, clima, lesões, odds)
- `analytics/` — modelagem estatística e matemática (sabermetria, projeções, Monte Carlo, Poisson, regressão, calibração)
- `models/` — estruturas de dados tipadas (projeções, value bets, bankroll)
- `database/` — persistência (SQLite via SQLAlemy/sqlite3)
- `reports/` — geração de relatórios e (futuramente) dashboard
- `utils/` — logging, cache e helpers transversais
- `tests/` — testes unitários e de integração (cobertura mínima de 90%)

## Etapa 1 — Fundação do projeto

Esta etapa estabelece apenas a base da aplicação, sem lógica de negócio:

- Estrutura completa de diretórios em camadas, cada um como pacote Python (`__init__.py`).
- `config.py`: configuração centralizada e validada com **Pydantic Settings**, carregada de `.env`. Inclui URLs base das APIs, caminho do banco, número de simulações Monte Carlo e os limiares mínimos de Value Bet (EV > 5%, Edge > 4%, Confiança > 70%) definidos na especificação do projeto.
- `utils/logger.py`: logging estruturado com **Loguru** (console colorido + arquivo rotativo diário em `logs/`, retenção de 30 dias).
- `app.py`: entrypoint mínimo que apenas valida se configuração e logging estão funcionando corretamente.
- `requirements.txt`: dependências do projeto (nota: `sqlite3` foi removida da lista original por ser módulo nativo do Python — instalá-la via pip falharia).
- `.env`: variáveis de ambiente (chaves de API). Placeholder vazio — preencha `ODDS_API_KEY` antes da etapa de integração de odds.
- `tests/test_config.py`: valida os valores padrão e o carregamento via variável de ambiente.

Nenhuma chamada de API externa, banco de dados ou modelo estatístico foi implementado ainda — isso será construído nas próximas etapas, uma de cada vez.

## Como executar

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r mlb_quantitative_engine\requirements.txt

# Validar a fundação (Etapa 1)
python -m mlb_quantitative_engine.app

# Rodar os testes
pytest mlb_quantitative_engine\tests -v
```
