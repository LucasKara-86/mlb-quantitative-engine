from __future__ import annotations

from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.utils.logger import log


def main() -> None:
    """Ponto de entrada da aplicação.

    Nesta Etapa 1, valida apenas a fundação do projeto: carregamento de
    configuração (config.py) e logging estruturado (utils/logger.py).
    As etapas seguintes irão acoplar aqui a leitura de jogos, projeções e relatórios.
    """
    log.info("MLB Quantitative Engine — fundação do projeto inicializada")
    log.info(f"MLB Stats API base URL: {settings.mlb_api_base_url}")
    log.info(f"Banco de dados configurado em: {settings.database_path}")
    log.info(f"Simulações Monte Carlo configuradas: {settings.monte_carlo_simulations:,}")

    if not settings.odds_api_key:
        log.warning(
            "ODDS_API_KEY não configurada em .env — necessária quando a integração "
            "de odds for implementada em uma etapa futura."
        )


if __name__ == "__main__":
    main()
