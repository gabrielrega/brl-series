# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/);
versionamento conforme [SemVer](https://semver.org/lang/pt-BR/).

## [Não lançado]

### Adicionado
- Análise GARCH(1,1) portada para a estrutura modular (`garch_analysis.py`,
  `run_garch_analysis`): estimação com inovação t de Student (fallback Normal),
  teste ARCH-LM, diagnósticos de resíduos e previsão de volatilidade a 1 ano.
- Análise VAR bivariada USD/BRL + SELIC portada para a estrutura modular
  (`var_analysis.py`, `run_var_analysis`): seleção de defasagem por AIC, teste de
  causalidade de Granger, funções de resposta a impulso e previsão de nível a 1 ano.
- Avaliação de volatilidade na CV rolling-origin compartilhada
  (`evaluation.rolling_origin_vol_cv`, `log_returns`, `realized_vol`) com baseline
  de volatilidade constante (`constant_vol_forecast`), análogo de volatilidade ao
  baseline random-walk de nível.
- `main.py` passa a baixar a série SELIC (BCB 432) e exibe duas tabelas de
  comparação: nível (RW, ARIMA, ETS, Prophet, VAR) e volatilidade (Const Vol, GARCH).
- Testes offline das novas funções (`test_models.py`).

### Modificado
- VAR entra na tabela de comparação de nível usando a mesma CV
  (initial=750, period=120, horizon=60) dos demais modelos univariados.
- Supressão de warnings passa a ser por categoria (ARIMA/VAR/GARCH) em vez de
  `filterwarnings("ignore")` global: silencia apenas o ruído benigno (índice sem
  frequência, KPSS nas bordas da tabela, escala/valores iniciais da `arch`) e
  mantém `ConvergenceWarning` visível para não esconder falhas de ajuste.
- Grid search do ARIMA deixa de usar `except` nu: captura `Exception` e registra
  o motivo do descarte de cada ordem.

## [0.1.0] - 2026-06-01
### Adicionado
- Estrutura modular: `main.py` + análises ARIMA/ETS/Prophet.
- CV rolling-origin unificada (`evaluation.rolling_origin_cv`) e baseline
  random-walk (`naive_rw_forecast`) para comparação apples-to-apples.
