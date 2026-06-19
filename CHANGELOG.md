# Changelog

Todas as mudanças notáveis deste projeto são documentadas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/);
versionamento conforme [SemVer](https://semver.org/lang/pt-BR/).

## [Não lançado]

## [0.2.0] - 2026-06-19

### Adicionado
- VAR passa a usar o **diferencial de juros Brasil–EUA** (`SELIC - Fed Funds`)
  como variável de taxa, em vez da SELIC pura — a UIP é sobre o *diferencial*.
  `main.py` baixa a Fed Funds do FRED (`download_fred_data`, série `DFF`, sem API
  key; `parse_fred_csv` isolado e testável) e alinha a taxa diária ao índice de
  dias úteis com forward-fill. Se o FRED estiver indisponível, o VAR cai de volta
  à SELIC pura. `run_var_analysis` ganha o parâmetro `rate_label`.
- **GARCH avaliado em múltiplos horizontes** (Fase 4b, `VOL_HORIZONS = (5, 21,
  60)`): cada horizonte ladrilha as próprias janelas de teste (`period == horizon`),
  rendendo muito mais folds nos prazos curtos e poder ao teste DM. **Achado:** em
  5 dias (100 folds) o GARCH bate a volatilidade constante de forma significativa
  (DM −3.47, p 0.001); o ganho some em 21 dias (p 0.28) e 60 dias (p 0.62) —
  clustering paga no curto prazo, consistente com Meese-Rogoff (que é sobre o
  nível, não a volatilidade).
- Testes offline (`test_models.py`): `parse_fred_csv` (cabeçalho atual e legado,
  tratamento de `.`) e mais folds em horizonte curto na CV de volatilidade.

### Adicionado
- `CLAUDE.md` a nível de projeto descrevendo a arquitetura modular atual
  (orquestração em `main.py`, CV unificada + Diebold-Mariano em `evaluation.py`,
  GARCH avaliado em CV de volatilidade à parte, convenções de `data/`/`assets/`
  e de silenciamento de warnings). O antigo havia sido removido no merge do PR
  #2 por documentar a arquitetura monolítica obsoleta.

### Corrigido
- `garch_analysis.py`: remove o `warnings.filterwarnings("ignore")` global que
  silenciava tudo — inclusive o `ConvergenceWarning` que o módulo pretende
  manter visível para sinalizar janelas que falham ao ajustar (e caem no
  fallback Normal). Os filtros benignos (`DataScaleWarning`,
  `StartingValueWarning`) já são tratados pontualmente.

### Modificado
- `main.py`: chamada à API do BCB passa a usar `https://`.
- Limpeza de imports mortos em `arima_analysis.py`, `ets_analysis.py` e
  `prophet_analysis.py` (métricas do sklearn, `sys`, `datetime`, `numpy`) que
  ficaram obsoletos após a CV migrar para `evaluation.py`.

### Adicionado
- Teste de Diebold-Mariano (`evaluation.diebold_mariano`) com correção de
  amostra pequena de Harvey-Leybourne-Newbold e variância de longo prazo
  Newey-West (truncada em `horizon-1`), para aferir se a diferença de acurácia
  entre dois modelos é significativa em vez de comparar só o MAE pontual.
- A CV rolling-origin (nível e volatilidade) passa a devolver os erros por
  ponto (`errors`, indexados por `(cutoff, data)`) para alimentar o DM.
- `main.py` ganha colunas `DM vs RW` (nível) e `DM vs CV` (volatilidade).
- Teste offline do DM (`test_models.py`): erros idênticos → stat≈0/p≈1; modelo
  uniformemente melhor → stat negativo significativo; sem interseção → None.
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
- `PERIOD` da CV reduzido de 120 para 60 (janelas de teste contíguas): de 4 para
  8 folds. Com mais folds e o teste DM, a vantagem do VAR sobre o random walk
  deixa de ser detectável (p = 0.87) — resultado mais honesto e Meese-Rogoff.
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
