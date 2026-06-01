# Relatório de Análise de Séries Temporais: Previsão da Taxa de Câmbio BRL/USD

## Introdução

Este relatório apresenta os resultados de uma análise de séries temporais da taxa de câmbio BRL/USD. O objetivo do projeto foi comparar o desempenho de três modelos de previsão diferentes: ARIMA, Prophet e ETS.

## Modelos Utilizados

Foram utilizados três modelos de previsão de séries temporais univariadas:

*   **ARIMA (Autoregressive Integrated Moving Average):** Um modelo estatístico clássico que utiliza os valores passados da série temporal para prever valores futuros.
*   **Prophet:** Uma biblioteca de previsão de séries temporais de código aberto desenvolvida pelo Facebook. É projetado para ser fácil de usar e robusto para uma ampla variedade de séries temporais.
*   **ETS (Exponential Smoothing):** Um modelo que atribui pesos exponencialmente decrescentes às observações passadas. Configurado com **tendência aditiva amortecida e sem sazonalidade** — a validação cruzada mostrou que qualquer componente sazonal (períodos de 5, 12, 21 ou 252 dias) era inerte para esta série, enquanto o amortecimento da tendência melhorou as previsões a 60 dias.

## Metodologia de Avaliação

Para que a comparação entre os modelos seja justa, **os três são avaliados sob o mesmo
protocolo**: validação cruzada *rolling-origin* (origem móvel), implementada em
`evaluation.py`. A cada corte, o modelo é treinado apenas com os dados até aquele ponto e
prevê o mesmo horizonte de 60 dias úteis, nas mesmas datas reais. As métricas (MAE, MAPE,
RMSE) são calculadas sobre o conjunto agrupado de todas as previsões.

Parâmetros: janela inicial de treino de 750 observações (~3 anos), passo de 120 observações
(~6 meses) entre cortes e horizonte de 60 observações (~3 meses) — resultando em 4 folds.

Incluímos também um **baseline ingênuo (random walk)** — prever que a taxa dos próximos 60
dias é igual ao último valor observado. Para uma série de câmbio, este é o benchmark que
qualquer modelo precisa superar para justificar sua complexidade.

> **Nota metodológica:** versões anteriores deste relatório comparavam métricas calculadas de
> formas diferentes para cada modelo (ARIMA por *walk-forward* de 1 passo, ETS por previsão
> estática com erro de alinhamento, Prophet por *cross-validation*). Isso favorecia
> artificialmente o ARIMA (MAPE de ~0,4%, irrealista para câmbio). Sob o protocolo unificado,
> os erros refletem a real dificuldade de prever a série a 60 dias.

## Resultados

As métricas utilizadas são o Erro Absoluto Médio (MAE), o Erro Percentual Absoluto Médio
(MAPE) e a Raiz do Erro Quadrático Médio (RMSE), todas em validação cruzada de 4 folds. Dados:
1254 observações de 2021-06-02 a 2026-05-29.

| Modelo        | MAE    | MAPE   | RMSE   |
|---------------|--------|--------|--------|
| **Naive RW**  | 0.1753 | 0.0308 | 0.2296 |
| ARIMA         | 0.1799 | 0.0317 | 0.2348 |
| ETS           | 0.1753 | 0.0308 | 0.2296 |
| Prophet       | 0.1975 | 0.0348 | 0.2465 |

## Análise dos Resultados

A leitura crítica vem da comparação com o baseline — e o resultado é contundente:

*   O **random walk ingênuo é o melhor** (empatado com o ETS). Nenhum modelo o supera.
*   O **ETS** (tendência amortecida, sem sazonalidade) é **numericamente idêntico ao random
    walk**: a configuração ótima colapsa para "prever o último valor", sem adicionar informação.
*   O **ARIMA** fica **pior** que o ingênuo (MAPE 3,17% vs 3,08%), apesar de toda a sua
    complexidade (identificação, grid search, diagnósticos de resíduo).
*   O **Prophet** é o pior dos quatro (MAPE 3,48%).

Este resultado é esperado e bem documentado na literatura (o *puzzle* de Meese-Rogoff): taxas
de câmbio se comportam muito próximo de um passeio aleatório, e modelos univariados raramente
batem o ingênuo fora da amostra em horizontes de meses.

> **Sensibilidade temporal:** com os dados em cache até 2026-01-16, o ARIMA superava o RW por
> uma margem mínima; com os dados até 2026-05-29 ele passa a perder para o ingênuo. A própria
> instabilidade do ranking reforça que as diferenças entre os modelos são ruído, não sinal.

## Conclusão

Para a série BRL/USD, sob avaliação justa e contra um baseline, **o passeio aleatório ("amanhã
≈ hoje") é tão bom ou melhor que qualquer um dos três modelos testados**. O ETS não se
distingue dele; ARIMA e Prophet ficam atrás. Nenhum dos modelos univariados justifica sua
complexidade neste horizonte.

Para genuinamente melhorar a previsão seria necessário ir além de modelos univariados — por
exemplo, incorporar variáveis exógenas (diferencial de juros, commodities, índice do dólar) ou
focar em prever volatilidade em vez do nível. O desempenho também pode variar com o tempo e as
condições de mercado, então recomenda-se reavaliar periodicamente.
