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
(MAPE) e a Raiz do Erro Quadrático Médio (RMSE), todas em validação cruzada de 4 folds.

| Modelo        | MAE    | MAPE   | RMSE   |
|---------------|--------|--------|--------|
| **Naive RW**  | 0.1340 | 0.0242 | 0.1806 |
| ARIMA         | 0.1305 | 0.0236 | 0.1748 |
| ETS           | 0.1340 | 0.0242 | 0.1806 |
| Prophet       | *(a regerar)* | *(a regerar)* | *(a regerar)* |

> Os números acima foram obtidos sobre os dados em cache (até 2026-01-16). A linha do Prophet
> será preenchida ao regenerar o relatório com `python main.py` (requer o pacote `prophet`
> instalado), garantindo que os quatro usem exatamente os mesmos cortes.

## Análise dos Resultados

A leitura crítica vem da comparação com o baseline:

*   O **ARIMA** tem o melhor desempenho, mas supera o random walk por uma margem **mínima**
    (MAPE 2,36% vs 2,42% — cerca de 2,5% de melhora relativa).
*   O **ETS** (tendência amortecida, sem sazonalidade) é **numericamente idêntico ao random
    walk**: a configuração ótima encontrada colapsa para "prever o último valor". Ou seja, o
    ETS não adiciona informação além do benchmark.
*   Nenhum modelo univariado supera o passeio aleatório de forma material a 60 dias.

Este resultado é esperado e bem documentado na literatura (o *puzzle* de Meese-Rogoff): taxas
de câmbio se comportam muito próximo de um passeio aleatório, e modelos univariados raramente
batem o ingênuo fora da amostra em horizontes de meses.

## Conclusão

Para a série BRL/USD, sob avaliação justa, o **ARIMA** é o melhor modelo — mas a vantagem sobre
o baseline ingênuo é pequena demais para ter relevância prática, e o ETS sequer se distingue do
random walk. A conclusão honesta é que, neste horizonte, **prever "amanhã ≈ hoje" é tão bom
quanto os modelos univariados testados**.

Para genuinamente melhorar a previsão seria necessário ir além de modelos univariados — por
exemplo, incorporar variáveis exógenas (diferencial de juros, commodities, índice do dólar) ou
focar em prever volatilidade em vez do nível. O desempenho também pode variar com o tempo e as
condições de mercado, então recomenda-se reavaliar periodicamente.
