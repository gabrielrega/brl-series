# Relatório de Análise de Séries Temporais: Previsão da Taxa de Câmbio BRL/USD

## Introdução

Este relatório apresenta os resultados de uma análise de séries temporais da taxa de câmbio BRL/USD. O objetivo do projeto foi comparar o desempenho de três modelos de previsão diferentes: ARIMA, Prophet e ETS.

## Modelos Utilizados

Foram utilizados três modelos de previsão de séries temporais univariadas:

*   **ARIMA (Autoregressive Integrated Moving Average):** Um modelo estatístico clássico que utiliza os valores passados da série temporal para prever valores futuros.
*   **Prophet:** Uma biblioteca de previsão de séries temporais de código aberto desenvolvida pelo Facebook. É projetado para ser fácil de usar e robusto para uma ampla variedade de séries temporais.
*   **ETS (Exponential Smoothing):** Um modelo que atribui pesos exponencialmente decrescentes às observações passadas. É eficaz na modelagem de tendências e sazonalidades.

## Metodologia de Avaliação

Para que a comparação entre os modelos seja justa, **os três são avaliados sob o mesmo
protocolo**: validação cruzada *rolling-origin* (origem móvel), implementada em
`evaluation.py`. A cada corte, o modelo é treinado apenas com os dados até aquele ponto e
prevê o mesmo horizonte de 60 dias úteis, nas mesmas datas reais. As métricas (MAE, MAPE,
RMSE) são calculadas sobre o conjunto agrupado de todas as previsões.

Parâmetros: janela inicial de treino de 750 observações (~3 anos), passo de 120 observações
(~6 meses) entre cortes e horizonte de 60 observações (~3 meses) — resultando em 4 folds.

> **Nota metodológica:** versões anteriores deste relatório comparavam métricas calculadas de
> formas diferentes para cada modelo (ARIMA por *walk-forward* de 1 passo, ETS por previsão
> estática com erro de alinhamento, Prophet por *cross-validation*). Isso favorecia
> artificialmente o ARIMA (MAPE de ~0,4%, irrealista para câmbio). Sob o protocolo unificado,
> os erros refletem a real dificuldade de prever a série a 60 dias.

## Resultados

As métricas utilizadas são o Erro Absoluto Médio (MAE), o Erro Percentual Absoluto Médio
(MAPE) e a Raiz do Erro Quadrático Médio (RMSE), todas em validação cruzada de 4 folds.

| Modelo  | MAE    | MAPE   | RMSE   |
|---------|--------|--------|--------|
| ARIMA   | 0.1305 | 0.0236 | 0.1748 |
| ETS     | 0.1476 | 0.0268 | 0.1966 |
| Prophet | *(a regerar)* | *(a regerar)* | *(a regerar)* |

> Os números de ARIMA e ETS acima foram obtidos sobre os dados em cache (até 2026-01-16). A
> linha do Prophet será preenchida ao regenerar o relatório com `python main.py` (requer o
> pacote `prophet` instalado), garantindo que os três usem exatamente os mesmos cortes.

## Análise dos Resultados

Sob o protocolo unificado, o modelo **ARIMA** ainda apresenta o melhor desempenho, com os
menores MAE, MAPE e RMSE, seguido de perto pelo **ETS**. A diferença entre os dois é pequena,
o que é esperado para uma série de câmbio próxima de um passeio aleatório. A linha do Prophet
será confirmada na regeneração.

## Conclusão

Para a série temporal da taxa de câmbio BRL/USD analisada, o modelo ARIMA apresentou o melhor
desempenho sob avaliação justa, com o ETS como alternativa competitiva. Vale ressaltar que, a
um horizonte de 60 dias, os erros (~2-3% de MAPE) mostram que nenhum dos modelos univariados
prevê o câmbio com alta precisão — coerente com a natureza da série.

É importante ressaltar que o desempenho dos modelos pode variar com o tempo e com as condições
do mercado. Portanto, é recomendável reavaliar e recalibrar os modelos periodicamente.
