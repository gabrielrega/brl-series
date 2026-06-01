# Relatório de Análise de Séries Temporais: Previsão da Taxa de Câmbio BRL/USD

## Introdução

Este relatório apresenta os resultados de uma análise de séries temporais da taxa de câmbio BRL/USD. O objetivo do projeto foi comparar o desempenho de três modelos de previsão diferentes: ARIMA, Prophet e ETS.

## Modelos Utilizados

Foram utilizados três modelos de previsão de séries temporais univariadas:

*   **ARIMA (Autoregressive Integrated Moving Average):** Um modelo estatístico clássico que utiliza os valores passados da série temporal para prever valores futuros.
*   **Prophet:** Uma biblioteca de previsão de séries temporais de código aberto desenvolvida pelo Facebook. É projetado para ser fácil de usar e robusto para uma ampla variedade de séries temporais.
*   **ETS (Exponential Smoothing):** Um modelo que atribui pesos exponencialmente decrescentes às observações passadas. É eficaz na modelagem de tendências e sazonalidades.

## Resultados

A tabela a seguir apresenta as métricas de avaliação de cada modelo. As métricas utilizadas foram o Erro Absoluto Médio (MAE), o Erro Percentual Absoluto Médio (MAPE) e a Raiz do Erro Quadrático Médio (RMSE).

| Modelo  | MAE    | MAPE   | RMSE   |
|---------|--------|--------|--------|
| ARIMA   | 0.0220 | 0.0041 | 0.0294 |
| Prophet | 0.1894 | 0.0333 | 0.2204 |
| ETS     | 0.0529 | 0.0097 | 0.0733 |

## Análise dos Resultados

Com base nas métricas de avaliação, o modelo **ARIMA** apresentou o melhor desempenho, com os menores valores de MAE, MAPE e RMSE. Isso indica que o modelo ARIMA foi o mais preciso na previsão da taxa de câmbio BRL/USD no período de teste.

O modelo **ETS** ficou em segundo lugar, com um desempenho razoável, mas inferior ao do ARIMA.

O modelo **Prophet**, por outro lado, teve o pior desempenho entre os três, com erros significativamente maiores.

## Conclusão

Para a série temporal da taxa de câmbio BRL/USD analisada, o modelo ARIMA foi o que apresentou o melhor desempenho. Recomenda-se a utilização do modelo ARIMA para futuras previsões desta série.

É importante ressaltar que o desempenho dos modelos pode variar com o tempo e com as condições do mercado. Portanto, é recomendável reavaliar e recalibrar os modelos periodicamente.
