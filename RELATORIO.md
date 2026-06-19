# Relatório de Análise de Séries Temporais: Previsão da Taxa de Câmbio BRL/USD

*Atualizado em 2026-06-19 — dados do BCB de 2021-06-21 a 2026-06-19 (1256 observações).*

## Introdução

Este relatório compara o desempenho de vários modelos de previsão para a taxa de
câmbio BRL/USD, **todos avaliados sob o mesmo protocolo** para que a comparação
seja honesta. A análise se divide em duas frentes com alvos distintos:

* **Nível** da taxa de câmbio (quanto vale o dólar): ARIMA, ETS, Prophet e um VAR
  bivariado com juros — todos medidos contra um baseline de *random walk*.
* **Volatilidade** dos retornos (o quanto a taxa oscila): GARCH(1,1), medido
  contra um baseline de volatilidade constante.

O resultado central é, em grande parte, **negativo e intencional**: nenhum modelo
de *nível* supera o passeio aleatório de forma significativa (o *puzzle* de
Meese-Rogoff). A exceção positiva está na **volatilidade em horizontes curtos**,
onde o GARCH genuinamente bate seu baseline.

## Modelos Utilizados

Modelos de **nível**:

* **ARIMA:** modelo estatístico clássico; ordem `(p,1,q)` escolhida por grid
  search em AIC (`p,q ∈ 0..2`). Selecionou ARIMA(1,1,2).
* **ETS (Exponential Smoothing):** tendência aditiva **amortecida, sem
  sazonalidade** — a CV mostrou qualquer componente sazonal (5, 12, 21 ou 252
  dias) inerte para esta série, enquanto o amortecimento da tendência ajudou a 60
  dias. Na prática colapsa para "prever o último valor".
* **Prophet:** biblioteca do Facebook, com sazonalidade semanal/anual e feriados
  brasileiros.
* **VAR bivariado (USD/BRL + juros):** motivado pela Paridade Descoberta de Juros
  (UIP). A UIP é sobre o **diferencial** de juros Brasil–EUA, então o modelo usa
  `SELIC − Fed Funds` quando a taxa americana (FRED, série `DFF`) está disponível,
  caindo de volta à **SELIC pura** caso contrário. *Neste run a Fed Funds não foi
  baixada, então os números abaixo são do VAR com a SELIC pura.* Inclui teste de
  causalidade de Granger e funções de resposta a impulso.

Modelo de **volatilidade**:

* **GARCH(1,1) com inovação t de Student** (fallback Normal): captura o
  *clustering* de volatilidade típico de câmbio (Engle 1982, Bollerslev 1986).
  Prevê a variância condicional, não o nível, e por isso é avaliado contra a
  volatilidade realizada em uma CV própria.

## Metodologia de Avaliação

Todos os modelos de nível são avaliados sob o mesmo protocolo de **validação
cruzada *rolling-origin*** (`evaluation.py`). A cada corte, o modelo é treinado só
com os dados até ali e prevê o mesmo horizonte, nas mesmas datas reais; as métricas
(MAE, MAPE, RMSE) são agrupadas sobre todas as previsões. Isso é o que torna os
números diretamente comparáveis.

Parâmetros: janela inicial de **750** observações (~3 anos), passo de **60**
observações entre cortes e horizonte de **60** dias úteis (~3 meses) → **8 folds**
com janelas de teste contíguas.

Como um MAE menor pode ser apenas ruído com poucos folds, cada modelo é comparado
ao baseline pelo **teste de Diebold-Mariano** (correção de amostra pequena de
Harvey-Leybourne-Newbold; variância de longo prazo Newey-West truncada em
`horizonte−1`). Um **DM negativo** indica que o modelo bate o baseline;
significância exige **p < 0,05**.

O baseline de nível é o **random walk** ("amanhã ≈ hoje"). A volatilidade tem seu
próprio protocolo: o GARCH é comparado à **volatilidade constante** (vol realizada
da janela de treino), nunca misturado à tabela de nível.

> **Nota metodológica:** versões antigas deste relatório comparavam métricas
> calculadas de formas diferentes por modelo (ARIMA por *walk-forward* de 1 passo,
> ETS com erro de alinhamento, Prophet por outra CV), o que favorecia
> artificialmente o ARIMA (MAPE ~0,4%, irrealista). Sob o protocolo unificado os
> erros refletem a real dificuldade de prever a série a 60 dias.

## Resultados — Nível

Validação cruzada de **8 folds**, horizonte de 60 dias úteis. `DM vs RW` é o teste
de Diebold-Mariano contra o random walk (negativo = melhor que o RW).

| Modelo        | MAE    | MAPE   | RMSE   | DM vs RW | p     |
|---------------|--------|--------|--------|----------|-------|
| **Naive RW**  | 0.1278 | 0.0231 | 0.1567 | —        | —     |
| ARIMA         | 0.1282 | 0.0231 | 0.1567 |  +0.107  | 0.915 |
| ETS           | 0.1278 | 0.0231 | 0.1567 |  −1.207  | 0.228 |
| Prophet       | 0.1929 | 0.0348 | 0.2370 |  +1.592  | 0.112 |
| VAR (SELIC)   | 0.1344 | 0.0242 | 0.1659 |  +1.146  | 0.252 |

**Nenhum modelo bate o random walk de forma significativa** (todos com p ≫ 0,05).
O ETS é numericamente idêntico ao RW; ARIMA empata; Prophet e VAR ficam atrás.

## Resultados — Volatilidade

GARCH(1,1) vs volatilidade constante, em pontos de volatilidade anualizada.

| Modelo      | MAE    | MAPE   | RMSE   | DM vs CV | p     |
|-------------|--------|--------|--------|----------|-------|
| Const Vol   | 2.4158 | 0.2611 | 2.9103 | —        | —     |
| GARCH (60d) | 1.9878 | 0.2001 | 2.3852 | −0.789   | 0.456 |

No horizonte de 60 dias o GARCH é numericamente melhor, mas **não
significativamente** — com apenas 8 folds o teste tem pouco poder. A história muda
ao olhar **horizontes mais curtos**, onde cada horizonte ladrilha as próprias
janelas de teste e rende muito mais folds:

| Horizonte | Folds | GARCH MAE | Const Vol MAE | DM vs CV | p         |
|-----------|-------|-----------|---------------|----------|-----------|
| **5 dias**  | 101 | 3.5446    | 4.4268        | **−5.044** | **0.000** |
| 21 dias   | 24    | 2.2572    | 2.7397        | −1.281   | 0.213     |
| 60 dias   | 8     | 1.9878    | 2.4158        | −0.789   | 0.456     |

**Em 5 dias o GARCH bate a volatilidade constante de forma fortemente
significativa** (DM −5,04, p < 0,001). O ganho some em 21 dias e 60 dias — o
*clustering* de volatilidade paga no curto prazo e se dilui à medida que o
horizonte cresce.

## Análise dos Resultados

* **Nível: Meese-Rogoff se confirma.** Câmbio diário se comporta muito próximo de
  um passeio aleatório; modelos univariados raramente batem o ingênuo fora da
  amostra em horizontes de meses. O ETS ótimo literalmente colapsa para "prever o
  último valor".
* **O VAR não ajuda fora da amostra — mesmo com sinal em amostra.** A SELIC
  **Granger-causa** o USD/BRL *dentro* da amostra (p < 0,05), mas essa informação
  **não se converte** em ganho de previsão detectável fora da amostra (UIP fraca):
  o VAR fica até pior que o RW (MAE 0.1344 vs 0.1278). Por construção, o modelo
  passa a usar o **diferencial SELIC − Fed Funds** quando a taxa americana está
  disponível — a forma teoricamente correta da UIP —, mas neste run rodou com a
  SELIC pura.
* **Volatilidade é onde há sinal real.** O GARCH é o **único modelo do projeto a
  vencer seu baseline com significância**, e faz isso justamente onde a teoria
  prevê: no curto prazo (5 dias), onde o *clustering* domina. Isso **não contradiz
  Meese-Rogoff** — prever *o quanto* a taxa oscila é um problema diferente de
  prever *para onde* ela vai. A volatilidade anualizada prevista para o próximo ano
  é de ~12,4%, com persistência (α+β) ≈ 0,99 (choques decaem lentamente, típico de
  câmbio).

> **Sensibilidade temporal:** o ranking de nível é instável entre janelas de dados
> (em recortes anteriores o ARIMA chegou a empatar ou superar o RW por margem
> mínima). Essa própria instabilidade reforça que as diferenças entre modelos de
> nível são ruído, não sinal — e que "não significativo" aqui significa "sem
> evidência", não "igual".

## Projeções para o Futuro (≈ 1 ano à frente)

Além do back-test, cada modelo gera uma previsão *forward* de 252 dias úteis
(~1 ano, até meados de junho/2027), salva em `assets/` (CSVs e gráficos). Partindo
do último valor observado — **USD/BRL = 5.144 em 2026-06-19** — os pontos a 1 ano
são:

| Modelo        | Projeção 1 ano | IC 95%          | Leitura                                  |
|---------------|----------------|-----------------|------------------------------------------|
| ETS           | 5.14           | —               | plano (tendência amortecida → RW)        |
| ARIMA         | 5.13           | 3.77 – 6.49     | plano, com incerteza enorme              |
| VAR (SELIC)   | 5.17           | —               | quase plano; extrapola a SELIC p/ ~16%   |
| Prophet       | 4.48           | 3.11 – 5.74     | aprecia o real — extrapolação arriscada  |

**Como ler estas projeções — com cautela.** Dado que *nenhum* modelo de nível bate
o random walk, a única projeção de nível realmente defensável é a do próprio
passeio aleatório: **"daqui a um ano ≈ hoje", ou seja ~5,14**. Não por acaso, ETS,
ARIMA e VAR convergem para perto disso — apenas reproduzem o último valor com ruído.
A divergência do **Prophet (4,48)** vem da sua extrapolação de tendência/
sazonalidade e **não deve ser tratada como sinal**: os intervalos de confiança a 1
ano são largos a ponto de englobar tanto forte apreciação quanto forte depreciação
(ARIMA: 3,77–6,49). Em outras palavras, a 1 ano a banda de incerteza é tão ampla
que o ponto central é quase irrelevante.

**A projeção forward que vale é a de volatilidade.** O GARCH projeta volatilidade
anualizada de **~12,4%** para o próximo ano (subindo de leve, de ~12,3%),
consistente com a alta persistência estimada (α+β ≈ 0,99). Diferente do nível, essa
é uma grandeza que o modelo demonstrou saber prever no curto prazo — e é a peça mais
acionável das projeções (dimensionar risco, bandas, *stops*), não a aposta
direcional no nível.

## Conclusão

Para o **nível** da BRL/USD, sob avaliação justa e contra um baseline, **o passeio
aleatório é tão bom ou melhor que qualquer um dos modelos testados** — incluindo o
VAR com juros. Nenhum modelo univariado de nível justifica sua complexidade neste
horizonte. Isso vale também para as **projeções forward**: a 1 ano, a melhor
estimativa de nível é "≈ hoje" (~5,14), com bandas de incerteza largas demais para
sustentar uma aposta direcional.

O ganho concreto aparece ao **mudar o alvo da previsão**: prever **volatilidade** em
vez de nível. O GARCH(1,1) bate a volatilidade constante de forma significativa em
horizontes curtos (5 dias), realizando exatamente a recomendação que versões
anteriores deste relatório deixavam como trabalho futuro.

Próximos passos naturais: (i) rodar o VAR com o diferencial `SELIC − Fed Funds`
completo (requer acesso ao FRED) e testar prazos de Treasury casados com a
expectativa cambial; (ii) explorar o ganho do GARCH em horizontes curtos com mais
profundidade. Recomenda-se reavaliar periodicamente — o desempenho varia com o
tempo e as condições de mercado.
