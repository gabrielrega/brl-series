# TODO — brl-series

Próximos passos a partir dos resultados do `main.py` com a SELIC real
(série BCB 432, jun/2021–jun/2026; rodado em 2026-06-01).

## Conclusões registradas (contexto)

- **VAR (câmbio + SELIC) é o único que bate o random walk** na CV de 60 dias:
  MAE 0.1725 vs 0.1753 (RW/ETS), 0.1799 (ARIMA), 0.1975 (Prophet).
- **Granger significativo: SELIC → USD/BRL, p = 0.008** com a SELIC real
  (no smoke com SELIC sintética dava não-significativo). Consistente com a
  Paridade de Juros — o diferencial de juros carrega informação preditiva.
- **Ressalva:** a vantagem do VAR é marginal (~1,6% de MAE) e só há **4 folds** —
  não é evidência forte; quebra apenas *parcial* do Meese-Rogoff.
- **ETS empata exato com o RW** (tendência amortecida sem sazonalidade ≈ RW).
- **GARCH(1,1) não bate a baseline de vol constante** em 60 dias (MAE 2.41 vs 2.32;
  só ganha no MAPE). Persistência 0.9892, ARCH-LM p<0.0001 (clustering confirmado).
  Vol anualizada prevista p/ 1 ano: ~11,87%.

## A fazer

- [ ] **Mais folds na CV** para robustez: reduzir `PERIOD` (ex.: 120 → 60/40) e
      reavaliar se a vantagem do VAR sobre o RW persiste. Hoje são só 4 folds.
- [ ] **Testar significância da diferença** VAR vs RW (ex.: Diebold-Mariano sobre
      os erros por fold), em vez de comparar só o MAE pontual.
- [ ] **Diferencial de juros Brasil–EUA explícito** no VAR, em vez da SELIC pura
      (UIP é sobre o *diferencial*): baixar a Fed Funds / Treasury e usar `selic - i_us`.
- [ ] **Horizontes alternativos para o GARCH**: avaliar vol em horizontes curtos
      (5/21 dias) onde o clustering tende a ajudar mais que a 60 dias.
- [ ] **Decidir na revisão do PR #2** como casar a linha modular com o `master`
      antigo (versões standalone de GARCH/VAR via CSV) — ver histórico divergente.
