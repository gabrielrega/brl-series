# TODO — brl-series

Próximos passos a partir dos resultados do `main.py` com a SELIC real
(série BCB 432; rodado em 2026-06-08 com `PERIOD=60` → **8 folds**).

## Conclusões registradas (contexto)

- **Com 8 folds e teste de Diebold-Mariano, nenhum modelo bate o baseline de
  forma significativa.** A "vantagem" do VAR sobre o random walk vira ruído:
  MAE 0.1190 vs 0.1197 (RW), mas DM = −0.166, **p = 0.868**. Meese-Rogoff
  praticamente intacto.
- **Tabela de nível (8 folds, horizonte 60):** RW 0.1197 · ARIMA 0.1242
  (DM +1.609, p 0.108) · ETS 0.1197 (DM −1.378, p 0.169) · Prophet 0.2000
  (DM +1.677, p 0.094) · VAR 0.1190 (DM −0.166, p 0.868). Nenhum p < 0.05.
- **Granger ainda significativo: SELIC → USD/BRL, p = 0.008.** Há informação
  preditiva *em amostra*, mas ela não se converte em ganho de previsão
  out-of-sample estatisticamente detectável (consistente com UIP fraca).
- **ETS ≈ RW** (tendência amortecida sem sazonalidade colapsa no random walk).
- **GARCH(1,1):** agora numericamente *abaixo* da vol constante nos dados novos
  (MAE 2.02 vs 2.31), mas a diferença **não é significativa** (DM −0.526,
  p = 0.615). Persistência 0.9895, vol anualizada prevista p/ 1 ano: ~12,06%.

## A fazer

- [x] **Mais folds na CV** (`PERIOD` 120 → 60): agora 8 folds, janelas de teste
      contíguas. A vantagem do VAR sobre o RW **não** persistiu.
- [x] **Testar significância da diferença** (Diebold-Mariano com correção HLN,
      `evaluation.diebold_mariano`): coluna `DM vs RW`/`DM vs CV` no `main.py`.
- [ ] **Diferencial de juros Brasil–EUA explícito** no VAR, em vez da SELIC pura
      (UIP é sobre o *diferencial*): baixar a Fed Funds / Treasury e usar `selic - i_us`.
- [ ] **Horizontes alternativos para o GARCH**: avaliar vol em horizontes curtos
      (5/21 dias) onde o clustering tende a ajudar mais que a 60 dias.
- [ ] **Decidir na revisão do PR #2** como casar a linha modular com o `master`
      antigo (versões standalone de GARCH/VAR via CSV) — ver histórico divergente.
