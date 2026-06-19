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
- [x] **Diferencial de juros Brasil–EUA explícito** no VAR, em vez da SELIC pura.
      `main.py` baixa a Fed Funds (FRED `DFF`, taxa de política overnight dos EUA,
      par natural da SELIC) e o VAR passa a usar `selic - fed_funds` como variável
      de juros; cai de volta à SELIC pura se o FRED estiver indisponível. O número
      out-of-sample só sai ao rodar `main.py` com rede (FRED).
- [x] **Horizontes alternativos para o GARCH** (Fase 4b, `VOL_HORIZONS = (5, 21,
      60)`, janelas contíguas → mais folds nos curtos). **Achado:** o clustering
      *paga* no curto prazo — em 5 dias (100 folds) o GARCH **bate** a vol constante
      de forma significativa (MAE 3.74 vs 4.42, DM −3.47, **p = 0.001**); em 21 dias
      vira ruído (DM −1.11, p 0.28) e em 60 some (DM −0.53, p 0.62). É o primeiro
      modelo do projeto a vencer seu baseline com significância — e **não** fere
      Meese-Rogoff, que é sobre o *nível*, não sobre a volatilidade.

## Concluído fora desta lista

- **Casar a linha modular com o `master` antigo:** resolvido — o PR #2 já foi
  mergeado (`ba2e33d`); a linha standalone com CSVs na raiz foi aposentada e a
  modular é a única. Item removido por estar obsoleto.

## A fazer (novos)

- [ ] **Diferencial com Treasury de prazo casado**: a SELIC é overnight, mas a
      expectativa cambial olha prazos maiores — testar `selic - DGS1`/`DGS2`
      (Treasury 1–2 anos) além do Fed Funds e ver se o Granger/IRF muda.
- [ ] **Confirmar o ganho do GARCH a 5 dias com a Fed Funds no VAR à parte**: o
      resultado de 5 dias é da volatilidade (univariado); não toca o nível.
