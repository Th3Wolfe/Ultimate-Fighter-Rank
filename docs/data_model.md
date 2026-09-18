# Modelo de dados

## fighters

- fighter_id
- first_name
- last_name
- nickname
- height_cm
- weight_lbs
- reach_cm
- stance
- dob

## events

- event_id
- name
- date
- location

## fights

- fight_id
- event_id
- fighter_a_id
- fighter_b_id
- result_a
- result_b
- weight_class
- is_title_bout
- is_interim_title
- is_tournament
- method
- ending_round
- ending_time
- time_format

## round_stats

- fight_id
- round
- fighter_id
- kd
- sig_str_landed
- sig_str_attempted
- sig_str_pct
- total_str_landed
- total_str_attempted
- td_landed
- td_attempted
- td_pct
- sub_att
- reversals
- control_seconds
- head_landed
- head_attempted
- body_landed
- body_attempted
- leg_landed
- leg_attempted
- distance_landed
- distance_attempted
- clinch_landed
- clinch_attempted
- ground_landed
- ground_attempted

---

As tabelas acima (`fighters`, `events`, `fights`, `round_stats`) vivem em `data/processed/`. As tabelas abaixo são derivadas delas e vivem em `data/features/`, `data/evaluation/` e `data/results/`, conforme a camada da arquitetura (seção 4 do `development_guideline.md`).

## data/features/round_stats_numeric.csv

Estatísticas por round (`round_stats`) já convertidas para tipos numéricos (percentuais, tempos em segundos), uma linha por `(fight_id, fighter_id, round)`.

- fight_id, fighter_id, round, fighter_name
- significant_strikes_landed/attempted, total_strikes_landed/attempted
- takedowns_landed/attempted, head/body/leg_strikes_landed/attempted
- distance/clinch/ground_strikes_landed/attempted
- significant_strikes_pct, takedown_pct
- knockdowns, submission_attempts, reversals, control_time_seconds

## data/features/fight_fighter_features.csv

`round_stats_numeric` agregado por round para o total da luta, uma linha por `(fight_id, fighter_id)`. Mesmas colunas de estatística que `round_stats_numeric`, sem a dimensão `round`, mais `rounds_with_stats`.

## data/features/fight_features.csv

Uma linha por luta (`fight_id`), com as estatísticas de `fight_fighter_features` já pivotadas lado a lado para `fighter_1` e `fighter_2` (sufixos `_fighter_1` / `_fighter_2`), mais as colunas `*_diff` (fighter_1 − fighter_2) para cada estatística numérica. Inclui também os metadados da luta (evento, bout, weight_class, method, ending_round/time, referee, details).

## data/features/fighter_fight_features.csv

Mesma informação de `fight_features`, mas na representação "fighter-fight" (uma linha por lutador por luta, não uma linha por luta) — cada luta aparece duas vezes, uma por perspectiva. Usada como base para o histórico temporal.

## data/features/fighter_history.csv

Núcleo do histórico temporal (seção 20/21 do `development_guideline.md`). Uma linha por `(fighter_id, fight_id)`, ordenada cronologicamente, contendo:

- os dados da própria luta (resultado, método, estatísticas daquela luta, `date`);
- **colunas `_before`**, calculadas apenas com lutas anteriores àquela data (nunca incluindo a luta atual): `career_fights_before`, `wins_before`, `losses_before`, `draws_before`, `nc_before`, `win_rate_before`, e somas/médias `_before` de cada estatística numérica (`sum_*_before`, `avg_*_before`).

Esta é a tabela que impede o data leakage descrito na seção 2.2 do `rating_methodology.md`: qualquer feature usada para prever uma luta vem de uma coluna `_before`.

## data/features/fighter_ratings.csv

Saída de produção do Elo v0.1 (`src/build_fighter_ratings.py`), uma linha por `(fight_id, fighter_id)`:

- fight_id, date, fighter_id, opponent_id, division
- result
- career_fights_before, is_debut, is_division_change
- k_factor (K efetivo usado nesta luta — base ou boosted)
- rating_updated (`False` para No Contest, seção 37.1)
- expected_score (P(fighter_id vence), calculado a partir de `rating_before`)
- rating_before, rating_after

## data/features/param_sweep/ e data/features/recency/

Saídas **experimentais** de `src/param_sweep.py` e `src/recency_sweep.py`, no mesmo formato de `fighter_ratings.csv`, uma subpasta/arquivo por configuração testada (ver `rating_methodology.md`, seções 37.5, 37.6 e 39). Nunca sobrescrevem o `fighter_ratings.csv` de produção.

## data/evaluation/evaluation_dataset.csv

Uma linha por luta avaliável (exclui Draw e NC, seção 38.3 do `rating_methodology.md`): `fight_id, date, fighter_1_id, fighter_2_id, y, p_win_rate, p_elo`, onde `y` é o rótulo binário (fighter_1 venceu) e `p_*` são as probabilidades de cada modelo antes da luta.

## data/evaluation/metrics_summary.csv

Uma linha por `(model, fold)`: `model, fold, n, log_loss, brier_score, accuracy, roc_auc`. Cobre o split simples (2019+) e cada janela do walk-forward, para Win Rate e Elo v0.1 (`rating_methodology.md`, seção 38.4-38.5).

## data/evaluation/calibration_tables.csv

Uma linha por `(model, bin de probabilidade)`: `model, bin, n, mean_predicted, observed_win_rate, gap` — usada para diagnosticar calibração (seção 38.5).

## data/evaluation/param_sweep/summary.csv

Mesmo formato de `metrics_summary.csv`, mais colunas de configuração (`base_k, boosted_k, scale, provisional_fights`) e o modelo comparado (`win_rate` ou `elo`) — uma linha por `(config, model, fold)`.

## data/evaluation/recency/summary.csv e fold_metrics.csv

`summary.csv`: métricas pooled por configuração de meia-vida (`config, half_life_years, log_loss, brier_score, roc_auc, accuracy`). `fold_metrics.csv`: o mesmo detalhado por fold, incluindo o split simples e o pooled (`config, model, fold, n, log_loss, brier_score, accuracy, roc_auc, half_life_years`).

## data/results/glicko2_temporal_results.csv e glicko2_predictions.csv

`glicko2_temporal_results.csv`: uma linha por fold do walk-forward (mais uma linha `POOLED`), com `fold_start, fold_end, n_fights, log_loss, brier, auc, accuracy`. `glicko2_predictions.csv`: previsão luta a luta.

## data/results/opponent_quality_temporal_results.csv e opponent_quality_predictions.csv

Uma linha por fold (mais `POOLED`), com métricas lado a lado para Elo isolado (`elo_*`) e Elo+SoS (`sos_*`): `fold_start, fold_end, n_fights, elo_log_loss, elo_brier, elo_auc, elo_accuracy, sos_log_loss, sos_brier, sos_auc, sos_accuracy`.

## data/results/performance_temporal_results.csv, performance_predictions.csv e performance_coefficients.csv

`performance_temporal_results.csv`: uma linha por `(fold, model)` para os modelos M0-M5 (`rating_methodology.md`, seção 42), com `n_train, n_test, log_loss, brier, auc, accuracy`. `performance_predictions.csv`: previsão luta a luta por modelo. `performance_coefficients.csv`: coeficientes da regressão logística de cada modelo, por fold e por feature (`model, fold, feature, coefficient, abs_coefficient`).
