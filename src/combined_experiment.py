"""
Combined Experiment — Recência + Performance (Ultimate Fighter Rank)

Item 9.5 do roadmap (docs/development_guideline.md, seção 53):
antes de promover qualquer extensão do Elo v0.1 para produção, é
preciso (a) validar a estabilidade fold a fold dos dois resultados
positivos encontrados até aqui — recência (rating_methodology.md,
seção 39) e performance/M4 (seção 42) — e (b) decidir se elas devem
ser combinadas ou tratadas como alternativas, já que foram avaliadas
de forma independente.

Este script reaproveita quase toda a lógica de
`src/performance_experiment.py` (parsing, agregação de performance,
pipeline logístico, métricas, folds walk-forward) e substitui apenas
o cálculo interno do Elo por uma versão com decaimento por recência
(mesma fórmula de `src/recency_sweep.py`, seção 39.1):

    R_efetivo = R_inicial + (R_histórico - R_inicial) * 2^(-Δt / H)

O decaimento é aplicado ao rating de cada lutador imediatamente
antes de calcular `elo_probability`/`elo_difference` da luta atual,
nunca reescrevendo o histórico já registrado (mesmo princípio da
seção 2.1: nenhuma informação futura contamina o passado).

Modelos comparados, todos com o mesmo harness walk-forward de
`performance_experiment.FOLDS`:

    M0            — Elo v0.1 puro (sem recência, sem performance)
    Recency_only  — Elo com decaimento por recência, sem performance
    M4            — Elo puro + todas as features de performance
    M4_Recency    — Elo com recência + todas as features de performance

Resultados:
    data/results/combined_temporal_results.csv
    data/results/combined_predictions.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.performance_experiment import (
    ALL_PERFORMANCE_FEATURES,
    BASE_K,
    BOOSTED_K,
    ELO_SCALE,
    FOLDS,
    INITIAL_RATING,
    PROVISIONAL_FIGHTS,
    aggregate_fight_performance,
    calculate_metrics,
    create_logistic_pipeline,
    evaluate_elo_baseline,
    get_performance_features,
    load_data,
    performance_from_row,
    prepare_fights,
    prepare_round_stats,
    update_performance_state,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"

TEMPORAL_RESULTS_PATH = RESULTS_DIR / "combined_temporal_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "combined_predictions.csv"


# ============================================================
# Meias-vidas candidatas.
#
# A seção 39.3 do rating_methodology.md deixou 3-5 anos como
# faixa vencedora, sem eleger um único valor com evidência
# fold a fold. Este experimento testa as duas pontas mais
# fortes dessa faixa (4 e 5 anos) combinadas com performance,
# em vez de assumir uma delas.
# ============================================================

HALF_LIVES_YEARS = [4, 5]


# ============================================================
# ELO COM RECÊNCIA (mesma fórmula de recency_sweep.py, seção 39.1)
# ============================================================


class RecencyFighterState:
    __slots__ = ("rating", "fights", "last_fight_date")

    def __init__(self):
        self.rating = INITIAL_RATING
        self.fights = 0
        self.last_fight_date = None


def apply_recency(rating, last_fight_date, current_date, half_life_years):
    """Decai `rating` em direção a INITIAL_RATING conforme os dias
    de inatividade desde `last_fight_date`. Sem luta anterior, não
    há decaimento (fator = 1.0)."""

    if last_fight_date is None or pd.isna(last_fight_date):
        return rating

    elapsed_days = max(0, (current_date - last_fight_date).days)
    half_life_days = half_life_years * 365.25
    factor = 2 ** (-elapsed_days / half_life_days)

    return INITIAL_RATING + (rating - INITIAL_RATING) * factor


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def get_k_factor(fights):
    return BOOSTED_K if fights < PROVISIONAL_FIGHTS else BASE_K


def update_elo(state_a, state_b, effective_a, effective_b, score_a):
    """Atualiza o rating histórico (não-decaído) de cada lutador a
    partir dos ratings EFETIVOS (pós-recência) usados na previsão,
    seguindo a mesma regra de recency_sweep.py: o decaimento nunca é
    escrito de volta no histórico, só usado para calcular a
    atualização do próximo estado."""

    probability_a = expected_score(effective_a, effective_b)

    k_a = get_k_factor(state_a.fights)
    k_b = get_k_factor(state_b.fights)

    state_a.rating = effective_a + k_a * (score_a - probability_a)
    state_b.rating = effective_b + k_b * ((1.0 - score_a) - (1.0 - probability_a))

    state_a.fights += 1
    state_b.fights += 1


# ============================================================
# BUILD HISTORICAL DATASET (Elo com recência + performance)
# ============================================================


def build_historical_dataset_with_recency(
    fights: pd.DataFrame,
    fight_performance: pd.DataFrame,
    half_life_years,
) -> pd.DataFrame:
    """
    Idêntico a performance_experiment.build_historical_dataset, exceto
    que o Elo usado para elo_probability/elo_difference é o rating
    EFETIVO pós-decaimento por recência (half_life_years=None reproduz
    exatamente o Elo v0.1 sem recência).
    """

    performance_lookup = {
        (row["fight_id"], row["fighter_id"]): row
        for _, row in fight_performance.iterrows()
    }

    elo_states: Dict[str, RecencyFighterState] = {}
    performance_states: Dict[str, Dict[str, float]] = {}

    rows: List[dict] = []

    for _, fight in fights.iterrows():
        fight_id = fight["fight_id"]
        current_date = fight["event_date"]

        fighter_a = fight["fighter_1_id"]
        fighter_b = fight["fighter_2_id"]

        result_a = str(fight["fighter_1_result"]).strip().upper()
        result_b = str(fight["fighter_2_result"]).strip().upper()

        state_a = elo_states.setdefault(fighter_a, RecencyFighterState())
        state_b = elo_states.setdefault(fighter_b, RecencyFighterState())

        perf_state_a = performance_states.setdefault(fighter_a, {})
        perf_state_b = performance_states.setdefault(fighter_b, {})

        perf_a = get_performance_features(perf_state_a)
        perf_b = get_performance_features(perf_state_b)

        if half_life_years is None:
            effective_a = state_a.rating
            effective_b = state_b.rating
        else:
            effective_a = apply_recency(
                state_a.rating, state_a.last_fight_date, current_date, half_life_years
            )
            effective_b = apply_recency(
                state_b.rating, state_b.last_fight_date, current_date, half_life_years
            )

        elo_probability_a = expected_score(effective_a, effective_b)

        row = {
            "fight_id": fight_id,
            "event_id": fight["event_id"],
            "event_date": current_date,
            "fighter_1_id": fighter_a,
            "fighter_2_id": fighter_b,
            "fighter_1_name": fight["fighter_1_name"],
            "fighter_2_name": fight["fighter_2_name"],
            "fighter_1_result": result_a,
            "fighter_2_result": result_b,
            "weight_class": fight["weight_class"],
            "elo_a": effective_a,
            "elo_b": effective_b,
            "elo_difference": effective_a - effective_b,
            "elo_probability": elo_probability_a,
        }

        for feature in ALL_PERFORMANCE_FEATURES:
            value_a = perf_a[feature]
            value_b = perf_b[feature]

            row[f"{feature}_a"] = value_a
            row[f"{feature}_b"] = value_b

            if pd.notna(value_a) and pd.notna(value_b):
                row[f"{feature}_difference"] = value_a - value_b
            else:
                row[f"{feature}_difference"] = np.nan

        if result_a == "W" and result_b == "L":
            row["target"] = 1
        elif result_a == "L" and result_b == "W":
            row["target"] = 0
        else:
            row["target"] = np.nan

        rows.append(row)

        # ====================================================
        # UPDATE ONLY AFTER THE FIGHT
        # ====================================================

        performance_row_a = performance_lookup.get((fight_id, fighter_a))
        performance_row_b = performance_lookup.get((fight_id, fighter_b))

        performance_a = (
            performance_from_row(performance_row_a)
            if performance_row_a is not None
            else None
        )
        performance_b = (
            performance_from_row(performance_row_b)
            if performance_row_b is not None
            else None
        )

        update_performance_state(perf_state_a, performance_a)
        update_performance_state(perf_state_b, performance_b)

        # NC e empate não alteram Elo (mesma regra do Elo v0.1 e do
        # performance_experiment.py original).
        if result_a == "W" and result_b == "L":
            update_elo(state_a, state_b, effective_a, effective_b, score_a=1.0)
        elif result_a == "L" and result_b == "W":
            update_elo(state_a, state_b, effective_a, effective_b, score_a=0.0)

        state_a.last_fight_date = current_date
        state_b.last_fight_date = current_date

    return pd.DataFrame(rows)


# ============================================================
# MODELOS COMPARADOS
# ============================================================

M4_FEATURES = [
    "elo_difference",
    *[f"{feature}_difference" for feature in ALL_PERFORMANCE_FEATURES],
]


def run_temporal_experiment_for_dataset(
    historical: pd.DataFrame,
    model_name: str,
) -> Tuple[List[dict], List[dict]]:
    """Roda o walk-forward de FOLDS para um único dataset (já construído
    com um half-life fixo) e um único nome de modelo lógico. Reaproveita
    a mesma estrutura de fold de performance_experiment.run_temporal_experiment,
    mas simplificada para um modelo por chamada."""

    temporal_results = []
    prediction_rows = []

    for fold_name, test_start, test_end in FOLDS:
        start = pd.Timestamp(test_start)
        end = pd.Timestamp(test_end)

        train = historical[
            (historical["event_date"] < start) & historical["target"].notna()
        ].copy()

        test = historical[
            (historical["event_date"] >= start)
            & (historical["event_date"] < end)
            & historical["target"].notna()
        ].copy()

        if test.empty:
            continue

        if model_name == "M0" or model_name.startswith("Recency_only"):
            metrics = evaluate_elo_baseline(test)

            temporal_results.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics,
                }
            )

            for _, row in test.iterrows():
                prediction_rows.append(
                    {
                        "fold": fold_name,
                        "model": model_name,
                        "fight_id": row["fight_id"],
                        "target": int(row["target"]),
                        "probability_fighter_1": row["elo_probability"],
                    }
                )

            continue

        # M4 / M4_Recency: regressão logística com elo_difference +
        # todas as features de performance, ajustada só com dados de
        # treino do fold (mesma disciplina da seção 41.1/42.1).
        X_train = train[M4_FEATURES]
        y_train = train["target"].astype(int)

        X_test = test[M4_FEATURES]
        y_test = test["target"].astype(int)

        pipeline = create_logistic_pipeline()
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test.to_numpy(), probabilities)

        temporal_results.append(
            {
                "fold": fold_name,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
            }
        )

        for fight_id, target, probability in zip(
            test["fight_id"], y_test, probabilities
        ):
            prediction_rows.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "fight_id": fight_id,
                    "target": int(target),
                    "probability_fighter_1": probability,
                }
            )

    return temporal_results, prediction_rows


def calculate_pooled_results(temporal_results: List[dict]) -> List[dict]:
    df = pd.DataFrame(temporal_results)

    pooled_rows = []

    for model_name, group in df.groupby("model"):
        n_total = int(group["n_test"].sum())

        pooled_rows.append(
            {
                "fold": "pooled test",
                "model": model_name,
                "n_train": None,
                "n_test": n_total,
                "log_loss": float(np.average(group["log_loss"], weights=group["n_test"])),
                "brier": float(np.average(group["brier"], weights=group["n_test"])),
                "auc": float(np.average(group["auc"], weights=group["n_test"])),
                "accuracy": float(np.average(group["accuracy"], weights=group["n_test"])),
            }
        )

    return pooled_rows


def main() -> None:
    print("=== COMBINED EXPERIMENT: RECÊNCIA + PERFORMANCE ===")

    fights_raw, events, round_stats = load_data()

    fights = prepare_fights(fights_raw, events)
    round_stats_prepared = prepare_round_stats(round_stats)
    fight_performance = aggregate_fight_performance(round_stats_prepared)

    all_temporal_results: List[dict] = []
    all_predictions: List[dict] = []

    # ---- M0: Elo v0.1 puro (sem recência, sem performance) ----
    print("\n--- M0: Elo v0.1 puro ---")
    dataset_no_recency = build_historical_dataset_with_recency(
        fights, fight_performance, half_life_years=None
    )

    temporal, predictions = run_temporal_experiment_for_dataset(
        dataset_no_recency, model_name="M0"
    )
    all_temporal_results.extend(temporal)
    all_predictions.extend(predictions)

    # ---- M4: Elo puro + performance (sem recência) ----
    print("--- M4: Elo + performance (sem recência) ---")
    temporal, predictions = run_temporal_experiment_for_dataset(
        dataset_no_recency, model_name="M4"
    )
    all_temporal_results.extend(temporal)
    all_predictions.extend(predictions)

    # ---- Recency_only e M4_Recency, para cada meia-vida candidata ----
    for half_life in HALF_LIVES_YEARS:
        label_recency = f"Recency_only_{half_life}y"
        label_combined = f"M4_Recency_{half_life}y"

        print(f"--- {label_recency} / {label_combined} ---")

        dataset_recency = build_historical_dataset_with_recency(
            fights, fight_performance, half_life_years=half_life
        )

        temporal, predictions = run_temporal_experiment_for_dataset(
            dataset_recency, model_name=label_recency
        )
        all_temporal_results.extend(temporal)
        all_predictions.extend(predictions)

        temporal, predictions = run_temporal_experiment_for_dataset(
            dataset_recency, model_name=label_combined
        )
        all_temporal_results.extend(temporal)
        all_predictions.extend(predictions)

    pooled = calculate_pooled_results(all_temporal_results)
    all_temporal_results.extend(pooled)

    temporal_df = pd.DataFrame(all_temporal_results)
    predictions_df = pd.DataFrame(all_predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporal_df.to_csv(TEMPORAL_RESULTS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\n=== RESULTADO POOLED ===")
    print(
        temporal_df[temporal_df["fold"] == "pooled test"]
        .sort_values("log_loss")
        .to_string(index=False)
    )

    print(f"\nSalvo em: {TEMPORAL_RESULTS_PATH}")
    print(f"Salvo em: {PREDICTIONS_PATH}")

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()
