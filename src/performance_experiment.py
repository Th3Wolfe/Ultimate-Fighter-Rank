"""
Performance Experiment — Ultimate Fighter Rank

Testa se estatísticas históricas de performance acrescentam poder
preditivo ao Elo v0.1.

Modelos:
    M0 - Elo
    M1 - Elo + Striking
    M2 - Elo + Grappling
    M3 - Elo + Dominance
    M4 - Elo + All Performance
    M5 - Performance sem Elo

Princípios:
    - nenhuma informação futura é utilizada;
    - histórico de performance é atualizado somente após cada luta;
    - percentuais são recalculados a partir dos totais;
    - ausência de stats não é convertida para zero;
    - scaler/imputer/modelo são ajustados somente no treino de cada fold;
    - avaliação exclusivamente temporal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FIGHTS_PATH = ROOT / "data" / "processed" / "fights.csv"
EVENTS_PATH = ROOT / "data" / "processed" / "events.csv"
ROUND_STATS_PATH = ROOT / "data" / "processed" / "round_stats.csv"

RESULTS_DIR = ROOT / "data" / "results"

TEMPORAL_RESULTS_PATH = RESULTS_DIR / "performance_temporal_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "performance_predictions.csv"
COEFFICIENTS_PATH = RESULTS_DIR / "performance_coefficients.csv"


# ============================================================
# ELO V0.1
# ============================================================

INITIAL_RATING = 1500.0
BASE_K = 32.0
BOOSTED_K = 64.0
PROVISIONAL_FIGHTS = 3
ELO_SCALE = 400.0


# ============================================================
# TEMPORAL FOLDS
# ============================================================

FOLDS = [
    ("2011-2014", "2011-01-01", "2014-01-01"),
    ("2014-2017", "2014-01-01", "2017-01-01"),
    ("2017-2020", "2017-01-01", "2020-01-01"),
    ("2020-2023", "2020-01-01", "2023-01-01"),
    ("2023-2026", "2023-01-01", "2026-01-01"),
    ("2026-2029", "2026-01-01", "2029-01-01"),
]


# ============================================================
# PERFORMANCE GROUPS
# ============================================================

STRIKING_FEATURES = [
    "sig_str_landed_per_fight",
    "sig_str_attempted_per_fight",
    "sig_str_pct",
    "total_str_landed_per_fight",
    "total_str_attempted_per_fight",
]

GRAPPLING_FEATURES = [
    "takedowns_per_fight",
    "td_attempted_per_fight",
    "td_pct",
    "submission_attempts_per_fight",
    "reversals_per_fight",
]

DOMINANCE_FEATURES = [
    "knockdowns_per_fight",
    "control_time_per_fight",
]

ALL_PERFORMANCE_FEATURES = (
    STRIKING_FEATURES
    + GRAPPLING_FEATURES
    + DOMINANCE_FEATURES
)


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class FighterState:
    rating: float = INITIAL_RATING
    fights: int = 0


@dataclass
class FightPerformance:
    sig_str_landed: float
    sig_str_attempted: float
    total_str_landed: float
    total_str_attempted: float
    takedowns: float
    td_attempted: float
    submission_attempts: float
    reversals: float
    knockdowns: float
    control_time: float


# ============================================================
# PARSING HELPERS
# ============================================================


def parse_landed_attempted(value) -> Tuple[float, float]:
    """
    Converte strings como:

        '23 of 50' -> (23, 50)
        '0 of 0'   -> (0, 0)
        '---'      -> (nan, nan)
    """
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()

    if text in {"", "---", "-", "nan", "NaN"}:
        return np.nan, np.nan

    match = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s+of\s+(-?\d+(?:\.\d+)?)\s*$",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return np.nan, np.nan

    return float(match.group(1)), float(match.group(2))


def parse_percentage(value) -> float:
    """
    Converte:

        '46%' -> 0.46
        '0%'  -> 0.0
        '---' -> nan

    O percentual original não será usado para agregação.
    Esta função existe para manter o parser disponível.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in {"", "---", "-", "nan", "NaN"}:
        return np.nan

    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return np.nan

    try:
        value_float = float(text)
    except ValueError:
        return np.nan

    if value_float > 1:
        return value_float / 100.0

    return value_float


def parse_time_seconds(value) -> float:
    """
    Converte:

        '1:02' -> 62
        '0:00' -> 0
        '---'  -> nan
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text in {"", "---", "-", "nan", "NaN"}:
        return np.nan

    parts = text.split(":")

    if len(parts) != 2:
        return np.nan

    try:
        minutes = float(parts[0])
        seconds = float(parts[1])
    except ValueError:
        return np.nan

    return minutes * 60.0 + seconds


# ============================================================
# LOAD DATA
# ============================================================


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Carregando: {FIGHTS_PATH}")
    fights = pd.read_csv(FIGHTS_PATH)

    print(f"Carregando: {EVENTS_PATH}")
    events = pd.read_csv(EVENTS_PATH)

    print(f"Carregando: {ROUND_STATS_PATH}")
    round_stats = pd.read_csv(ROUND_STATS_PATH)

    return fights, events, round_stats


# ============================================================
# PREPARE FIGHTS
# ============================================================


def prepare_fights(
    fights: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adiciona event_date aos fights, corrige a orientação dos
    lutadores de forma determinística e ordena cronologicamente.

    A ordem dos lutadores é definida pelo fighter_id:
        fighter_1_id < fighter_2_id

    Essa correção é feita somente na cópia utilizada pelo
    experimento. O fights.csv original permanece inalterado.

    Ordem temporal:
        date
        fight_id
    """

    # --------------------------------------------------------
    # Trabalhamos sobre uma cópia para não modificar o
    # DataFrame original recebido pela função.
    # --------------------------------------------------------

    fights = fights.copy()
    events = events.copy()

    # --------------------------------------------------------
    # EVENT DATE
    # --------------------------------------------------------

    possible_date_columns = [
        "date",
        "event_date",
        "event_date_parsed",
    ]

    date_column = None

    for column in possible_date_columns:
        if column in events.columns:
            date_column = column
            break

    if date_column is None:
        raise ValueError(
            "Não encontrei coluna de data em events.csv. "
            f"Colunas disponíveis: {list(events.columns)}"
        )

    events["event_date"] = pd.to_datetime(
        events[date_column],
        errors="coerce",
    )

    fights = fights.merge(
        events[["event_id", "event_date"]],
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    if fights["event_date"].isna().any():
        missing = int(fights["event_date"].isna().sum())

        raise ValueError(
            f"{missing} lutas ficaram sem event_date após o merge."
        )

    # --------------------------------------------------------
    # ORDEM CANÔNICA DOS LUTADORES
    #
    # O BOUT original possui uma orientação que, principalmente
    # nos eventos antigos, apresenta forte relação com o vencedor.
    #
    # Para o experimento, não queremos que "fighter_1" carregue
    # essa informação.
    #
    # A posição passa a depender SOMENTE do fighter_id:
    #
    #     fighter_1_id < fighter_2_id
    #
    # Quando necessário, trocamos juntos:
    #     - fighter_id
    #     - fighter_name
    #     - result
    #
    # O fights.csv original não é modificado.
    # --------------------------------------------------------

    swap_mask = (
        fights["fighter_1_id"].notna()
        & fights["fighter_2_id"].notna()
        & (
            fights["fighter_1_id"]
            > fights["fighter_2_id"]
        )
    )

    for column_1, column_2 in [
        ("fighter_1_id", "fighter_2_id"),
        ("fighter_1_name", "fighter_2_name"),
        ("fighter_1_result", "fighter_2_result"),
    ]:

        temporary = fights.loc[
            swap_mask,
            column_1,
        ].copy()

        fights.loc[
            swap_mask,
            column_1,
        ] = fights.loc[
            swap_mask,
            column_2,
        ].values

        fights.loc[
            swap_mask,
            column_2,
        ] = temporary.values

    # --------------------------------------------------------
    # VALIDAÇÃO DA ORDEM
    # --------------------------------------------------------

    invalid_order = fights[
        fights["fighter_1_id"].notna()
        & fights["fighter_2_id"].notna()
        & (
            fights["fighter_1_id"]
            >= fights["fighter_2_id"]
        )
    ]

    if not invalid_order.empty:
        raise RuntimeError(
            "Falha na ordenação canônica dos lutadores: "
            f"{len(invalid_order)} lutas."
        )

    # --------------------------------------------------------
    # ORDENAÇÃO TEMPORAL
    # --------------------------------------------------------

    fights = fights.sort_values(
        ["event_date", "fight_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return fights


# ============================================================
# PREPARE ROUND STATS
# ============================================================


def prepare_round_stats(
    round_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converte stats textuais para valores numéricos.
    """

    stats = round_stats.copy()

    landed_attempted_columns = [
        "significant_strikes",
        "total_strikes",
        "takedowns",
        "head_strikes",
        "body_strikes",
        "leg_strikes",
        "distance_strikes",
        "clinch_strikes",
        "ground_strikes",
    ]

    for column in landed_attempted_columns:
        landed_values = []
        attempted_values = []

        for value in stats[column]:
            landed, attempted = parse_landed_attempted(value)
            landed_values.append(landed)
            attempted_values.append(attempted)

        prefix = column.replace("_strikes", "")

        stats[f"_{column}_landed"] = landed_values
        stats[f"_{column}_attempted"] = attempted_values

    stats["_control_time_seconds"] = stats["control_time"].apply(
        parse_time_seconds
    )

    numeric_columns = [
        "knockdowns",
        "submission_attempts",
        "reversals",
    ]

    for column in numeric_columns:
        stats[column] = pd.to_numeric(
            stats[column],
            errors="coerce",
        )

    return stats


# ============================================================
# AGGREGATE ROUND STATS -> FIGHT STATS
# ============================================================


def aggregate_fight_performance(
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transforma os stats por round em stats por luta.

    Percentuais NÃO são calculados como média dos percentuais
    dos rounds.

    Exemplo:
        round 1 = 10/20
        round 2 = 90/100

    resultado da luta:
        100/120 = 83.33%

    e não:
        (50% + 90%) / 2
    """

    group_columns = [
        "fight_id",
        "fighter_id",
    ]

    grouped = (
        stats
        .groupby(group_columns, as_index=False)
        .agg(
            sig_str_landed=(
                "_significant_strikes_landed",
                "sum",
            ),
            sig_str_attempted=(
                "_significant_strikes_attempted",
                "sum",
            ),
            total_str_landed=(
                "_total_strikes_landed",
                "sum",
            ),
            total_str_attempted=(
                "_total_strikes_attempted",
                "sum",
            ),
            takedowns=(
                "_takedowns_landed",
                "sum",
            ),
            td_attempted=(
                "_takedowns_attempted",
                "sum",
            ),
            submission_attempts=(
                "submission_attempts",
                "sum",
            ),
            reversals=(
                "reversals",
                "sum",
            ),
            knockdowns=(
                "knockdowns",
                "sum",
            ),
            control_time=(
                "_control_time_seconds",
                "sum",
            ),
            rounds_with_stats=(
                "round",
                "count",
            ),
        )
    )

    # Se todos os valores de uma métrica forem NaN dentro de uma
    # luta, o sum padrão poderia retornar 0. Restauramos NaN nesses casos.
    metric_sources = {
        "sig_str_landed": "_significant_strikes_landed",
        "sig_str_attempted": "_significant_strikes_attempted",
        "total_str_landed": "_total_strikes_landed",
        "total_str_attempted": "_total_strikes_attempted",
        "takedowns": "_takedowns_landed",
        "td_attempted": "_takedowns_attempted",
        "submission_attempts": "submission_attempts",
        "reversals": "reversals",
        "knockdowns": "knockdowns",
        "control_time": "_control_time_seconds",
    }

    for output_column, source_column in metric_sources.items():
        availability = (
            stats
            .groupby(group_columns)[source_column]
            .apply(lambda x: x.notna().any())
            .reset_index(name="_has_data")
        )

        grouped = grouped.merge(
            availability,
            on=group_columns,
            how="left",
        )

        grouped.loc[
            ~grouped["_has_data"],
            output_column,
        ] = np.nan

        grouped = grouped.drop(columns="_has_data")

    # Eficiência recalculada a partir dos totais da luta.
    grouped["sig_str_pct"] = np.where(
        grouped["sig_str_attempted"] > 0,
        grouped["sig_str_landed"]
        / grouped["sig_str_attempted"],
        np.nan,
    )

    grouped["td_pct"] = np.where(
        grouped["td_attempted"] > 0,
        grouped["takedowns"]
        / grouped["td_attempted"],
        np.nan,
    )

    return grouped


# ============================================================
# PERFORMANCE STATE
# ============================================================


def performance_from_row(
    row: pd.Series,
) -> Optional[FightPerformance]:
    """
    Converte uma linha agregada de performance em objeto.

    Retorna None quando não existe nenhum stat utilizável.
    """

    performance_columns = [
        "sig_str_landed",
        "sig_str_attempted",
        "total_str_landed",
        "total_str_attempted",
        "takedowns",
        "td_attempted",
        "submission_attempts",
        "reversals",
        "knockdowns",
        "control_time",
    ]

    if not any(pd.notna(row[column]) for column in performance_columns):
        return None

    return FightPerformance(
        sig_str_landed=row["sig_str_landed"],
        sig_str_attempted=row["sig_str_attempted"],
        total_str_landed=row["total_str_landed"],
        total_str_attempted=row["total_str_attempted"],
        takedowns=row["takedowns"],
        td_attempted=row["td_attempted"],
        submission_attempts=row["submission_attempts"],
        reversals=row["reversals"],
        knockdowns=row["knockdowns"],
        control_time=row["control_time"],
    )


def update_running_average(
    state: Dict[str, float],
    prefix: str,
    value: float,
) -> None:
    """
    Atualiza média incrementalmente.

    Estado:
        {prefix}_sum
        {prefix}_count
    """

    if pd.isna(value):
        return

    state[f"{prefix}_sum"] = (
        state.get(f"{prefix}_sum", 0.0) + float(value)
    )

    state[f"{prefix}_count"] = (
        state.get(f"{prefix}_count", 0) + 1
    )


def get_running_average(
    state: Dict[str, float],
    prefix: str,
) -> float:
    count = state.get(f"{prefix}_count", 0)

    if count == 0:
        return np.nan

    return (
        state[f"{prefix}_sum"]
        / count
    )


def get_performance_features(
    state: Dict[str, float],
) -> Dict[str, float]:
    """
    Retorna o estado histórico de performance antes da luta.
    """

    return {
        "sig_str_landed_per_fight": get_running_average(
            state,
            "sig_str_landed",
        ),
        "sig_str_attempted_per_fight": get_running_average(
            state,
            "sig_str_attempted",
        ),
        "sig_str_pct": get_running_average(
            state,
            "sig_str_pct",
        ),
        "total_str_landed_per_fight": get_running_average(
            state,
            "total_str_landed",
        ),
        "total_str_attempted_per_fight": get_running_average(
            state,
            "total_str_attempted",
        ),
        "takedowns_per_fight": get_running_average(
            state,
            "takedowns",
        ),
        "td_attempted_per_fight": get_running_average(
            state,
            "td_attempted",
        ),
        "td_pct": get_running_average(
            state,
            "td_pct",
        ),
        "submission_attempts_per_fight": get_running_average(
            state,
            "submission_attempts",
        ),
        "reversals_per_fight": get_running_average(
            state,
            "reversals",
        ),
        "knockdowns_per_fight": get_running_average(
            state,
            "knockdowns",
        ),
        "control_time_per_fight": get_running_average(
            state,
            "control_time",
        ),
    }


def update_performance_state(
    state: Dict[str, float],
    performance: Optional[FightPerformance],
) -> None:
    """
    Atualiza o histórico somente depois da previsão.

    Cada luta recebe peso 1, independentemente do número de rounds.
    """

    if performance is None:
        return

    update_running_average(
        state,
        "sig_str_landed",
        performance.sig_str_landed,
    )

    update_running_average(
        state,
        "sig_str_attempted",
        performance.sig_str_attempted,
    )

    if (
        pd.notna(performance.sig_str_landed)
        and pd.notna(performance.sig_str_attempted)
        and performance.sig_str_attempted > 0
    ):
        sig_pct = (
            performance.sig_str_landed
            / performance.sig_str_attempted
        )

        update_running_average(
            state,
            "sig_str_pct",
            sig_pct,
        )

    update_running_average(
        state,
        "total_str_landed",
        performance.total_str_landed,
    )

    update_running_average(
        state,
        "total_str_attempted",
        performance.total_str_attempted,
    )

    update_running_average(
        state,
        "takedowns",
        performance.takedowns,
    )

    update_running_average(
        state,
        "td_attempted",
        performance.td_attempted,
    )

    if (
        pd.notna(performance.takedowns)
        and pd.notna(performance.td_attempted)
        and performance.td_attempted > 0
    ):
        td_pct = (
            performance.takedowns
            / performance.td_attempted
        )

        update_running_average(
            state,
            "td_pct",
            td_pct,
        )

    update_running_average(
        state,
        "submission_attempts",
        performance.submission_attempts,
    )

    update_running_average(
        state,
        "reversals",
        performance.reversals,
    )

    update_running_average(
        state,
        "knockdowns",
        performance.knockdowns,
    )

    update_running_average(
        state,
        "control_time",
        performance.control_time,
    )


# ============================================================
# ELO HELPERS
# ============================================================


def expected_score(
    rating_a: float,
    rating_b: float,
) -> float:
    return 1.0 / (
        1.0
        + 10.0 ** (
            (rating_b - rating_a)
            / ELO_SCALE
        )
    )


def get_k_factor(
    fights: int,
) -> float:
    if fights < PROVISIONAL_FIGHTS:
        return BOOSTED_K

    return BASE_K


def update_elo(
    state_a: FighterState,
    state_b: FighterState,
    score_a: float,
) -> None:
    """
    Atualiza Elo após a luta.

    score_a:
        1.0 = vitória A
        0.0 = derrota A
    """

    rating_a = state_a.rating
    rating_b = state_b.rating

    probability_a = expected_score(
        rating_a,
        rating_b,
    )

    k_a = get_k_factor(state_a.fights)
    k_b = get_k_factor(state_b.fights)

    state_a.rating = (
        rating_a
        + k_a
        * (score_a - probability_a)
    )

    state_b.rating = (
        rating_b
        + k_b
        * ((1.0 - score_a) - (1.0 - probability_a))
    )

    state_a.fights += 1
    state_b.fights += 1


# ============================================================
# BUILD HISTORICAL DATASET
# ============================================================


def build_historical_dataset(
    fights: pd.DataFrame,
    fight_performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Percorre todas as lutas em ordem temporal.

    Para cada luta salva:
        - Elo antes da luta;
        - performance histórica antes da luta;
        - resultado;
        - diferenças entre os lutadores.

    Só depois da observação é que os estados são atualizados.
    """

    performance_lookup = {
        (row["fight_id"], row["fighter_id"]): row
        for _, row in fight_performance.iterrows()
    }

    elo_states: Dict[str, FighterState] = {}
    performance_states: Dict[str, Dict[str, float]] = {}

    rows: List[dict] = []

    for _, fight in fights.iterrows():
        fight_id = fight["fight_id"]

        fighter_a = fight["fighter_1_id"]
        fighter_b = fight["fighter_2_id"]

        result_a = str(
            fight["fighter_1_result"]
        ).strip().upper()

        result_b = str(
            fight["fighter_2_result"]
        ).strip().upper()

        state_a = elo_states.setdefault(
            fighter_a,
            FighterState(),
        )

        state_b = elo_states.setdefault(
            fighter_b,
            FighterState(),
        )

        perf_state_a = performance_states.setdefault(
            fighter_a,
            {},
        )

        perf_state_b = performance_states.setdefault(
            fighter_b,
            {},
        )

        perf_a = get_performance_features(
            perf_state_a
        )

        perf_b = get_performance_features(
            perf_state_b
        )

        elo_probability_a = expected_score(
            state_a.rating,
            state_b.rating,
        )

        row = {
            "fight_id": fight_id,
            "event_id": fight["event_id"],
            "event_date": fight["event_date"],
            "fighter_1_id": fighter_a,
            "fighter_2_id": fighter_b,
            "fighter_1_name": fight["fighter_1_name"],
            "fighter_2_name": fight["fighter_2_name"],
            "fighter_1_result": result_a,
            "fighter_2_result": result_b,
            "weight_class": fight["weight_class"],
            "elo_a": state_a.rating,
            "elo_b": state_b.rating,
            "elo_difference": (
                state_a.rating
                - state_b.rating
            ),
            "elo_probability": elo_probability_a,
        }

        for feature in ALL_PERFORMANCE_FEATURES:
            value_a = perf_a[feature]
            value_b = perf_b[feature]

            row[f"{feature}_a"] = value_a
            row[f"{feature}_b"] = value_b

            if pd.notna(value_a) and pd.notna(value_b):
                row[f"{feature}_difference"] = (
                    value_a - value_b
                )
            else:
                row[f"{feature}_difference"] = np.nan

        # Resultado utilizado somente para avaliação e posterior update.
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

        performance_row_a = performance_lookup.get(
            (fight_id, fighter_a)
        )

        performance_row_b = performance_lookup.get(
            (fight_id, fighter_b)
        )

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

        update_performance_state(
            perf_state_a,
            performance_a,
        )

        update_performance_state(
            perf_state_b,
            performance_b,
        )

        # NC e empate não alteram Elo.
        if result_a == "W" and result_b == "L":
            update_elo(
                state_a,
                state_b,
                score_a=1.0,
            )

        elif result_a == "L" and result_b == "W":
            update_elo(
                state_a,
                state_b,
                score_a=0.0,
            )

        # NC/NC e D/D:
        # não atualizam rating nem experiência Elo.

    historical = pd.DataFrame(rows)

    return historical


# ============================================================
# MODEL HELPERS
# ============================================================


MODEL_FEATURES = {
    "M0_Elo": [
        "elo_difference",
    ],
    "M1_Elo_Striking": [
        "elo_difference",
        *[
            f"{feature}_difference"
            for feature in STRIKING_FEATURES
        ],
    ],
    "M2_Elo_Grappling": [
        "elo_difference",
        *[
            f"{feature}_difference"
            for feature in GRAPPLING_FEATURES
        ],
    ],
    "M3_Elo_Dominance": [
        "elo_difference",
        *[
            f"{feature}_difference"
            for feature in DOMINANCE_FEATURES
        ],
    ],
    "M4_Elo_All_Performance": [
        "elo_difference",
        *[
            f"{feature}_difference"
            for feature in ALL_PERFORMANCE_FEATURES
        ],
    ],
    "M5_Performance_Only": [
        *[
            f"{feature}_difference"
            for feature in ALL_PERFORMANCE_FEATURES
        ],
    ],
}


def create_logistic_pipeline() -> Pipeline:
    """
    Imputer + StandardScaler + LogisticRegression.

    Todos os parâmetros são ajustados exclusivamente no treino
    do fold.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
    }


# ============================================================
# BASELINE ELO
# ============================================================


def evaluate_elo_baseline(
    test: pd.DataFrame,
) -> Dict[str, float]:
    valid = test[
        test["target"].notna()
        & test["elo_probability"].notna()
    ].copy()

    y_true = valid["target"].astype(int).to_numpy()
    probabilities = valid["elo_probability"].to_numpy()

    return calculate_metrics(
        y_true,
        probabilities,
    )


# ============================================================
# COEFFICIENTS
# ============================================================


def extract_coefficients(
    pipeline: Pipeline,
    model_name: str,
    fold_name: str,
    features: List[str],
) -> List[dict]:
    """
    Salva coeficientes da regressão logística.

    Como as features são padronizadas, os coeficientes são
    comparáveis dentro do modelo/fold.
    """

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    logistic = pipeline.named_steps["logistic"]

    original_features = list(features)

    indicator_count = len(
        imputer.indicator_.features_
    ) if imputer.add_indicator else 0

    feature_names = original_features.copy()

    if indicator_count:
        feature_names.extend(
            [
                f"{original_features[index]}__missing"
                for index in imputer.indicator_.features_
            ]
        )

    coefficients = logistic.coef_[0]

    rows = []

    for index, coefficient in enumerate(coefficients):
        feature_name = (
            feature_names[index]
            if index < len(feature_names)
            else f"feature_{index}"
        )

        rows.append(
            {
                "model": model_name,
                "fold": fold_name,
                "feature": feature_name,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
            }
        )

    return rows


# ============================================================
# TEMPORAL EXPERIMENT
# ============================================================


def run_temporal_experiment(
    historical: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    temporal_results = []
    prediction_rows = []
    coefficient_rows = []

    for fold_name, test_start, test_end in FOLDS:

        start = pd.Timestamp(test_start)
        end = pd.Timestamp(test_end)

        train = historical[
            (historical["event_date"] < start)
            & historical["target"].notna()
        ].copy()

        test = historical[
            (historical["event_date"] >= start)
            & (historical["event_date"] < end)
            & historical["target"].notna()
        ].copy()

        print()
        print("-" * 70)
        print(f"Fold: {fold_name}")
        print(
            f"Treino: antes de {test_start}"
        )
        print(
            f"Teste: {test_start} -> {test_end}"
        )
        print(f"Lutas treino: {len(train)}")
        print(f"Lutas teste:  {len(test)}")

        if test.empty:
            print("Fold sem dados. Ignorando.")
            continue

        # ====================================================
        # BASELINE ELO
        # ====================================================

        elo_metrics = evaluate_elo_baseline(test)

        print()
        print("Elo baseline:")
        print(
            f"  Log Loss: {elo_metrics['log_loss']:.4f}"
        )
        print(
            f"  Brier:    {elo_metrics['brier']:.4f}"
        )
        print(
            f"  AUC:      {elo_metrics['auc']:.4f}"
        )
        print(
            f"  Accuracy: {elo_metrics['accuracy']:.4f}"
        )

        temporal_results.append(
            {
                "fold": fold_name,
                "model": "M0_Elo",
                "n_train": len(train),
                "n_test": len(test),
                **elo_metrics,
            }
        )

        for _, row in test.iterrows():
            prediction_rows.append(
                {
                    "fold": fold_name,
                    "model": "M0_Elo",
                    "fight_id": row["fight_id"],
                    "event_date": row["event_date"],
                    "fighter_1_id": row["fighter_1_id"],
                    "fighter_2_id": row["fighter_2_id"],
                    "fighter_1_name": row["fighter_1_name"],
                    "fighter_2_name": row["fighter_2_name"],
                    "target": int(row["target"]),
                    "probability_fighter_1": row[
                        "elo_probability"
                    ],
                }
            )

        # ====================================================
        # PERFORMANCE MODELS
        # ====================================================

        for model_name, features in MODEL_FEATURES.items():

            if model_name == "M0_Elo":
                continue

            print()
            print(f"{model_name}:")

            X_train = train[features]
            y_train = train["target"].astype(int)

            X_test = test[features]
            y_test = test["target"].astype(int)

            pipeline = create_logistic_pipeline()

            print("\n=== DIAGNÓSTICO ===")
            print("Modelo:", model_name)
            print("Fold:", fold_name)
            print("Target treino:", y_train.mean())

            print("\nMissing X_train:")
            print(X_train.isna().mean().sort_values(ascending=False).to_string())

            print("\nMissing X_test:")
            print(X_test.isna().mean().sort_values(ascending=False).to_string())

            print("\nMédias X_train:")
            print(X_train.mean(numeric_only=True).to_string())

            print("\nMédias X_test:")
            print(X_test.mean(numeric_only=True).to_string())

            pipeline.fit(
                X_train,
                y_train,
            )

            probabilities = pipeline.predict_proba(
                X_test
            )[:, 1]

            metrics = calculate_metrics(
                y_test.to_numpy(),
                probabilities,
            )

            print(
                f"  Log Loss: {metrics['log_loss']:.4f}"
            )
            print(
                f"  Brier:    {metrics['brier']:.4f}"
            )
            print(
                f"  AUC:      {metrics['auc']:.4f}"
            )
            print(
                f"  Accuracy: {metrics['accuracy']:.4f}"
            )

            temporal_results.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics,
                }
            )

            coefficient_rows.extend(
                extract_coefficients(
                    pipeline,
                    model_name,
                    fold_name,
                    features,
                )
            )

            for row_index, (_, row) in enumerate(
                test.iterrows()
            ):
                prediction_rows.append(
                    {
                        "fold": fold_name,
                        "model": model_name,
                        "fight_id": row["fight_id"],
                        "event_date": row["event_date"],
                        "fighter_1_id": row["fighter_1_id"],
                        "fighter_2_id": row["fighter_2_id"],
                        "fighter_1_name": row["fighter_1_name"],
                        "fighter_2_name": row["fighter_2_name"],
                        "target": int(row["target"]),
                        "probability_fighter_1": probabilities[
                            row_index
                        ],
                    }
                )

    return (
        pd.DataFrame(temporal_results),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(coefficient_rows),
    )


# ============================================================
# POOLED RESULTS
# ============================================================


def calculate_pooled_results(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for model_name, group in predictions.groupby(
        "model"
    ):
        y_true = group["target"].astype(int).to_numpy()

        probabilities = group[
            "probability_fighter_1"
        ].to_numpy()

        metrics = calculate_metrics(
            y_true,
            probabilities,
        )

        rows.append(
            {
                "model": model_name,
                "n_fights": len(group),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("=" * 70)
    print("Performance — Ultimate Fighter Rank")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    fights, events, round_stats = load_data()

    print()
    print(f"Lutas carregadas: {len(fights)}")
    print(f"Rounds/stats carregados: {len(round_stats)}")

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    fights = prepare_fights(
        fights,
        events,
    )

    round_stats = prepare_round_stats(
        round_stats
    )

    fight_performance = aggregate_fight_performance(
        round_stats
    )

    print()
    print(
        f"Lutas com performance agregada: "
        f"{fight_performance['fight_id'].nunique()}"
    )

    missing_stats = (
        len(fights)
        - fights["fight_id"]
        .isin(
            fight_performance["fight_id"]
        )
        .sum()
    )

    print(
        f"Lutas sem stats: {missing_stats}"
    )

    # --------------------------------------------------------
    # HISTORICAL DATASET
    # --------------------------------------------------------

    print()
    print("Construindo histórico temporal...")

    historical = build_historical_dataset(
        fights,
        fight_performance,
    )

    valid_results = historical[
        historical["target"].notna()
    ]

    print(
        f"Lutas W/L válidas: "
        f"{len(valid_results)}"
    )

    print(
        f"Lutas NC/empate ignoradas: "
        f"{len(historical) - len(valid_results)}"
    )

    # --------------------------------------------------------
    # TEMPORAL EXPERIMENT
    # --------------------------------------------------------

    temporal_results, predictions, coefficients = (
        run_temporal_experiment(
            historical
        )
    )

    # --------------------------------------------------------
    # POOLED
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTADO POOLED")
    print("=" * 70)

    pooled = calculate_pooled_results(
        predictions
    )

    for _, row in pooled.iterrows():

        print()
        print(f"{row['model']}:")

        print(
            f"  Lutas:    {int(row['n_fights'])}"
        )

        print(
            f"  Log Loss: {row['log_loss']:.4f}"
        )

        print(
            f"  Brier:    {row['brier']:.4f}"
        )

        print(
            f"  AUC:      {row['auc']:.4f}"
        )

        print(
            f"  Accuracy: {row['accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal_results.to_csv(
        TEMPORAL_RESULTS_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )

    print()
    print("Resultados salvos em:")

    print(
        f"  {TEMPORAL_RESULTS_PATH}"
    )

    print(
        f"  {PREDICTIONS_PATH}"
    )

    print(
        f"  {COEFFICIENTS_PATH}"
    )

    print()
    print("=" * 70)
    print("Experimento concluído.")
    print("=" * 70)


if __name__ == "__main__":
    main()