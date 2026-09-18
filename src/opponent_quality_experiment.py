"""
Experimento de Qualidade do Oponente — Ultimate Fighter Rank.

Hipótese:
    A força dos adversários enfrentados por cada lutador contém
    informação adicional ao rating Elo atual.

Metodologia:
    - Elo é atualizado cronologicamente, luta a luta.
    - Para cada lutador, calculamos Strength of Schedule (SoS):
        média dos ratings Elo dos adversários enfrentados anteriormente.
    - A previsão utiliza:
        diferença de Elo
        diferença de SoS
    - Os pesos da combinação são aprendidos SOMENTE com dados
      anteriores ao período de teste.
    - Walk-forward temporal, sem random split.
    - Cada luta é prevista antes de atualizar os ratings.
    - NC e empates não entram na avaliação nem atualizam Elo.

Modelo:
    Elo baseline:
        P(A) = sigmoid(q * (Elo_A - Elo_B))

    Elo + Opponent Quality:
        P(A) = sigmoid(
            intercept
            + beta_elo * (Elo_A - Elo_B)
            + beta_sos * (SoS_A - SoS_B)
        )

A combinação é uma regressão logística simples e temporalmente
treinada. Isso permite testar se SoS adiciona informação além
do Elo, sem escolher manualmente um peso arbitrário.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

FIGHTS_PATH = PROCESSED_DIR / "fights.csv"
EVENTS_PATH = PROCESSED_DIR / "events.csv"

RESULTS_PATH = (
    RESULTS_DIR / "opponent_quality_temporal_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR / "opponent_quality_predictions.csv"
)


# ============================================================
# ELO
# ============================================================

INITIAL_RATING = 1500.0

BASE_K = 32.0
BOOSTED_K = 64.0

ELO_SCALE = 400.0

PROVISIONAL_FIGHTS = 3


# ============================================================
# WALK-FORWARD
# ============================================================

WALK_FORWARD_FIRST_TEST_START = "2011-01-01"
WALK_FORWARD_STEP_YEARS = 3


# ============================================================
# ESTADO DO LUTADOR
# ============================================================

@dataclass
class FighterState:
    """
    Estado histórico de um lutador.

    rating:
        Elo atual.

    fights:
        número de lutas válidas já disputadas.

    opponent_rating_sum:
        soma dos ratings pré-luta dos adversários.

    opponent_count:
        quantidade de adversários válidos enfrentados.
    """

    rating: float = INITIAL_RATING
    fights: int = 0

    opponent_rating_sum: float = 0.0
    opponent_count: int = 0

    @property
    def opponent_quality(self) -> float:
        """
        Strength of Schedule.

        Média do rating Elo dos adversários enfrentados
        anteriormente.

        Para um lutador sem histórico:
            retorna INITIAL_RATING.
        """

        if self.opponent_count == 0:
            return INITIAL_RATING

        return (
            self.opponent_rating_sum
            / self.opponent_count
        )


# ============================================================
# UTILITÁRIOS
# ============================================================

def get_or_create_state(
    states: dict[str, FighterState],
    fighter_id: str,
) -> FighterState:
    """
    Retorna o estado de um lutador.

    Se não existir, cria com Elo inicial.
    """

    fighter_id = str(fighter_id)

    if fighter_id not in states:
        states[fighter_id] = FighterState()

    return states[fighter_id]


def expected_score(
    rating_a: float,
    rating_b: float,
) -> float:
    """
    Probabilidade Elo de A vencer B.
    """

    exponent = (
        -(rating_a - rating_b)
        / ELO_SCALE
    )

    probability = (
        1.0
        / (1.0 + 10.0 ** exponent)
    )

    return float(
        np.clip(
            probability,
            1e-15,
            1.0 - 1e-15,
        )
    )


def get_k_factor(
    fighter_a: FighterState,
    fighter_b: FighterState,
) -> float:
    """
    K-factor usado pelo Elo v0.1.

    Um lutador permanece provisional durante suas
    primeiras PROVISIONAL_FIGHTS lutas.

    Se qualquer um dos dois lutadores estiver
    provisional, usamos BOOSTED_K.
    """

    if (
        fighter_a.fights < PROVISIONAL_FIGHTS
        or fighter_b.fights < PROVISIONAL_FIGHTS
    ):
        return BOOSTED_K

    return BASE_K


# ============================================================
# RESULTADO
# ============================================================

def get_outcome(
    fighter_1_result: str,
    fighter_2_result: str,
) -> float | None:
    """
    Converte o resultado:

        1.0 -> fighter 1 venceu
        0.0 -> fighter 2 venceu
        None -> NC / empate / inválido
    """

    result_1 = str(
        fighter_1_result
    ).strip().upper()

    result_2 = str(
        fighter_2_result
    ).strip().upper()

    if result_1 == "W" and result_2 == "L":
        return 1.0

    if result_1 == "L" and result_2 == "W":
        return 0.0

    return None


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def prepare_fights(
    fights: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara fights.csv.

    A data é recuperada de events.csv através de event_id.
    """

    fights = fights.copy()
    events = events.copy()

    fights["event_id"] = (
        fights["event_id"].astype(str)
    )

    events["event_id"] = (
        events["event_id"].astype(str)
    )

    events_required = [
        "event_id",
        "date",
    ]

    missing = [
        column
        for column in events_required
        if column not in events.columns
    ]

    if missing:
        raise ValueError(
            "events.csv não possui as colunas necessárias: "
            f"{missing}"
        )

    fights = fights.merge(
        events[events_required],
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    fights["date"] = pd.to_datetime(
        fights["date"],
        errors="coerce",
    )

    if fights["date"].isna().any():
        missing_dates = int(
            fights["date"].isna().sum()
        )

        raise ValueError(
            f"{missing_dates} lutas ficaram sem data."
        )

    fights["fighter_a_id"] = (
        fights["fighter_1_id"].astype(str)
    )

    fights["fighter_b_id"] = (
        fights["fighter_2_id"].astype(str)
    )

    fights["outcome"] = [
        get_outcome(
            result_1,
            result_2,
        )
        for result_1, result_2 in zip(
            fights["fighter_1_result"],
            fights["fighter_2_result"],
        )
    ]

    fights = fights.sort_values(
        ["date", "fight_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return fights


# ============================================================
# WALK-FORWARD
# ============================================================

def generate_walk_forward_folds(
    dataset: pd.DataFrame,
    first_test_start: str,
    step_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Gera os mesmos folds do experimento oficial.

    Exemplo:

        2011-2014
        2014-2017
        2017-2020
        2020-2023
        2023-2026
        2026-2029
    """

    first_test_start = pd.Timestamp(
        first_test_start
    )

    max_date = dataset["date"].max()

    folds = []

    test_start = first_test_start

    while test_start <= max_date:

        test_end = (
            test_start
            + pd.DateOffset(years=step_years)
        )

        folds.append(
            (
                test_start,
                test_end,
            )
        )

        test_start = test_end

    return folds


# ============================================================
# ATUALIZAÇÃO DO ELO
# ============================================================

def update_fight(
    states: dict[str, FighterState],
    fighter_a_id: str,
    fighter_b_id: str,
    outcome: float,
) -> None:
    """
    Atualiza Elo e Strength of Schedule após uma luta.

    IMPORTANTE:
        Os ratings pré-luta são capturados antes
        de qualquer atualização.
    """

    fighter_a_id = str(fighter_a_id)
    fighter_b_id = str(fighter_b_id)

    state_a = get_or_create_state(
        states,
        fighter_a_id,
    )

    state_b = get_or_create_state(
        states,
        fighter_b_id,
    )

    # --------------------------------------------------------
    # Estado pré-luta
    # --------------------------------------------------------

    rating_a = state_a.rating
    rating_b = state_b.rating

    # --------------------------------------------------------
    # Atualiza Strength of Schedule
    #
    # Cada lutador recebe o rating pré-luta
    # do adversário como uma observação de qualidade.
    # --------------------------------------------------------

    state_a.opponent_rating_sum += rating_b
    state_a.opponent_count += 1

    state_b.opponent_rating_sum += rating_a
    state_b.opponent_count += 1

    # --------------------------------------------------------
    # Elo
    # --------------------------------------------------------

    probability_a = expected_score(
        rating_a,
        rating_b,
    )

    if float(outcome) == 1.0:
        score_a = 1.0
        score_b = 0.0
    else:
        score_a = 0.0
        score_b = 1.0

    k_factor = get_k_factor(
        state_a,
        state_b,
    )

    state_a.rating = (
        rating_a
        + k_factor
        * (score_a - probability_a)
    )

    state_b.rating = (
        rating_b
        + k_factor
        * (score_b - (1.0 - probability_a))
    )

    # --------------------------------------------------------
    # Experiência
    # --------------------------------------------------------

    state_a.fights += 1
    state_b.fights += 1


# ============================================================
# CONSTRUÇÃO DO ESTADO
# ============================================================

def build_historical_state(
    fights: pd.DataFrame,
) -> dict[str, FighterState]:
    """
    Reproduz cronologicamente todas as lutas históricas
    anteriores ao período de teste.
    """

    states: dict[str, FighterState] = {}

    for row in fights.itertuples(
        index=False
    ):

        outcome = row.outcome

        if pd.isna(outcome):
            continue

        update_fight(
            states=states,
            fighter_a_id=row.fighter_a_id,
            fighter_b_id=row.fighter_b_id,
            outcome=float(outcome),
        )

    return states


# ============================================================
# FEATURES DE OPPONENT QUALITY
# ============================================================

def get_features(
    state_a: FighterState,
    state_b: FighterState,
) -> tuple[float, float]:
    """
    Retorna:

        elo_difference
        sos_difference

    Tudo é calculado usando somente o estado
    disponível antes da luta.
    """

    elo_difference = (
        state_a.rating
        - state_b.rating
    )

    sos_difference = (
        state_a.opponent_quality
        - state_b.opponent_quality
    )

    return (
        float(elo_difference),
        float(sos_difference),
    )


# ============================================================
# MODELO DE OPPONENT QUALITY
# ============================================================

def fit_opponent_quality_model(
    train_features: pd.DataFrame,
) -> LogisticRegression:
    """
    Treina o modelo:

        P(A) =
            sigmoid(
                intercept
                + beta_elo * elo_difference
                + beta_sos * sos_difference
            )

    O modelo é treinado exclusivamente no passado
    disponível antes do fold de teste.
    """

    if train_features.empty:
        raise ValueError(
            "Não existem features suficientes para "
            "treinar o modelo de Opponent Quality."
        )

    X = train_features[
        [
            "elo_difference",
            "sos_difference",
        ]
    ]

    y = train_features["y"].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "O conjunto de treino possui apenas uma classe."
        )

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float]:
    """
    Calcula:

        Log Loss
        Brier
        AUC
        Accuracy
    """

    y_true = pd.Series(
        y_true
    ).astype(int)

    probabilities = pd.Series(
        probabilities
    ).astype(float)

    probabilities = probabilities.clip(
        1e-15,
        1.0 - 1e-15,
    )

    metrics = {
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "brier": float(
            np.mean(
                (
                    probabilities
                    - y_true
                ) ** 2
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                probabilities >= 0.5,
            )
        ),
    }

    if y_true.nunique() >= 2:
        metrics["auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    else:
        metrics["auc"] = np.nan

    return metrics


# ============================================================
# FEATURES HISTÓRICAS PARA TREINAMENTO
# ============================================================

def build_training_features(
    fights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria features históricas para todas as lutas do conjunto
    de treinamento.

    Cada linha utiliza apenas informações disponíveis
    imediatamente antes daquela luta.

    Além das features, o estado Elo/SoS é atualizado somente
    depois da criação da linha.
    """

    states: dict[str, FighterState] = {}

    rows = []

    for row in fights.itertuples(
        index=False
    ):

        outcome = row.outcome

        if pd.isna(outcome):
            continue

        fighter_a = str(
            row.fighter_a_id
        )

        fighter_b = str(
            row.fighter_b_id
        )

        state_a = get_or_create_state(
            states,
            fighter_a,
        )

        state_b = get_or_create_state(
            states,
            fighter_b,
        )

        elo_difference, sos_difference = (
            get_features(
                state_a,
                state_b,
            )
        )

        # ----------------------------------------------------
        # Guarda somente o estado pré-luta
        # ----------------------------------------------------

        rows.append(
            {
                "fight_id": row.fight_id,
                "date": row.date,
                "elo_difference": elo_difference,
                "sos_difference": sos_difference,
                "y": float(outcome),
            }
        )

        # ----------------------------------------------------
        # Atualiza depois da observação
        # ----------------------------------------------------

        update_fight(
            states=states,
            fighter_a_id=fighter_a,
            fighter_b_id=fighter_b,
            outcome=float(outcome),
        )

    return pd.DataFrame(rows)


# ============================================================
# EXECUÇÃO DO FOLD
# ============================================================

def run_fold(
    dataset: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[
    dict[str, float] | None,
    pd.DataFrame,
]:
    """
    Executa um fold.

    O procedimento é:

        1. separar passado e teste;
        2. construir features históricas do passado;
        3. treinar modelo Elo + SoS;
        4. reconstruir o estado histórico;
        5. percorrer o teste cronologicamente;
        6. prever antes de atualizar;
        7. registrar Elo baseline e modelo SoS.
    """

    train = dataset[
        dataset["date"] < test_start
    ].copy()

    test = dataset[
        (dataset["date"] >= test_start)
        & (dataset["date"] < test_end)
        & dataset["outcome"].notna()
    ].copy()

    train = train.sort_values(
        ["date", "fight_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    test = test.sort_values(
        ["date", "fight_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    if test.empty:
        return None, pd.DataFrame()

    # --------------------------------------------------------
    # Features históricas do treino
    # --------------------------------------------------------

    train_features = build_training_features(
        train
    )

    if train_features.empty:
        return None, pd.DataFrame()

    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model = fit_opponent_quality_model(
        train_features
    )

    # --------------------------------------------------------
    # Estado histórico
    #
    # Recomeçamos do zero para garantir que o estado
    # utilizado no teste seja exatamente o estado
    # existente antes de test_start.
    # --------------------------------------------------------

    states = build_historical_state(
        train
    )

    predictions = []

    # --------------------------------------------------------
    # Teste sequencial
    # --------------------------------------------------------

    for row in test.itertuples(
        index=False
    ):

        fighter_a = str(
            row.fighter_a_id
        )

        fighter_b = str(
            row.fighter_b_id
        )

        outcome = float(
            row.outcome
        )

        state_a = get_or_create_state(
            states,
            fighter_a
        )

        state_b = get_or_create_state(
            states,
            fighter_b
        )

        # ----------------------------------------------------
        # Estado pré-luta
        # ----------------------------------------------------

        rating_a = state_a.rating
        rating_b = state_b.rating

        sos_a = state_a.opponent_quality
        sos_b = state_b.opponent_quality

        elo_difference = (
            rating_a
            - rating_b
        )

        sos_difference = (
            sos_a
            - sos_b
        )

        # ----------------------------------------------------
        # Elo baseline
        # ----------------------------------------------------

        probability_elo = expected_score(
            rating_a,
            rating_b,
        )

        # ----------------------------------------------------
        # Elo + Opponent Quality
        # ----------------------------------------------------

        X_test = pd.DataFrame(
            [
                {
                    "elo_difference": elo_difference,
                    "sos_difference": sos_difference,
                }
            ]
        )

        probability_sos = float(
            model.predict_proba(
                X_test
            )[0, 1]
        )

        # ----------------------------------------------------
        # Salva previsão
        # ----------------------------------------------------

        predictions.append(
            {
                "fight_id": row.fight_id,
                "date": row.date,
                "fighter_a_id": fighter_a,
                "fighter_b_id": fighter_b,
                "y": outcome,

                "rating_a_before": rating_a,
                "rating_b_before": rating_b,

                "sos_a_before": sos_a,
                "sos_b_before": sos_b,

                "elo_difference": elo_difference,
                "sos_difference": sos_difference,

                "probability_elo": probability_elo,
                "probability_elo_sos": probability_sos,
            }
        )

        # ----------------------------------------------------
        # Atualiza depois da previsão
        # ----------------------------------------------------

        update_fight(
            states=states,
            fighter_a_id=fighter_a,
            fighter_b_id=fighter_b,
            outcome=outcome,
        )

    predictions_df = pd.DataFrame(
        predictions
    )

    if predictions_df.empty:
        return None, predictions_df

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    elo_metrics = calculate_metrics(
        predictions_df["y"],
        predictions_df["probability_elo"],
    )

    sos_metrics = calculate_metrics(
        predictions_df["y"],
        predictions_df["probability_elo_sos"],
    )

    metrics = {
        "n_fights": len(predictions_df),

        "elo_log_loss": elo_metrics[
            "log_loss"
        ],
        "elo_brier": elo_metrics[
            "brier"
        ],
        "elo_auc": elo_metrics[
            "auc"
        ],
        "elo_accuracy": elo_metrics[
            "accuracy"
        ],

        "sos_log_loss": sos_metrics[
            "log_loss"
        ],
        "sos_brier": sos_metrics[
            "brier"
        ],
        "sos_auc": sos_metrics[
            "auc"
        ],
        "sos_accuracy": sos_metrics[
            "accuracy"
        ],
    }

    return metrics, predictions_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Opponent Quality — Ultimate Fighter Rank")
    print("=" * 70)

    # --------------------------------------------------------
    # Carregamento
    # --------------------------------------------------------

    if not FIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {FIGHTS_PATH}"
        )

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {EVENTS_PATH}"
        )

    print(
        f"\nCarregando: {FIGHTS_PATH}"
    )

    fights = pd.read_csv(
        FIGHTS_PATH
    )

    print(
        f"Carregando: {EVENTS_PATH}"
    )

    events = pd.read_csv(
        EVENTS_PATH
    )

    print(
        f"\nLutas carregadas: {len(fights)}"
    )

    print(
        f"Eventos carregados: {len(events)}"
    )

    # --------------------------------------------------------
    # Preparação
    # --------------------------------------------------------

    dataset = prepare_fights(
        fights=fights,
        events=events,
    )

    valid_fights = dataset[
        dataset["outcome"].notna()
    ]

    ignored_fights = dataset[
        dataset["outcome"].isna()
    ]

    print(
        f"\nDatas disponíveis: "
        f"{dataset['date'].min().date()} -> "
        f"{dataset['date'].max().date()}"
    )

    print(
        f"Lutas totais: {len(dataset)}"
    )

    print(
        f"Lutas W/L válidas: "
        f"{len(valid_fights)}"
    )

    print(
        f"Lutas NC/empate ignoradas: "
        f"{len(ignored_fights)}"
    )

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------

    folds = generate_walk_forward_folds(
        dataset=dataset,
        first_test_start=(
            WALK_FORWARD_FIRST_TEST_START
        ),
        step_years=(
            WALK_FORWARD_STEP_YEARS
        ),
    )

    print(
        f"\nFolds temporais: {len(folds)}"
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    fold_results = []
    all_predictions = []

    # --------------------------------------------------------
    # Walk-forward
    # --------------------------------------------------------

    for test_start, test_end in folds:

        print("\n" + "-" * 70)

        print(
            f"Fold: "
            f"{test_start.year}-"
            f"{test_end.year}"
        )

        print(
            f"Teste: "
            f"{test_start.date()} -> "
            f"{test_end.date()}"
        )

        metrics, predictions = run_fold(
            dataset=dataset,
            test_start=test_start,
            test_end=test_end,
        )

        if metrics is None:
            print(
                "Sem dados suficientes para este fold."
            )
            continue

        # ----------------------------------------------------
        # Identificação
        # ----------------------------------------------------

        result = {
            "fold_start": (
                test_start.date().isoformat()
            ),
            "fold_end": (
                test_end.date().isoformat()
            ),
            **metrics,
        }

        fold_results.append(
            result
        )

        predictions = predictions.copy()

        predictions["fold_start"] = (
            test_start.date().isoformat()
        )

        predictions["fold_end"] = (
            test_end.date().isoformat()
        )

        all_predictions.append(
            predictions
        )

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------

        print(
            f"Lutas: {metrics['n_fights']}"
        )

        print("\nElo baseline:")

        print(
            f"  Log Loss: "
            f"{metrics['elo_log_loss']:.4f}"
        )

        print(
            f"  Brier:    "
            f"{metrics['elo_brier']:.4f}"
        )

        print(
            f"  AUC:      "
            f"{metrics['elo_auc']:.4f}"
        )

        print(
            f"  Accuracy: "
            f"{metrics['elo_accuracy']:.4f}"
        )

        print("\nElo + Opponent Quality:")

        print(
            f"  Log Loss: "
            f"{metrics['sos_log_loss']:.4f}"
        )

        print(
            f"  Brier:    "
            f"{metrics['sos_brier']:.4f}"
        )

        print(
            f"  AUC:      "
            f"{metrics['sos_auc']:.4f}"
        )

        print(
            f"  Accuracy: "
            f"{metrics['sos_accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # Verificação
    # --------------------------------------------------------

    if not fold_results:
        raise RuntimeError(
            "Nenhum fold produziu resultados."
        )

    # --------------------------------------------------------
    # Resultados por fold
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        fold_results
    )

    # --------------------------------------------------------
    # Pooled
    # --------------------------------------------------------

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    pooled_elo = calculate_metrics(
        predictions_df["y"],
        predictions_df["probability_elo"],
    )

    pooled_sos = calculate_metrics(
        predictions_df["y"],
        predictions_df["probability_elo_sos"],
    )

    pooled_row = {
        "fold_start": "POOLED",
        "fold_end": "POOLED",
        "n_fights": len(
            predictions_df
        ),

        "elo_log_loss": pooled_elo[
            "log_loss"
        ],
        "elo_brier": pooled_elo[
            "brier"
        ],
        "elo_auc": pooled_elo[
            "auc"
        ],
        "elo_accuracy": pooled_elo[
            "accuracy"
        ],

        "sos_log_loss": pooled_sos[
            "log_loss"
        ],
        "sos_brier": pooled_sos[
            "brier"
        ],
        "sos_auc": pooled_sos[
            "auc"
        ],
        "sos_accuracy": pooled_sos[
            "accuracy"
        ],
    }

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [pooled_row]
            ),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Criação do diretório
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Salva
    # --------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTADO POOLED")
    print("=" * 70)

    print(
        f"Lutas avaliadas: "
        f"{pooled_row['n_fights']}"
    )

    print("\nElo baseline:")

    print(
        f"  Log Loss: "
        f"{pooled_row['elo_log_loss']:.4f}"
    )

    print(
        f"  Brier:    "
        f"{pooled_row['elo_brier']:.4f}"
    )

    print(
        f"  AUC:      "
        f"{pooled_row['elo_auc']:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{pooled_row['elo_accuracy']:.4f}"
    )

    print("\nElo + Opponent Quality:")

    print(
        f"  Log Loss: "
        f"{pooled_row['sos_log_loss']:.4f}"
    )

    print(
        f"  Brier:    "
        f"{pooled_row['sos_brier']:.4f}"
    )

    print(
        f"  AUC:      "
        f"{pooled_row['sos_auc']:.4f}"
    )

    print(
        f"  Accuracy: "
        f"{pooled_row['sos_accuracy']:.4f}"
    )

    print("\nResultados salvos em:")

    print(
        f"  {RESULTS_PATH}"
    )

    print(
        f"  {PREDICTIONS_PATH}"
    )

    print("\n" + "=" * 70)
    print("Experimento concluído.")
    print("=" * 70)


if __name__ == "__main__":
    main()