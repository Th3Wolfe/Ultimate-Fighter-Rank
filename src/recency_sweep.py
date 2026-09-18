"""
Experimento de recência para o Elo.

Compara o Elo v0.1 (baseline, sem recência) com versões que
aplicam decaimento temporal do rating em direção ao rating inicial.

Fluxo:

    rating_after da luta anterior
            ↓
    decaimento pela inatividade
            ↓
    rating_before efetivo
            ↓
    expected_score
            ↓
    resultado da luta
            ↓
    rating_after

A recência NÃO altera retroativamente o rating histórico.
Ela apenas reduz a distância do rating em relação ao rating
inicial antes de uma nova luta, quando houve um período de
inatividade.

Fórmula:

    R_efetivo = R_inicial + (R_histórico - R_inicial) * fator

    fator = 2 ** (-Δt / H)

onde:

    Δt = dias desde a última luta
    H  = meia-vida em dias

Exemplo:

    rating histórico = 1700
    rating inicial = 1500
    meia-vida = 2 anos

    0 anos -> 1700
    2 anos -> 1600
    4 anos -> 1550
    6 anos -> 1525

O baseline sem recência continua sendo o Elo v0.1 oficial.

Resultados:

    data/evaluation/recency/summary.csv
    data/evaluation/recency/fold_metrics.csv

Ratings:

    data/features/recency/<config>/fighter_ratings.csv
"""

from pathlib import Path

import pandas as pd

from src.build_fighter_ratings import (
    DEFAULT_CONFIG,
    expected_score,
    load_and_merge_fights,
    normalize_division,
    validate_output,
)

from src.evaluate_ratings import (
    HISTORY_FILE,
    build_evaluation_dataset,
    run_simple_split,
    run_walk_forward,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_DIR = ROOT_DIR / "data" / "features"
EVALUATION_DIR = ROOT_DIR / "data" / "evaluation"

RECENCY_FEATURES_DIR = FEATURES_DIR / "recency"
RECENCY_EVALUATION_DIR = EVALUATION_DIR / "recency"


# ---------------------------------------------------------------------
# Configurações do experimento
# ---------------------------------------------------------------------

RECENCY_CONFIGS = {
    "baseline": None,
    "half_life_1y": 1,
    "half_life_2y": 2,
    "half_life_3y": 3,
    "half_life_4y": 4,
    "half_life_5y": 5,
}


# ---------------------------------------------------------------------
# Recência
# ---------------------------------------------------------------------

def apply_recency(
    rating,
    last_fight_date,
    current_date,
    initial_rating,
    half_life_years,
):
    """
    Aplica decaimento exponencial do rating histórico em direção
    ao rating inicial.

    Retorna:

        effective_rating
        days_since_last_fight
        recency_factor

    Para um lutador sem luta anterior, não existe período de
    inatividade. Nesse caso, o rating permanece no valor inicial
    e o fator é 1.0.
    """

    if last_fight_date is None or pd.isna(last_fight_date):
        return (
            rating,
            0,
            1.0,
        )

    elapsed_days = max(
        0,
        (current_date - last_fight_date).days,
    )

    half_life_days = half_life_years * 365.25

    factor = 2 ** (
        -elapsed_days / half_life_days
    )

    effective_rating = (
        initial_rating
        + (rating - initial_rating) * factor
    )

    return (
        effective_rating,
        elapsed_days,
        factor,
    )


# ---------------------------------------------------------------------
# Construção dos ratings com recência
# ---------------------------------------------------------------------

def build_ratings_with_recency(
    fights,
    config=DEFAULT_CONFIG,
    half_life_years=None,
):
    """
    Calcula ratings Elo com ou sem recência.

    half_life_years=None:
        reproduz o comportamento do Elo v0.1 original.

    half_life_years > 0:
        aplica decaimento temporal antes de cada luta.

    Estrutura conceitual:

        rating histórico
            ↓
        recência
            ↓
        rating efetivo antes da luta
            ↓
        Elo update
            ↓
        novo rating histórico
    """

    # -----------------------------------------------------------------
    # Baseline
    #
    # O baseline é delegado diretamente para build_ratings().
    # Isso garante que o Elo v0.1 continue sendo exatamente o mesmo
    # modelo já validado anteriormente.
    # -----------------------------------------------------------------

    if half_life_years is None:

        from src.build_fighter_ratings import build_ratings

        return build_ratings(
            fights,
            config=config,
        )

    if half_life_years <= 0:
        raise ValueError(
            "half_life_years deve ser maior que zero."
        )

    fights = (
        fights
        .sort_values(
            ["date", "fight_id"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # Estado histórico dos lutadores
    # -----------------------------------------------------------------

    historical_rating = {}
    fights_count = {}
    fights_since_switch = {}
    current_division = {}
    last_fight_date = {}

    # -----------------------------------------------------------------
    # Inicialização
    # -----------------------------------------------------------------

    def ensure_known(fighter_id):

        historical_rating.setdefault(
            fighter_id,
            config.initial_rating,
        )

        fights_count.setdefault(
            fighter_id,
            0,
        )

        fights_since_switch.setdefault(
            fighter_id,
            0,
        )

        current_division.setdefault(
            fighter_id,
            None,
        )

        last_fight_date.setdefault(
            fighter_id,
            None,
        )

    # -----------------------------------------------------------------
    # Experiência / provisional
    # -----------------------------------------------------------------

    def is_provisional(fighter_id):

        return (
            fights_count[fighter_id]
            < config.provisional_fights
            or
            fights_since_switch[fighter_id]
            < config.provisional_fights
        )

    def k_factor(fighter_id):

        if is_provisional(fighter_id):
            return config.boosted_k

        return config.base_k

    rows = []

    # -----------------------------------------------------------------
    # Processamento cronológico
    # -----------------------------------------------------------------

    for fight in fights.itertuples(index=False):

        fighter_1_id = fight.fighter_1_id
        fighter_2_id = fight.fighter_2_id

        ensure_known(
            fighter_1_id
        )

        ensure_known(
            fighter_2_id
        )

        current_date = pd.Timestamp(
            fight.date
        )

        division = normalize_division(
            fight.weight_class
        )

        # -------------------------------------------------------------
        # Mudança de divisão
        # -------------------------------------------------------------

        division_change = {}

        for fighter_id in (
            fighter_1_id,
            fighter_2_id,
        ):

            previous_division = (
                current_division[fighter_id]
            )

            changed = (
                division is not None
                and previous_division is not None
                and division != previous_division
                and division not in {
                    "Catch Weight",
                    "Open Weight",
                }
                and previous_division not in {
                    "Catch Weight",
                    "Open Weight",
                }
            )

            division_change[fighter_id] = changed

            if changed:

                fights_since_switch[
                    fighter_id
                ] = 0

        # -------------------------------------------------------------
        # Estreia
        # -------------------------------------------------------------

        is_debut = {
            fighter_1_id:
                fights_count[
                    fighter_1_id
                ] == 0,

            fighter_2_id:
                fights_count[
                    fighter_2_id
                ] == 0,
        }

        # -------------------------------------------------------------
        # Rating histórico
        #
        # É o rating resultante da última luta processada.
        # -------------------------------------------------------------

        historical_rating_1 = (
            historical_rating[
                fighter_1_id
            ]
        )

        historical_rating_2 = (
            historical_rating[
                fighter_2_id
            ]
        )

        # -------------------------------------------------------------
        # Aplicação da recência
        # -------------------------------------------------------------

        (
            rating_1_before,
            days_since_1,
            recency_factor_1,
        ) = apply_recency(
            historical_rating_1,
            last_fight_date[
                fighter_1_id
            ],
            current_date,
            config.initial_rating,
            half_life_years,
        )

        (
            rating_2_before,
            days_since_2,
            recency_factor_2,
        ) = apply_recency(
            historical_rating_2,
            last_fight_date[
                fighter_2_id
            ],
            current_date,
            config.initial_rating,
            half_life_years,
        )

        # -------------------------------------------------------------
        # Expectativa pré-luta
        # -------------------------------------------------------------

        expected_1 = expected_score(
            rating_1_before,
            rating_2_before,
            scale=config.scale,
        )

        expected_2 = (
            1.0 - expected_1
        )

        # -------------------------------------------------------------
        # Resultado
        # -------------------------------------------------------------

        result_1 = fight.fighter_1_result
        result_2 = fight.fighter_2_result

        is_no_contest = (
            result_1 == "NC"
            or result_2 == "NC"
        )

        # -------------------------------------------------------------
        # K
        # -------------------------------------------------------------

        k_1 = k_factor(
            fighter_1_id
        )

        k_2 = k_factor(
            fighter_2_id
        )

        # -------------------------------------------------------------
        # Atualização Elo
        # -------------------------------------------------------------

        if is_no_contest:

            rating_1_after = (
                rating_1_before
            )

            rating_2_after = (
                rating_2_before
            )

        else:

            if result_1 == "W":

                actual_1 = 1.0
                actual_2 = 0.0

            elif result_1 == "L":

                actual_1 = 0.0
                actual_2 = 1.0

            elif result_1 == "D":

                actual_1 = 0.5
                actual_2 = 0.5

            else:

                raise ValueError(
                    f"Resultado inesperado na luta "
                    f"{fight.fight_id}: {result_1}."
                )

            rating_1_after = (
                rating_1_before
                + k_1 * (
                    actual_1
                    - expected_1
                )
            )

            rating_2_after = (
                rating_2_before
                + k_2 * (
                    actual_2
                    - expected_2
                )
            )

        # -------------------------------------------------------------
        # Registro do lutador 1
        # -------------------------------------------------------------

        rows.append(
            {
                "fight_id": fight.fight_id,
                "date": fight.date,
                "fighter_id": fighter_1_id,
                "opponent_id": fighter_2_id,
                "division": division,
                "result": result_1,
                "career_fights_before": (
                    fights_count[
                        fighter_1_id
                    ]
                ),
                "is_debut": (
                    is_debut[
                        fighter_1_id
                    ]
                ),
                "is_division_change": (
                    division_change[
                        fighter_1_id
                    ]
                ),
                "k_factor": k_1,
                "rating_updated": (
                    not is_no_contest
                ),
                "rating_historical_before": (
                    historical_rating_1
                ),
                "rating_before": (
                    rating_1_before
                ),
                "rating_after": (
                    rating_1_after
                ),
                "days_since_last_fight": (
                    days_since_1
                ),
                "recency_factor": (
                    recency_factor_1
                ),
                "expected_score": (
                    expected_1
                ),
            }
        )

        # -------------------------------------------------------------
        # Registro do lutador 2
        # -------------------------------------------------------------

        rows.append(
            {
                "fight_id": fight.fight_id,
                "date": fight.date,
                "fighter_id": fighter_2_id,
                "opponent_id": fighter_1_id,
                "division": division,
                "result": result_2,
                "career_fights_before": (
                    fights_count[
                        fighter_2_id
                    ]
                ),
                "is_debut": (
                    is_debut[
                        fighter_2_id
                    ]
                ),
                "is_division_change": (
                    division_change[
                        fighter_2_id
                    ]
                ),
                "k_factor": k_2,
                "rating_updated": (
                    not is_no_contest
                ),
                "rating_historical_before": (
                    historical_rating_2
                ),
                "rating_before": (
                    rating_2_before
                ),
                "rating_after": (
                    rating_2_after
                ),
                "days_since_last_fight": (
                    days_since_2
                ),
                "recency_factor": (
                    recency_factor_2
                ),
                "expected_score": (
                    expected_2
                ),
            }
        )

        # -------------------------------------------------------------
        # Novo rating histórico
        #
        # O decay não é acumulado como se fosse uma atualização
        # permanente. O rating_after desta luta passa a ser o novo
        # rating histórico.
        # -------------------------------------------------------------

        historical_rating[
            fighter_1_id
        ] = rating_1_after

        historical_rating[
            fighter_2_id
        ] = rating_2_after

        # -------------------------------------------------------------
        # Atualização de estado
        # -------------------------------------------------------------

        for fighter_id in (
            fighter_1_id,
            fighter_2_id,
        ):

            fights_count[
                fighter_id
            ] += 1

            fights_since_switch[
                fighter_id
            ] += 1

            last_fight_date[
                fighter_id
            ] = current_date

            if division is not None:

                current_division[
                    fighter_id
                ] = division

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Validação específica da recência
# ---------------------------------------------------------------------

def validate_recency_output(
    ratings,
    fights,
    initial_rating,
    half_life_years,
):
    """
    Validação específica para ratings com recência.

    No Elo tradicional:

        rating_before atual
        =
        rating_after anterior

    No Elo com recência isso NÃO é necessariamente verdade.

    A continuidade correta é:

        rating_after anterior
                ↓
        rating_historical_before atual
                ↓
        aplicação do decay
                ↓
        rating_before atual

    Portanto, esta validação verifica a continuidade histórica e,
    separadamente, a aplicação correta do decaimento.
    """

    errors = []

    expected_rows = len(fights) * 2

    # -----------------------------------------------------------------
    # Número de linhas
    # -----------------------------------------------------------------

    if len(ratings) != expected_rows:

        errors.append(
            "Número de linhas diferente do esperado: "
            f"{len(ratings)} != {expected_rows}."
        )

    # -----------------------------------------------------------------
    # Participações duplicadas
    # -----------------------------------------------------------------

    duplicated = ratings.duplicated(
        subset=[
            "fight_id",
            "fighter_id",
        ]
    ).sum()

    if duplicated > 0:

        errors.append(
            f"Participações duplicadas: {duplicated}."
        )

    # -----------------------------------------------------------------
    # Colunas obrigatórias
    # -----------------------------------------------------------------

    required_columns = [
        "rating_historical_before",
        "rating_before",
        "rating_after",
        "recency_factor",
        "days_since_last_fight",
        "expected_score",
    ]

    for column in required_columns:

        if column not in ratings.columns:

            errors.append(
                f"Coluna obrigatória ausente: {column}."
            )

    # -----------------------------------------------------------------
    # Valores numéricos
    # -----------------------------------------------------------------

    numeric_columns = [
        "rating_historical_before",
        "rating_before",
        "rating_after",
        "recency_factor",
        "days_since_last_fight",
        "expected_score",
    ]

    for column in numeric_columns:

        if column not in ratings.columns:
            continue

        values = pd.to_numeric(
            ratings[column],
            errors="coerce",
        )

        if not values.notna().all():

            errors.append(
                f"Valores inválidos em {column}."
            )

    # -----------------------------------------------------------------
    # Ordenação temporal
    # -----------------------------------------------------------------

    ordered = ratings.copy()

    ordered["date"] = pd.to_datetime(
        ordered["date"]
    )

    ordered = (
        ordered
        .sort_values(
            [
                "fighter_id",
                "date",
                "fight_id",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # Tolerância numérica
    # -----------------------------------------------------------------

    tolerance = 1e-8

    # -----------------------------------------------------------------
    # Validação por lutador
    # -----------------------------------------------------------------

    for fighter_id, group in ordered.groupby(
        "fighter_id",
        sort=False,
    ):

        group = group.reset_index(
            drop=True
        )

        for i, row in group.iterrows():

            # ---------------------------------------------------------
            # Estreia
            # ---------------------------------------------------------

            if row["is_debut"]:

                if not (
                    abs(
                        row[
                            "rating_historical_before"
                        ]
                        - initial_rating
                    )
                    <= tolerance
                ):

                    errors.append(
                        "Rating histórico de estreia "
                        "diferente do inicial: "
                        f"{row['fight_id']} / "
                        f"{fighter_id}."
                    )

                if not (
                    abs(
                        row["rating_before"]
                        - initial_rating
                    )
                    <= tolerance
                ):

                    errors.append(
                        "Rating efetivo de estreia "
                        "diferente do inicial: "
                        f"{row['fight_id']} / "
                        f"{fighter_id}."
                    )

            # ---------------------------------------------------------
            # Continuidade histórica
            #
            # O rating_after anterior deve ser exatamente o rating
            # histórico antes da luta atual.
            # ---------------------------------------------------------

            if i > 0:

                previous = group.iloc[
                    i - 1
                ]

                if not (
                    abs(
                        row[
                            "rating_historical_before"
                        ]
                        - previous[
                            "rating_after"
                        ]
                    )
                    <= tolerance
                ):

                    errors.append(
                        "rating_historical_before "
                        "não corresponde ao rating_after "
                        "anterior: "
                        f"{row['fight_id']} / "
                        f"{fighter_id}."
                    )

            # ---------------------------------------------------------
            # Validação do número de dias
            # ---------------------------------------------------------

            days = row[
                "days_since_last_fight"
            ]

            if days < 0:

                errors.append(
                    "days_since_last_fight negativo: "
                    f"{row['fight_id']} / "
                    f"{fighter_id}."
                )

            if i == 0 and days != 0:

                errors.append(
                    "Luta de estreia possui "
                    "days_since_last_fight diferente de zero: "
                    f"{row['fight_id']} / "
                    f"{fighter_id}."
                )

            # ---------------------------------------------------------
            # Fator de recência
            # ---------------------------------------------------------

            factor = row[
                "recency_factor"
            ]

            expected_factor = (
                2 ** (
                    -days
                    / (
                        half_life_years
                        * 365.25
                    )
                )
                if days > 0
                else 1.0
            )

            if not (
                abs(
                    factor
                    - expected_factor
                )
                <= tolerance
            ):

                errors.append(
                    "recency_factor inválido: "
                    f"{row['fight_id']} / "
                    f"{fighter_id}."
                )

            # ---------------------------------------------------------
            # Rating efetivo após recência
            # ---------------------------------------------------------

            historical = row[
                "rating_historical_before"
            ]

            expected_rating_before = (
                initial_rating
                + (
                    historical
                    - initial_rating
                )
                * expected_factor
            )

            if not (
                abs(
                    row["rating_before"]
                    - expected_rating_before
                )
                <= tolerance
            ):

                errors.append(
                    "rating_before não corresponde "
                    "ao decay aplicado: "
                    f"{row['fight_id']} / "
                    f"{fighter_id}."
                )

            # ---------------------------------------------------------
            # Rating efetivo não pode estar além do rating histórico
            # em relação ao rating inicial.
            #
            # Isso verifica a direção do decay.
            # ---------------------------------------------------------

            historical_distance = abs(
                historical
                - initial_rating
            )

            effective_distance = abs(
                row["rating_before"]
                - initial_rating
            )

            if (
                effective_distance
                > historical_distance
                + tolerance
            ):

                errors.append(
                    "Decay aumentou a distância "
                    "em relação ao rating inicial: "
                    f"{row['fight_id']} / "
                    f"{fighter_id}."
                )

            # ---------------------------------------------------------
            # NC
            #
            # O NC não atualiza o rating histórico.
            # Como o rating histórico antes da luta é transformado
            # em rating efetivo pelo decay, o rating_after deve ser
            # igual ao rating_before.
            # ---------------------------------------------------------

            if row["result"] == "NC":

                if not (
                    abs(
                        row["rating_after"]
                        - row["rating_before"]
                    )
                    <= tolerance
                ):

                    errors.append(
                        "No Contest alterou rating: "
                        f"{row['fight_id']} / "
                        f"{fighter_id}."
                    )

    # -----------------------------------------------------------------
    # Resultado
    # -----------------------------------------------------------------

    if errors:

        print(
            "\n=== ERROS DE VALIDAÇÃO RECENCY ==="
        )

        for error in errors[:20]:
            print(
                f"- {error}"
            )

        if len(errors) > 20:

            print(
                f"- ... e mais "
                f"{len(errors) - 20} erros."
            )

        raise ValueError(
            "Ratings com recência inválidos."
        )

    print(
        "\n=== VALIDAÇÃO RECENCY ==="
    )

    print(
        "Linhas esperadas: OK"
    )

    print(
        f"Participações duplicadas: {duplicated}"
    )

    print(
        "Continuidade histórica: OK"
    )

    print(
        "Decay de recência: OK"
    )

    print(
        "Ratings de estreia: OK"
    )

    print(
        "No Contest sem alteração: OK"
    )

    print(
        "Ratings finitos: OK"
    )


# ---------------------------------------------------------------------
# Execução de uma configuração
# ---------------------------------------------------------------------

def run_configuration(
    fights,
    events,
    history,
    config_name,
    half_life_years,
):
    print(
        "\n" + "=" * 70
    )

    print(
        f"CONFIGURAÇÃO: {config_name}"
    )

    if half_life_years is None:

        print(
            "Recência: desativada "
            "(baseline Elo v0.1)"
        )

    else:

        print(
            "Recência: meia-vida de "
            f"{half_life_years} ano(s)"
        )

    # -----------------------------------------------------------------
    # Gera ratings
    # -----------------------------------------------------------------

    ratings = build_ratings_with_recency(
        fights,
        config=DEFAULT_CONFIG,
        half_life_years=half_life_years,
    )

    # -----------------------------------------------------------------
    # Validação
    #
    # Baseline usa a validação original.
    # Recência usa a validação específica, porque rating_before
    # pode ser diferente do rating_after da luta anterior.
    # -----------------------------------------------------------------

    if half_life_years is None:

        validate_output(
            ratings,
            fights,
            initial_rating=(
                DEFAULT_CONFIG.initial_rating
            ),
        )

    else:

        validate_recency_output(
            ratings,
            fights,
            initial_rating=(
                DEFAULT_CONFIG.initial_rating
            ),
            half_life_years=half_life_years,
        )

    # -----------------------------------------------------------------
    # Diretório da configuração
    # -----------------------------------------------------------------

    config_dir = (
        RECENCY_FEATURES_DIR
        / config_name
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Salva ratings
    # -----------------------------------------------------------------

    ratings_file = (
        config_dir
        / "fighter_ratings.csv"
    )

    ratings = (
        ratings
        .sort_values(
            [
                "date",
                "fight_id",
                "fighter_id",
            ]
        )
        .reset_index(drop=True)
    )

    ratings.to_csv(
        ratings_file,
        index=False,
    )

    print(
        f"Ratings salvos em: {ratings_file}"
    )

    # -----------------------------------------------------------------
    # Dataset de avaliação
    #
    # `fights` já contém a coluna date porque veio de
    # load_and_merge_fights().
    #
    # build_evaluation_dataset() adiciona date a partir de events.
    # Para evitar date_x/date_y, removemos date somente da cópia
    # usada na avaliação.
    # -----------------------------------------------------------------

    evaluation_fights = (
        fights
        .drop(
            columns=["date"],
            errors="ignore",
        )
        .copy()
    )

    dataset = build_evaluation_dataset(
        evaluation_fights,
        events,
        ratings,
        history,
        verbose=False,
    )

    # -----------------------------------------------------------------
    # Simple split
    # -----------------------------------------------------------------

    simple_metrics, _ = run_simple_split(
        dataset,
        "2019-01-01",
        verbose=False,
    )

    # -----------------------------------------------------------------
    # Walk-forward
    # -----------------------------------------------------------------

    walk_metrics = run_walk_forward(
        dataset,
        "2011-01-01",
        3,
        verbose=False,
    )

    # -----------------------------------------------------------------
    # Consolidação
    # -----------------------------------------------------------------

    metrics = pd.concat(
        [
            simple_metrics,
            walk_metrics,
        ],
        ignore_index=True,
    )

    # O experimento de recência avalia somente o Elo.
    # run_simple_split() e run_walk_forward() também retornam
    # métricas do baseline Win Rate.
    if "model" not in metrics.columns:
        raise ValueError(
            "Coluna 'model' não encontrada nas métricas de avaliação."
        )

    metrics = metrics[
        metrics["model"] == "elo"
    ].copy()

    if metrics.empty:
        raise ValueError(
            "Nenhuma métrica do modelo Elo foi encontrada."
        )

    metrics.insert(
        0,
        "config",
        config_name,
    )

    metrics["half_life_years"] = (
        half_life_years
    )

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print(
        "=== RECENCY SWEEP — ELO ==="
    )

    # -----------------------------------------------------------------
    # Diretórios
    # -----------------------------------------------------------------

    RECENCY_FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RECENCY_EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Dados
    #
    # load_and_merge_fights() adiciona a data dos eventos e garante
    # que o rating tenha a informação temporal necessária.
    # -----------------------------------------------------------------

    print(
        "\nLendo dados..."
    )

    fights = load_and_merge_fights()

    print(
        f"Lutas carregadas: {len(fights)}"
    )

    # -----------------------------------------------------------------
    # Eventos
    # -----------------------------------------------------------------

    events_file = (
        ROOT_DIR
        / "data"
        / "processed"
        / "events.csv"
    )

    events = pd.read_csv(
        events_file
    )

    events["date"] = pd.to_datetime(
        events["date"]
    )

    # -----------------------------------------------------------------
    # Histórico
    # -----------------------------------------------------------------

    history = pd.read_csv(
        HISTORY_FILE
    )

    # -----------------------------------------------------------------
    # Executa todas as configurações
    # -----------------------------------------------------------------

    all_metrics = []

    for (
        config_name,
        half_life_years,
    ) in RECENCY_CONFIGS.items():

        metrics = run_configuration(
            fights=fights,
            events=events,
            history=history,
            config_name=config_name,
            half_life_years=half_life_years,
        )

        all_metrics.append(
            metrics
        )

    # -----------------------------------------------------------------
    # Consolidação final
    # -----------------------------------------------------------------

    results = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Salva métricas por fold
    # -----------------------------------------------------------------

    fold_metrics_file = (
        RECENCY_EVALUATION_DIR
        / "fold_metrics.csv"
    )

    results.to_csv(
        fold_metrics_file,
        index=False,
    )

    # -----------------------------------------------------------------
    # Resumo pooled
    # -----------------------------------------------------------------

    pooled = results[
        results["fold"] == "pooled test"
    ].copy()

    summary = (
        pooled[
            [
                "config",
                "half_life_years",
                "log_loss",
                "brier_score",
                "roc_auc",
                "accuracy",
            ]
        ]
        .sort_values(
            "log_loss",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    summary_file = (
        RECENCY_EVALUATION_DIR
        / "summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # -----------------------------------------------------------------
    # Resultado agregado
    # -----------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "=== RESULTADO AGREGADO ==="
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # -----------------------------------------------------------------
    # Melhor configuração por fold
    #
    # Apenas para análise exploratória.
    #
    # Não significa que devemos selecionar uma meia-vida diferente
    # para cada período.
    # -----------------------------------------------------------------

    walk_forward = results[
        ~results["fold"].isin(
            [
                "test (simple split)",
                "pooled test",
            ]
        )
    ].copy()

    if not walk_forward.empty:

        best_indices = (
            walk_forward
            .groupby("fold")["log_loss"]
            .idxmin()
        )

        best_by_fold = (
            walk_forward
            .loc[
                best_indices
            ]
            [
                [
                    "fold",
                    "config",
                    "half_life_years",
                    "log_loss",
                    "brier_score",
                    "roc_auc",
                    "accuracy",
                ]
            ]
            .sort_values(
                "fold"
            )
        )

        print(
            "\n=== MELHOR POR FOLD "
            "(LOG LOSS) ==="
        )

        print(
            best_by_fold.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.4f}",
            )
        )

    # -----------------------------------------------------------------
    # Arquivos gerados
    # -----------------------------------------------------------------

    print(
        "\nArquivos gerados:"
    )

    print(
        f"- {summary_file}"
    )

    print(
        f"- {fold_metrics_file}"
    )

    print(
        f"- {RECENCY_FEATURES_DIR}"
    )

    print(
        "\n=== CONCLUÍDO ==="
    )


if __name__ == "__main__":
    main()