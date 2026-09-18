"""
Avaliação temporal: Win Rate baseline vs. Elo v0.1.

Ver:
- docs/rating_methodology.md, seções 23, 24, 25.
- docs/development_guideline.md, seções 27, 28, 29, 53.

Este script NÃO treina nenhum modelo com parâmetros livres. Win
Rate e Elo v0.1 são baselines determinísticos (o "aprendizado" do
Elo já está embutido no rating acumulado ao longo do tempo, luta a
luta, em `fighter_ratings.csv`). Por isso, aqui "treino" não
significa "ajustar parâmetros": significa apenas "período usado
como aquecimento/contexto histórico", e "teste" é a janela onde as
métricas de fato são calculadas. Essa distinção passa a importar de
verdade quando modelos com parâmetros livres (ex.: Logistic
Regression sobre várias features) entrarem na comparação — a
estrutura de fold já fica pronta para isso.

Decisões metodológicas desta primeira versão (registradas aqui, não
arbitrárias — seção 26 do guideline):

1. Win Rate -> probabilidade de A vencer B:

       P(A vence) = win_rate_A / (win_rate_A + win_rate_B)

   Estreantes (win_rate_before ausente, 0 lutas) e o caso
   degenerado win_rate_A + win_rate_B == 0 (os dois com 0% de
   aproveitamento) recebem probabilidade neutra (0.5). Isso é
   análogo à forma como o Elo trata estreantes: rating inicial
   igual para todo mundo (ver INITIAL_RATING em
   build_fighter_ratings.py).

2. Elo -> probabilidade de A vencer B:

   A coluna `expected_score` de `fighter_ratings.csv` já É a
   probabilidade do Elo antes da luta, calculada a partir de
   `rating_before` (ver expected_score() em
   build_fighter_ratings.py). Não recalculamos nada: apenas
   reaproveitamos esse valor, do ponto de vista do fighter_1 de
   cada luta.

3. Lutas excluídas da avaliação:

   - No Contest: não representam vitória/derrota de ninguém
     (mesma lógica que já vale para o Elo não atualizar rating
     nesses casos — coluna `rating_updated`).
   - Empate (Draw): não existe "vencedor" para Accuracy/AUC, e
     tratar empate como y=0.5 misturaria dois tipos de incerteza
     diferentes (incerteza da previsão vs. resultado
     objetivamente sem vencedor) nas métricas de Log
     Loss/Brier/Calibration.

   Essas exclusões usam o resultado de fights.csv
   (fighter_1_result) e a flag `rating_updated` do Elo, que já
   captura NC.

4. Ponto de vista da observação:

   Cada luta gera exatamente uma observação, do ponto de vista do
   fighter_1 (perspectiva de fights.csv). y = 1 se fighter_1
   venceu, y = 0 se fighter_2 venceu.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"
EVALUATION_DIR = ROOT_DIR / "data" / "evaluation"

FIGHTS_FILE = PROCESSED_DIR / "fights.csv"
EVENTS_FILE = PROCESSED_DIR / "events.csv"
RATINGS_FILE = FEATURES_DIR / "fighter_ratings.csv"
HISTORY_FILE = FEATURES_DIR / "fighter_history.csv"

DATASET_OUTPUT_FILE = EVALUATION_DIR / "evaluation_dataset.csv"
METRICS_OUTPUT_FILE = EVALUATION_DIR / "metrics_summary.csv"
CALIBRATION_OUTPUT_FILE = EVALUATION_DIR / "calibration_tables.csv"

# Split temporal simples, para validar o harness rapidamente
# (seção 53 do development_guideline.md usa este mesmo exemplo).
SIMPLE_SPLIT_CUTOFF = "2019-01-01"

# Walk-forward: primeira janela de teste e tamanho de cada janela
# seguinte, reproduzindo o exemplo da seção 23 do
# rating_methodology.md (treino 1994->2010, teste 2011-2013;
# depois treino 1994->2013, teste 2014-2016; ...).
WALK_FORWARD_FIRST_TEST_START = "2011-01-01"
WALK_FORWARD_STEP_YEARS = 3

MODELS = ["win_rate", "elo"]
PROB_COLUMNS = {"win_rate": "p_win_rate", "elo": "p_elo"}


# ============================================================
# Construção do dataset de avaliação
# ============================================================


def win_rate_probability(win_rate_a, win_rate_b):
    """
    Converte dois win rates em P(A vence), pela normalização

        win_rate_A / (win_rate_A + win_rate_B)

    Retorna 0.5 quando qualquer um dos dois é ausente (estreante)
    ou quando a soma dos dois é zero (ambos com 0% de
    aproveitamento). Ver decisão 1 no docstring do módulo.
    """

    if pd.isna(win_rate_a) or pd.isna(win_rate_b):
        return 0.5

    denominator = win_rate_a + win_rate_b

    if denominator == 0:
        return 0.5

    return win_rate_a / denominator


def load_inputs(ratings_file=RATINGS_FILE, history_file=HISTORY_FILE):
    """
    `ratings_file`/`history_file` são parametrizáveis para que
    `src/param_sweep.py` possa reaproveitar este harness apontando
    para um `fighter_ratings.csv` alternativo (uma config de Elo
    diferente) sem duplicar a lógica de avaliação.
    """

    fights = pd.read_csv(FIGHTS_FILE)
    events = pd.read_csv(EVENTS_FILE)
    ratings = pd.read_csv(ratings_file)
    history = pd.read_csv(history_file)

    events["date"] = pd.to_datetime(events["date"])
    ratings["date"] = pd.to_datetime(ratings["date"])

    return fights, events, ratings, history


def build_evaluation_dataset(fights, events, ratings, history, verbose=True):
    """
    Monta um dataset com uma linha por luta (do ponto de vista do
    fighter_1), contendo:

    - y: 1 se fighter_1 venceu, 0 se fighter_2 venceu
    - p_win_rate: P(fighter_1 vence) segundo o baseline Win Rate
    - p_elo: P(fighter_1 vence) segundo o Elo v0.1
    - date: data do evento

    Lutas com Draw ou No Contest são excluídas (decisão 3 do
    docstring do módulo).
    """

    event_dates = events[["event_id", "date"]].drop_duplicates(
        subset=["event_id"]
    )

    fights = fights.merge(
        event_dates, on="event_id", how="left", validate="many_to_one"
    )

    # --- Elo: reaproveita expected_score já calculado por
    # build_fighter_ratings.py, do ponto de vista do fighter_1.
    elo_fighter_1 = ratings.rename(
        columns={
            "fighter_id": "fighter_1_id",
            "expected_score": "p_elo",
            "rating_before": "elo_rating_before_1",
        }
    )[["fight_id", "fighter_1_id", "p_elo", "elo_rating_before_1", "rating_updated"]]

    dataset = fights.merge(
        elo_fighter_1,
        on=["fight_id", "fighter_1_id"],
        how="left",
        validate="one_to_one",
    )

    # --- Win Rate: win_rate_before de cada lutador na luta.
    history_slim = history[["fight_id", "fighter_id", "win_rate_before"]]

    history_1 = history_slim.rename(
        columns={
            "fighter_id": "fighter_1_id",
            "win_rate_before": "win_rate_before_1",
        }
    )
    history_2 = history_slim.rename(
        columns={
            "fighter_id": "fighter_2_id",
            "win_rate_before": "win_rate_before_2",
        }
    )

    dataset = dataset.merge(
        history_1, on=["fight_id", "fighter_1_id"], how="left", validate="one_to_one"
    )
    dataset = dataset.merge(
        history_2, on=["fight_id", "fighter_2_id"], how="left", validate="one_to_one"
    )

    dataset["p_win_rate"] = dataset.apply(
        lambda row: win_rate_probability(
            row["win_rate_before_1"], row["win_rate_before_2"]
        ),
        axis=1,
    )

    # --- Rótulo (y) e exclusões.
    is_draw = dataset["fighter_1_result"] == "D"
    is_no_contest = ~dataset["rating_updated"].astype(bool)

    excluded = is_draw | is_no_contest

    dataset["y"] = np.select(
        [dataset["fighter_1_result"] == "W", dataset["fighter_1_result"] == "L"],
        [1, 0],
        default=np.nan,
    )

    n_total = len(dataset)
    n_draw = int(is_draw.sum())
    n_no_contest = int(is_no_contest.sum())

    dataset = dataset.loc[~excluded].copy()

    missing_elo = dataset["p_elo"].isna().sum()

    if missing_elo:
        raise ValueError(
            f"{missing_elo} lutas sem p_elo após o merge com "
            "fighter_ratings.csv. Verifique se os dois arquivos "
            "estão sincronizados (mesmo fights.csv de origem)."
        )

    if verbose:
        print("\n=== CONSTRUÇÃO DO DATASET DE AVALIAÇÃO ===")
        print(f"Lutas totais em fights.csv: {n_total}")
        print(f"  Excluídas por Draw: {n_draw}")
        print(f"  Excluídas por No Contest: {n_no_contest}")
        print(f"Lutas avaliáveis: {len(dataset)}")
        print(
            f"  Estreantes/degenerados no Win Rate (p=0.5 por regra): "
            f"{(dataset['win_rate_before_1'].isna() | dataset['win_rate_before_2'].isna()).sum()}"
        )

    return dataset[
        [
            "fight_id",
            "date",
            "fighter_1_id",
            "fighter_2_id",
            "y",
            "p_win_rate",
            "p_elo",
        ]
    ].reset_index(drop=True)


# ============================================================
# Métricas
# ============================================================


def compute_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)

    n = len(y_true)
    n_classes = len(np.unique(y_true))

    metrics = {
        "n": n,
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, (y_prob >= 0.5).astype(int)),
    }

    if n_classes < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


def calibration_table(y_true, y_prob, n_bins=10):
    """
    Agrupa as previsões em bins de probabilidade e compara, em
    cada bin, a probabilidade média prevista com a taxa de vitória
    observada. Um modelo bem calibrado tem as duas colunas
    próximas (seção 24 do rating_methodology.md).
    """

    frame = pd.DataFrame({"y": y_true, "p": y_prob})

    bins = np.linspace(0, 1, n_bins + 1)
    frame["bin"] = pd.cut(frame["p"], bins=bins, include_lowest=True)

    table = frame.groupby("bin", observed=True).agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_win_rate=("y", "mean"),
    )

    table["gap"] = table["mean_predicted"] - table["observed_win_rate"]

    return table.reset_index()


def print_metrics_table(rows):
    header = f"{'modelo':<12}{'fold':<20}{'n':>8}{'log_loss':>12}{'brier':>10}{'auc':>10}{'accuracy':>10}"
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['model']:<12}{row['fold']:<20}{row['n']:>8}"
            f"{row['log_loss']:>12.4f}{row['brier_score']:>10.4f}"
            f"{row['roc_auc']:>10.4f}{row['accuracy']:>10.4f}"
        )


# ============================================================
# Avaliação temporal
# ============================================================


def run_simple_split(dataset, cutoff_date, verbose=True):
    cutoff = pd.Timestamp(cutoff_date)

    train = dataset[dataset["date"] < cutoff]
    test = dataset[dataset["date"] >= cutoff]

    if verbose:
        print("\n=== SPLIT TEMPORAL SIMPLES ===")
        print(f"Corte: {cutoff.date()}")
        print(
            f"Treino (contexto histórico, não há fit): {len(train)} lutas "
            f"({train['date'].min().date()} a {train['date'].max().date()})"
        )
        print(
            f"Teste: {len(test)} lutas "
            f"({test['date'].min().date()} a {test['date'].max().date()})"
        )

    rows = []

    for model in MODELS:
        prob_col = PROB_COLUMNS[model]
        metrics = compute_metrics(test["y"], test[prob_col])
        rows.append({"model": model, "fold": "test (simple split)", **metrics})

    if verbose:
        print()
        print_metrics_table(rows)

    return pd.DataFrame(rows), test


def generate_walk_forward_folds(dataset, first_test_start, step_years):
    first_test_start = pd.Timestamp(first_test_start)
    max_date = dataset["date"].max()

    folds = []
    test_start = first_test_start

    while test_start <= max_date:
        test_end = test_start + pd.DateOffset(years=step_years)
        folds.append((test_start, test_end))
        test_start = test_end

    return folds


def run_walk_forward(dataset, first_test_start, step_years, verbose=True):
    folds = generate_walk_forward_folds(dataset, first_test_start, step_years)

    if verbose:
        print("\n=== WALK-FORWARD (JANELAS EXPANSIVAS) ===")

    rows = []
    pooled_test = {model: [] for model in MODELS}

    for test_start, test_end in folds:
        train = dataset[dataset["date"] < test_start]
        test = dataset[
            (dataset["date"] >= test_start) & (dataset["date"] < test_end)
        ]

        if len(test) == 0:
            continue

        fold_label = f"{test_start.date()}→{test_end.date()}"

        if verbose:
            print(
                f"\nFold {fold_label} | treino (histórico até aqui): "
                f"{len(train)} lutas | teste: {len(test)} lutas"
            )

        for model in MODELS:
            prob_col = PROB_COLUMNS[model]
            metrics = compute_metrics(test["y"], test[prob_col])
            rows.append({"model": model, "fold": fold_label, **metrics})
            pooled_test[model].append(test[["y", prob_col]])

    if verbose:
        print()
        print_metrics_table(rows)
        print("\n--- Agregado (todas as janelas de teste, sem sobreposição) ---")

    pooled_rows = []

    for model in MODELS:
        prob_col = PROB_COLUMNS[model]
        pooled = pd.concat(pooled_test[model], ignore_index=True)
        metrics = compute_metrics(pooled["y"], pooled[prob_col])
        pooled_rows.append({"model": model, "fold": "pooled test", **metrics})

    if verbose:
        print_metrics_table(pooled_rows)

    return pd.DataFrame(rows + pooled_rows)


def evaluate_elo_config(fights, events, history, ratings_file, verbose=False):
    """
    Reaproveita o harness de avaliação (dataset + split simples +
    walk-forward) para um `fighter_ratings.csv` alternativo, sem
    duplicar a lógica de `main()`. Usado por `src/param_sweep.py`
    para comparar configurações de Elo (seção 53, item 5 do
    development_guideline.md).

    `fights`/`events`/`history` são recebidos já carregados (o
    sweep os lê uma única vez, já que não mudam entre configs);
    apenas os ratings mudam por config.
    """

    ratings = pd.read_csv(ratings_file)
    ratings["date"] = pd.to_datetime(ratings["date"])

    dataset = build_evaluation_dataset(fights, events, ratings, history, verbose=verbose)

    simple_split_metrics, _ = run_simple_split(
        dataset, SIMPLE_SPLIT_CUTOFF, verbose=verbose
    )
    walk_forward_metrics = run_walk_forward(
        dataset, WALK_FORWARD_FIRST_TEST_START, WALK_FORWARD_STEP_YEARS, verbose=verbose
    )

    return pd.concat([simple_split_metrics, walk_forward_metrics], ignore_index=True)


def main():
    print("=== EVALUATE RATINGS: Win Rate vs. Elo v0.1 ===")

    fights, events, ratings, history = load_inputs()
    dataset = build_evaluation_dataset(fights, events, ratings, history)

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_OUTPUT_FILE, index=False)
    print(f"\nDataset de avaliação salvo em: {DATASET_OUTPUT_FILE}")

    simple_split_metrics, simple_split_test = run_simple_split(
        dataset, SIMPLE_SPLIT_CUTOFF
    )

    walk_forward_metrics = run_walk_forward(
        dataset, WALK_FORWARD_FIRST_TEST_START, WALK_FORWARD_STEP_YEARS
    )

    all_metrics = pd.concat(
        [simple_split_metrics, walk_forward_metrics], ignore_index=True
    )
    all_metrics.to_csv(METRICS_OUTPUT_FILE, index=False)
    print(f"\nResumo de métricas salvo em: {METRICS_OUTPUT_FILE}")

    print("\n=== CALIBRATION (split simples, conjunto de teste) ===")

    calibration_rows = []

    for model in MODELS:
        prob_col = PROB_COLUMNS[model]
        table = calibration_table(simple_split_test["y"], simple_split_test[prob_col])
        table.insert(0, "model", model)
        calibration_rows.append(table)
        print(f"\n-- {model} --")
        print(table.to_string(index=False))

    calibration_df = pd.concat(calibration_rows, ignore_index=True)
    calibration_df.to_csv(CALIBRATION_OUTPUT_FILE, index=False)
    print(f"\nTabelas de calibração salvas em: {CALIBRATION_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
