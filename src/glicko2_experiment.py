"""
Experimento Glicko-2 para o Ultimate Fighter Rank.

Objetivo:
- Implementar Glicko-2 usando a biblioteca glicko2==2.1.0.
- Avaliar o modelo com walk-forward temporal.
- Usar exatamente os mesmos períodos temporais do experimento oficial.
- Predizer cada luta antes de atualizar o rating.
- Ignorar NC/empates no alvo e na atualização do rating.
- Preservar a ordem cronológica das lutas.
- Produzir métricas por fold e métricas pooled.

IMPORTANTE:
- Este experimento usa Glicko-2 sem aging por calendário.
- A atualização ocorre luta a luta, em ordem cronológica.
- O rating inicial segue os valores padrão do Glicko-2:
    rating = 1500
    RD = 350
    volatility = 0.06
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import glicko2


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"

FIGHTS_PATH = PROCESSED_DIR / "fights.csv"
EVENTS_PATH = PROCESSED_DIR / "events.csv"

RESULTS_PATH = RESULTS_DIR / "glicko2_temporal_results.csv"
PREDICTIONS_PATH = RESULTS_DIR / "glicko2_predictions.csv"


# Valores padrão do Glicko-2
INITIAL_RATING = 1500.0
INITIAL_RD = 350.0
INITIAL_VOL = 0.06


# Mesmo protocolo temporal usado pelo experimento oficial
WALK_FORWARD_FIRST_TEST_START = "2011-01-01"
WALK_FORWARD_STEP_YEARS = 3


# ============================================================
# ESTADO DO LUTADOR
# ============================================================

@dataclass
class FighterState:
    """
    Estado atual de um lutador no Glicko-2.
    """

    player: glicko2.Player

    @property
    def rating(self) -> float:
        return float(self.player.rating)

    @property
    def rd(self) -> float:
        return float(self.player.rd)

    @property
    def vol(self) -> float:
        return float(self.player.vol)


# ============================================================
# CRIAÇÃO / RECUPERAÇÃO DE LUTADORES
# ============================================================

def get_or_create_player(
    states: dict[str, FighterState],
    fighter_id: str,
) -> FighterState:
    """
    Retorna o estado do lutador.

    Se ainda não existir, cria com os parâmetros iniciais
    padrão do Glicko-2.
    """

    fighter_id = str(fighter_id)

    if fighter_id not in states:
        player = glicko2.Player(
            rating=INITIAL_RATING,
            rd=INITIAL_RD,
            vol=INITIAL_VOL,
        )

        states[fighter_id] = FighterState(player=player)

    return states[fighter_id]


# ============================================================
# PROBABILIDADE GLICKO-2
# ============================================================

def glicko_expected_score(
    rating_a: float,
    rd_a: float,
    rating_b: float,
    rd_b: float,
) -> float:
    """
    Calcula P(A vencer B) segundo a função esperada do Glicko-2.

    Diferentemente do Elo clássico, a fórmula considera também
    o RD do adversário.

    Fórmula:

        E = 1 / (1 + exp(-g(RD_b) * (r_a - r_b) / 173.7178))

    onde:

        g(RD) = 1 / sqrt(1 + 3 * RD² / pi²)
    """

    # Conversão de rating/RD da escala original para a escala
    # interna do Glicko-2.
    q = np.log(10) / 400.0

    # Função g(RD).
    g = 1.0 / np.sqrt(
        1.0 + (3.0 * (q ** 2) * (rd_b ** 2)) / (np.pi ** 2)
    )

    exponent = -g * q * (rating_a - rating_b)

    probability = 1.0 / (1.0 + np.exp(exponent))

    return float(np.clip(probability, 1e-15, 1.0 - 1e-15))


# ============================================================
# RESULTADO DA LUTA
# ============================================================

def get_outcome(
    fighter_1_result: str,
    fighter_2_result: str,
) -> float | None:
    """
    Converte o resultado da luta para o alvo:

        1.0 -> fighter 1 venceu
        0.0 -> fighter 2 venceu
        None -> NC / empate / resultado não utilizável
    """

    result_1 = str(fighter_1_result).strip().upper()
    result_2 = str(fighter_2_result).strip().upper()

    if result_1 == "W" and result_2 == "L":
        return 1.0

    if result_1 == "L" and result_2 == "W":
        return 0.0

    # NC/NC
    if result_1 == "NC" or result_2 == "NC":
        return None

    # Empates
    if result_1 == "D" or result_2 == "D":
        return None

    return None


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def prepare_fights(
    fights: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara fights.csv para o experimento.

    O arquivo de lutas contém:
        fighter_1_id
        fighter_2_id
        fighter_1_result
        fighter_2_result

    A data está em events.csv e é recuperada através de event_id.
    """

    fights = fights.copy()
    events = events.copy()

    # --------------------------------------------------------
    # Normalização dos IDs
    # --------------------------------------------------------

    fights["event_id"] = fights["event_id"].astype(str)
    events["event_id"] = events["event_id"].astype(str)

    # --------------------------------------------------------
    # Recupera a data do evento
    # --------------------------------------------------------

    event_columns = ["event_id", "date"]

    missing_event_columns = [
        column
        for column in event_columns
        if column not in events.columns
    ]

    if missing_event_columns:
        raise ValueError(
            "events.csv não possui as colunas necessárias: "
            f"{missing_event_columns}"
        )

    fights = fights.merge(
        events[event_columns],
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Verifica datas
    # --------------------------------------------------------

    fights["date"] = pd.to_datetime(
        fights["date"],
        errors="coerce",
    )

    missing_dates = int(fights["date"].isna().sum())

    if missing_dates > 0:
        raise ValueError(
            f"{missing_dates} lutas ficaram sem data após o merge com events.csv."
        )

    # --------------------------------------------------------
    # IDs dos lutadores
    # --------------------------------------------------------

    fights["fighter_a_id"] = fights["fighter_1_id"].astype(str)
    fights["fighter_b_id"] = fights["fighter_2_id"].astype(str)

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    fights["outcome"] = [
        get_outcome(result_1, result_2)
        for result_1, result_2 in zip(
            fights["fighter_1_result"],
            fights["fighter_2_result"],
        )
    ]

    # --------------------------------------------------------
    # Ordenação histórica
    #
    # Regra do projeto:
    #   date -> fight_id -> fighter_id
    #
    # Como cada luta aparece uma vez aqui:
    #   date -> fight_id
    # --------------------------------------------------------

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
    Gera os mesmos folds temporais usados pelo experimento oficial.

    Exemplo:

        2011-2014
        2014-2017
        2017-2020
        2020-2023
        2023-2026
        2026-2029
    """

    first_test_start = pd.Timestamp(first_test_start)

    max_date = dataset["date"].max()

    folds = []

    test_start = first_test_start

    while test_start <= max_date:
        test_end = test_start + pd.DateOffset(years=step_years)

        folds.append(
            (
                test_start,
                test_end,
            )
        )

        test_start = test_end

    return folds


# ============================================================
# ATUALIZAÇÃO DE UMA LUTA
# ============================================================

def update_fight(
    states: dict[str, FighterState],
    fighter_a_id: str,
    fighter_b_id: str,
    outcome: float,
) -> None:
    """
    Atualiza os dois lutadores após uma luta.

    É extremamente importante capturar o estado pré-luta
    dos dois jogadores antes de atualizar qualquer um deles.

    Caso contrário, o segundo jogador poderia receber o rating
    já atualizado do primeiro.
    """

    fighter_a_id = str(fighter_a_id)
    fighter_b_id = str(fighter_b_id)

    state_a = get_or_create_player(
        states,
        fighter_a_id,
    )

    state_b = get_or_create_player(
        states,
        fighter_b_id,
    )

    # --------------------------------------------------------
    # Snapshot do estado ANTES da luta
    # --------------------------------------------------------

    rating_a = state_a.rating
    rd_a = state_a.rd

    rating_b = state_b.rating
    rd_b = state_b.rd

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    if float(outcome) == 1.0:
        score_a = 1.0
        score_b = 0.0
    else:
        score_a = 0.0
        score_b = 1.0

    # --------------------------------------------------------
    # Atualiza A
    # --------------------------------------------------------

    state_a.player.update_player(
        [rating_b],
        [rd_b],
        [score_a],
    )

    # --------------------------------------------------------
    # Atualiza B
    #
    # Usa os valores de A capturados ANTES da atualização.
    # --------------------------------------------------------

    state_b.player.update_player(
        [rating_a],
        [rd_a],
        [score_b],
    )


# ============================================================
# CONSTRUÇÃO DO ESTADO HISTÓRICO
# ============================================================

def build_historical_state(
    fights: pd.DataFrame,
) -> dict[str, FighterState]:
    """
    Reproduz todas as lutas históricas até o início do período
    de teste.

    NC e empates não atualizam o rating.
    """

    states: dict[str, FighterState] = {}

    for row in fights.itertuples(index=False):

        fighter_a = str(row.fighter_a_id)
        fighter_b = str(row.fighter_b_id)
        outcome = row.outcome

        # ----------------------------------------------------
        # IMPORTANTE:
        #
        # O pandas transforma valores ausentes em NaN.
        # Portanto:
        #
        #     outcome is None
        #
        # não é suficiente.
        #
        # ----------------------------------------------------

        if pd.isna(outcome):
            continue

        update_fight(
            states=states,
            fighter_a_id=fighter_a,
            fighter_b_id=fighter_b,
            outcome=float(outcome),
        )

    return states


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float]:
    """
    Calcula as métricas usadas no projeto.

    Métricas:
        Log Loss
        Brier
        ROC AUC
        Accuracy
    """

    y_true = pd.Series(y_true).astype(int)
    probabilities = pd.Series(probabilities).astype(float)

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
                (probabilities - y_true) ** 2
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                probabilities >= 0.5,
            )
        ),
    }

    # ROC AUC só é definido quando existem as duas classes.
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
# EXECUÇÃO DE UM FOLD
# ============================================================

def run_fold(
    dataset: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[dict[str, float] | None, pd.DataFrame]:
    """
    Executa um fold temporal.

    Treino:
        todas as lutas antes de test_start.

    Teste:
        [test_start, test_end)

    Para cada luta de teste:

        1. calcula probabilidade com rating pré-luta;
        2. registra previsão;
        3. atualiza Glicko-2 após conhecer o resultado.
    """

    train = dataset[
        dataset["date"] < test_start
    ].copy()

    test = dataset[
        (dataset["date"] >= test_start)
        & (dataset["date"] < test_end)
        & dataset["outcome"].notna()
    ].copy()

    # --------------------------------------------------------
    # Ordenação temporal
    # --------------------------------------------------------

    train = train.sort_values(
        ["date", "fight_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    test = test.sort_values(
        ["date", "fight_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Estado histórico
    # --------------------------------------------------------

    states = build_historical_state(train)

    # --------------------------------------------------------
    # Predições
    # --------------------------------------------------------

    predictions = []

    for row in test.itertuples(index=False):

        fighter_a = str(row.fighter_a_id)
        fighter_b = str(row.fighter_b_id)

        outcome = float(row.outcome)

        # Cria lutadores estreantes caso necessário.
        state_a = get_or_create_player(
            states,
            fighter_a,
        )

        state_b = get_or_create_player(
            states,
            fighter_b,
        )

        # ----------------------------------------------------
        # Snapshot pré-luta
        # ----------------------------------------------------

        rating_a = state_a.rating
        rd_a = state_a.rd

        rating_b = state_b.rating
        rd_b = state_b.rd

        # ----------------------------------------------------
        # Probabilidade ANTES da atualização
        # ----------------------------------------------------

        probability_a = glicko_expected_score(
            rating_a=rating_a,
            rd_a=rd_a,
            rating_b=rating_b,
            rd_b=rd_b,
        )

        predictions.append(
            {
                "fight_id": row.fight_id,
                "date": row.date,
                "fighter_a_id": fighter_a,
                "fighter_b_id": fighter_b,
                "y": outcome,
                "probability_a": probability_a,
                "rating_a_before": rating_a,
                "rd_a_before": rd_a,
                "vol_a_before": state_a.vol,
                "rating_b_before": rating_b,
                "rd_b_before": rd_b,
                "vol_b_before": state_b.vol,
            }
        )

        # ----------------------------------------------------
        # Atualiza SOMENTE depois da previsão
        # ----------------------------------------------------

        update_fight(
            states=states,
            fighter_a_id=fighter_a,
            fighter_b_id=fighter_b,
            outcome=outcome,
        )

    # --------------------------------------------------------
    # DataFrame de previsões
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(predictions)

    if predictions_df.empty:
        return None, predictions_df

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    metrics = calculate_metrics(
        predictions_df["y"],
        predictions_df["probability_a"],
    )

    metrics["n_fights"] = int(len(predictions_df))

    return metrics, predictions_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("Glicko-2 — Ultimate Fighter Rank")
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

    print(f"\nCarregando: {FIGHTS_PATH}")
    fights = pd.read_csv(FIGHTS_PATH)

    print(f"Carregando: {EVENTS_PATH}")
    events = pd.read_csv(EVENTS_PATH)

    print(f"\nLutas carregadas: {len(fights)}")
    print(f"Eventos carregados: {len(events)}")

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
        f"Lutas W/L válidas: {len(valid_fights)}"
    )

    print(
        f"Lutas NC/empate ignoradas: {len(ignored_fights)}"
    )

    # --------------------------------------------------------
    # Folds
    # --------------------------------------------------------

    folds = generate_walk_forward_folds(
        dataset=dataset,
        first_test_start=WALK_FORWARD_FIRST_TEST_START,
        step_years=WALK_FORWARD_STEP_YEARS,
    )

    print(
        f"\nFolds temporais: {len(folds)}"
    )

    print(
        f"Primeiro teste: {WALK_FORWARD_FIRST_TEST_START}"
    )

    print(
        f"Tamanho do fold: {WALK_FORWARD_STEP_YEARS} anos"
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
            f"{test_start.year}-{test_end.year}"
        )

        print(
            f"Teste: "
            f"{test_start.date()} -> {test_end.date()}"
        )

        metrics, predictions = run_fold(
            dataset=dataset,
            test_start=test_start,
            test_end=test_end,
        )

        if metrics is None:
            print("Sem lutas válidas neste fold.")
            continue

        # ----------------------------------------------------
        # Identificação do fold
        # ----------------------------------------------------

        metrics["fold_start"] = test_start.date().isoformat()
        metrics["fold_end"] = test_end.date().isoformat()

        # Reorganiza para facilitar leitura.
        metrics = {
            "fold_start": metrics.pop("fold_start"),
            "fold_end": metrics.pop("fold_end"),
            "n_fights": metrics.pop("n_fights"),
            "log_loss": metrics.pop("log_loss"),
            "brier": metrics.pop("brier"),
            "auc": metrics.pop("auc"),
            "accuracy": metrics.pop("accuracy"),
        }

        fold_results.append(metrics)

        predictions = predictions.copy()

        predictions["fold_start"] = (
            test_start.date().isoformat()
        )

        predictions["fold_end"] = (
            test_end.date().isoformat()
        )

        all_predictions.append(predictions)

        # ----------------------------------------------------
        # Exibição
        # ----------------------------------------------------

        print(
            f"Lutas: {metrics['n_fights']}"
        )

        print(
            f"Log Loss: {metrics['log_loss']:.4f}"
        )

        print(
            f"Brier:    {metrics['brier']:.4f}"
        )

        print(
            f"AUC:      {metrics['auc']:.4f}"
        )

        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

    # --------------------------------------------------------
    # Verificação
    # --------------------------------------------------------

    if not fold_results:
        raise RuntimeError(
            "Nenhum fold produziu resultados."
        )

    # --------------------------------------------------------
    # DataFrame dos folds
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

    pooled_metrics = calculate_metrics(
        predictions_df["y"],
        predictions_df["probability_a"],
    )

    pooled_row = {
        "fold_start": "POOLED",
        "fold_end": "POOLED",
        "n_fights": len(predictions_df),
        "log_loss": pooled_metrics["log_loss"],
        "brier": pooled_metrics["brier"],
        "auc": pooled_metrics["auc"],
        "accuracy": pooled_metrics["accuracy"],
    }

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame([pooled_row]),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Cria diretório
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Salva resultados
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
    print("RESULTADO POOLED — GLICKO-2")
    print("=" * 70)

    print(
        f"Lutas:    {pooled_row['n_fights']}"
    )

    print(
        f"Log Loss: {pooled_row['log_loss']:.4f}"
    )

    print(
        f"Brier:    {pooled_row['brier']:.4f}"
    )

    print(
        f"AUC:      {pooled_row['auc']:.4f}"
    )

    print(
        f"Accuracy: {pooled_row['accuracy']:.4f}"
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