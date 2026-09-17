from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT_DIR / "data" / "features"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

INPUT_FILE = FEATURES_DIR / "fighter_fight_features.csv"
EVENTS_FILE = PROCESSED_DIR / "events.csv"
OUTPUT_FILE = FEATURES_DIR / "fighter_history.csv"


# Métricas que serão acumuladas historicamente.
HISTORICAL_METRICS = [
    "significant_strikes_landed",
    "significant_strikes_attempted",
    "total_strikes_landed",
    "total_strikes_attempted",
    "takedowns_landed",
    "takedowns_attempted",
    "knockdowns",
    "submission_attempts",
    "reversals",
    "control_time_seconds",
]


def validate_input(df):
    errors = []

    required = [
        "fight_id",
        "event_id",
        "fighter_id",
        "opponent_id",
        "result",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        errors.append(
            f"Colunas obrigatórias ausentes: {missing}"
        )

    if errors:
        for error in errors:
            print(f"- {error}")

        raise ValueError("Entrada inválida.")


def build_history(df):
    """
    Constrói features pré-luta.

    IMPORTANTE:
    As features históricas de uma luta são calculadas
    antes de incorporar os dados dessa própria luta.
    """

    df = df.copy()

    # Garantimos que cada luta seja processada na ordem correta.
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "fight_id", "fighter_id"]
    ).reset_index(drop=True)

    # Contadores históricos.
    df["career_fights_before"] = (
        df.groupby("fighter_id").cumcount()
    )

    df["wins_before"] = (
        df.groupby("fighter_id")["result"]
        .transform(lambda x: (x.shift(1) == "W").cumsum())
    )

    df["losses_before"] = (
        df.groupby("fighter_id")["result"]
        .transform(lambda x: (x.shift(1) == "L").cumsum())
    )

    df["draws_before"] = (
        df.groupby("fighter_id")["result"]
        .transform(lambda x: (x.shift(1) == "D").cumsum())
    )

    df["nc_before"] = (
        df.groupby("fighter_id")["result"]
        .transform(lambda x: (x.shift(1) == "NC").cumsum())
    )

    df["win_rate_before"] = (
        df["wins_before"]
        / df["career_fights_before"].replace(0, pd.NA)
    )

    # Médias históricas das métricas de performance.
    for metric in HISTORICAL_METRICS:
        historical_sum = (
            df.groupby("fighter_id")[metric]
            .transform(
                lambda x: x.shift(1).expanding().sum()
            )
        )

        historical_count = (
            df.groupby("fighter_id")[metric]
            .transform(
                lambda x: x.shift(1).expanding().count()
            )
        )

        df[f"sum_{metric}_before"] = historical_sum

        df[f"avg_{metric}_before"] = (
            historical_sum
            / historical_count.replace(0, pd.NA)
        )

    # Média histórica de eficiência.
    for metric in [
        "significant_strikes_pct",
        "takedown_pct",
    ]:
        historical_sum = (
            df.groupby("fighter_id")[metric]
            .transform(
                lambda x: x.shift(1).expanding().sum()
            )
        )

        historical_count = (
            df.groupby("fighter_id")[metric]
            .transform(
                lambda x: x.shift(1).expanding().count()
            )
        )

        df[f"avg_{metric}_before"] = (
            historical_sum
            / historical_count.replace(0, pd.NA)
        )

    return df


def validate_history(df, original):
    errors = []

    # Cada linha original deve continuar existindo.
    if len(df) != len(original):
        errors.append(
            f"Quantidade de linhas alterada: "
            f"{len(df)} vs {len(original)}."
        )

    # Nenhuma participação pode aparecer duas vezes.
    duplicates = df.duplicated(
        subset=["fight_id", "fighter_id"],
        keep=False,
    )

    if duplicates.any():
        errors.append(
            f"Participações duplicadas: "
            f"{duplicates.sum()}."
        )

    # A primeira luta de cada lutador obrigatoriamente
    # possui zero lutas anteriores.
    first_fights = df["career_fights_before"] == 0

    invalid_first = (
        df.loc[first_fights, "wins_before"] != 0
    ) | (
        df.loc[first_fights, "losses_before"] != 0
    ) | (
        df.loc[first_fights, "draws_before"] != 0
    ) | (
        df.loc[first_fights, "nc_before"] != 0
    )

    if invalid_first.any():
        errors.append(
            "Primeiras lutas possuem histórico anterior."
        )

    # O histórico nunca pode ultrapassar o número
    # de lutas anteriores.
    invalid_career = (
        df["wins_before"]
        + df["losses_before"]
        + df["draws_before"]
        + df["nc_before"]
        != df["career_fights_before"]
    )

    if invalid_career.any():
        errors.append(
            "Contadores históricos não fecham."
        )

    # Win rate precisa estar entre 0 e 1.
    invalid_rate = (
        df["win_rate_before"].notna()
        & (
            (df["win_rate_before"] < 0)
            | (df["win_rate_before"] > 1)
        )
    )

    if invalid_rate.any():
        errors.append(
            f"Win rates inválidos: {invalid_rate.sum()}."
        )

    # Para qualquer lutador, o histórico deve ser
    # não decrescente ao longo da carreira.
    history_order = df.sort_values(
        ["fighter_id", "date", "fight_id"]
    )

    previous = history_order.groupby(
        "fighter_id"
    )["career_fights_before"].shift(1)

    invalid_order = (
        previous.notna()
        & (
            history_order["career_fights_before"]
            <= previous
        )
    )

    if invalid_order.any():
        errors.append(
            f"Histórico cronológico inválido: "
            f"{invalid_order.sum()}."
        )

    if errors:
        print("\n=== ERROS DE VALIDAÇÃO ===")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "Histórico inválido. Arquivo não será salvo."
        )

    print("\n=== VALIDAÇÃO ===")
    print("Linhas preservadas: OK")
    print("Participações duplicadas: 0")
    print("Primeiras lutas sem histórico: OK")
    print("Contadores históricos: OK")
    print("Win rate: OK")
    print("Ordem cronológica: OK")


def main():
    print("=== BUILD FIGHTER HISTORY ===")

    print("\nLendo fighter_fight_features.csv...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Participações carregadas: {len(df)}")

    validate_input(df)

    print("\nLendo events.csv...")

    events = pd.read_csv(EVENTS_FILE)

    events["date"] = pd.to_datetime(events["date"])

    print(f"Eventos carregados: {len(events)}")

    # Só precisamos da data para ordenar a carreira.
    event_dates = events[["event_id", "date"]].drop_duplicates(
        subset=["event_id"]
    )

    df = df.merge(
        event_dates,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    missing_dates = df["date"].isna().sum()

    if missing_dates:
        print(
            f"\nAviso: {missing_dates} participações "
            "não possuem data de evento."
        )

    print("\nConstruindo histórico pré-luta...")

    df = build_history(df)

    validate_history(
        df,
        pd.read_csv(INPUT_FILE),
    )

    # Mantemos a ordem cronológica final.
    df = df.sort_values(
        ["date", "fight_id", "fighter_id"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nArquivo salvo em:")
    print(OUTPUT_FILE)

    print("\n=== RESUMO ===")

    print(
        f"Participações:          {len(df)}"
    )

    print(
        f"Lutadores:              "
        f"{df['fighter_id'].nunique()}"
    )

    print(
        f"Estreias:               "
        f"{(df['career_fights_before'] == 0).sum()}"
    )

    print(
        f"Participações com "
        f"histórico:              "
        f"{(df['career_fights_before'] > 0).sum()}"
    )

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()