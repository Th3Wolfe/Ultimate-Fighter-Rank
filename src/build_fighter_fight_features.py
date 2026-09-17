from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT_DIR / "data" / "features"

INPUT_FILE = FEATURES_DIR / "fight_features.csv"
OUTPUT_FILE = FEATURES_DIR / "fighter_fight_features.csv"


STAT_COLUMNS = [
    "significant_strikes_landed",
    "significant_strikes_attempted",
    "total_strikes_landed",
    "total_strikes_attempted",
    "takedowns_landed",
    "takedowns_attempted",
    "head_strikes_landed",
    "head_strikes_attempted",
    "body_strikes_landed",
    "body_strikes_attempted",
    "leg_strikes_landed",
    "leg_strikes_attempted",
    "distance_strikes_landed",
    "distance_strikes_attempted",
    "clinch_strikes_landed",
    "clinch_strikes_attempted",
    "ground_strikes_landed",
    "ground_strikes_attempted",
    "knockdowns",
    "submission_attempts",
    "reversals",
    "control_time_seconds",
    "rounds_with_stats",
    "significant_strikes_pct",
    "takedown_pct",
]


def parse_time_to_seconds(value):
    """
    Converte tempos no formato M:SS para segundos.
    Exemplos:
        3:44 -> 224
        5:00 -> 300
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value or ":" not in value:
        return None

    try:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    except (ValueError, TypeError):
        return None


def build_fighter_rows(df):
    rows = []

    for _, fight in df.iterrows():
        base = {
            "fight_id": fight["fight_id"],
            "event_id": fight["event_id"],
            "event_name": fight["event_name"],
            "bout": fight["bout"],
            "weight_class": fight["weight_class"],
            "method": fight["method"],
            "ending_round": fight["ending_round"],
            "ending_time": fight["ending_time"],
            "time_format": fight["time_format"],
            "referee": fight["referee"],
            "details": fight["details"],
        }

        duration_seconds = parse_time_to_seconds(fight["ending_time"])

        base["fight_duration_seconds"] = (
            duration_seconds
            if pd.notna(duration_seconds)
            else None
        )

        for fighter_number, opponent_number in [(1, 2), (2, 1)]:
            fighter_id = fight[f"fighter_{fighter_number}_id"]
            opponent_id = fight[f"fighter_{opponent_number}_id"]

            row = base.copy()

            row["fighter_id"] = fighter_id
            row["fighter_name"] = fight[f"fighter_{fighter_number}_name"]

            row["opponent_id"] = opponent_id
            row["opponent_name"] = fight[f"fighter_{opponent_number}_name"]

            row["result"] = fight[f"fighter_{fighter_number}_result"]
            row["opponent_result"] = fight[f"fighter_{opponent_number}_result"]

            for column in STAT_COLUMNS:
                source_column = (
                    f"{column}_fighter_{fighter_number}"
                )

                row[column] = fight[source_column]

            rows.append(row)

    return pd.DataFrame(rows)


def validate(df, source):
    errors = []

    expected_rows = len(source) * 2

    if len(df) != expected_rows:
        errors.append(
            f"Quantidade de linhas inesperada: "
            f"{len(df)} (esperado: {expected_rows})."
        )

    duplicate_keys = df.duplicated(
        subset=["fight_id", "fighter_id"],
        keep=False,
    )

    if duplicate_keys.any():
        errors.append(
            f"Combinações fight_id/fighter_id duplicadas: "
            f"{duplicate_keys.sum()}."
        )

    fight_counts = df.groupby("fight_id")["fighter_id"].nunique()

    invalid_fights = fight_counts[fight_counts != 2]

    if len(invalid_fights) > 0:
        errors.append(
            f"Lutas com quantidade diferente de 2 lutadores: "
            f"{len(invalid_fights)}."
        )

    self_fights = df[
        df["fighter_id"] == df["opponent_id"]
    ]

    if not self_fights.empty:
        errors.append(
            f"Lutas onde fighter_id == opponent_id: "
            f"{len(self_fights)}."
        )

    invalid_results = ~df["result"].isin(["W", "L", "D", "NC"])

    if invalid_results.any():
        errors.append(
            f"Resultados inválidos: "
            f"{invalid_results.sum()}."
        )

    invalid_opponent_results = ~df["opponent_result"].isin(
        ["W", "L", "D", "NC"]
    )

    if invalid_opponent_results.any():
        errors.append(
            f"Resultados do oponente inválidos: "
            f"{invalid_opponent_results.sum()}."
        )

    invalid_duration = (
        df["fight_duration_seconds"].notna()
        & (df["fight_duration_seconds"] < 0)
    )

    if invalid_duration.any():
        errors.append(
            f"Durações negativas: {invalid_duration.sum()}."
        )

    # Quando há estatísticas, landed nunca pode ser maior que attempted.
    for metric in [
        "significant_strikes",
        "total_strikes",
        "takedowns",
        "head_strikes",
        "body_strikes",
        "leg_strikes",
        "distance_strikes",
        "clinch_strikes",
        "ground_strikes",
    ]:
        landed = df[f"{metric}_landed"]
        attempted = df[f"{metric}_attempted"]

        invalid = (
            landed.notna()
            & attempted.notna()
            & (landed > attempted)
        )

        if invalid.any():
            errors.append(
                f"{metric}: landed > attempted em "
                f"{invalid.sum()} registros."
            )

    # Percentuais devem permanecer entre 0 e 100.
    for column in [
        "significant_strikes_pct",
        "takedown_pct",
    ]:
        invalid = (
            df[column].notna()
            & (
                (df[column] < 0)
                | (df[column] > 100)
            )
        )

        if invalid.any():
            errors.append(
                f"{column}: valores fora de 0-100: "
                f"{invalid.sum()}."
            )

    if errors:
        print("\n=== ERROS DE VALIDAÇÃO ===")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "Dataset inválido. Arquivo não será salvo."
        )

    print("\n=== VALIDAÇÃO ===")
    print("Linhas preservadas: OK")
    print("Combinações fight/fighter: OK")
    print("Duas participações por luta: OK")
    print("Fighter != opponent: OK")
    print("Resultados: OK")
    print("Duração: OK")
    print("Landed <= Attempted: OK")
    print("Percentuais: OK")


def main():
    print("=== BUILD FIGHTER FIGHT FEATURES ===")

    print("\nLendo fight_features.csv...")

    source = pd.read_csv(INPUT_FILE)

    print(f"Lutas carregadas: {len(source)}")

    print("\nTransformando uma linha por lutador...")

    df = build_fighter_rows(source)

    print(f"Combinações luta/lutador: {len(df)}")

    validate(df, source)

    # Ordenação estável para facilitar inspeção.
    df = df.sort_values(
        ["fight_id", "fighter_id"]
    ).reset_index(drop=True)

    print("\nSalvando arquivo...")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nArquivo salvo em:")
    print(OUTPUT_FILE)

    print("\n=== RESUMO ===")

    print(f"Lutas:                  {source['fight_id'].nunique()}")
    print(f"Lutadores/luta:         {len(df)}")
    print(
        f"Lutas com stats:        "
        f"{df['fight_id'].notna().groupby(df['fight_id']).any().sum()}"
    )
    print(
        f"Duração disponível:     "
        f"{df['fight_duration_seconds'].notna().groupby(df['fight_id']).any().sum()}"
    )

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()