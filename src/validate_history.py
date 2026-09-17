from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT_DIR / "data" / "features"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=== VALIDATE FIGHTER HISTORY ===")

    path = FEATURES_DIR / "fighter_history.csv"

    print(f"\nLendo: {path}")

    df = pd.read_csv(path)

    print(f"Participações carregadas: {len(df)}")

    issues = []

    # ========================================================
    # CHAVES
    # ========================================================

    print("\n=== IDENTIDADE ===")

    duplicate_keys = df[
        df.duplicated(
            subset=["fight_id", "fighter_id"],
            keep=False,
        )
    ]

    print(
        "Duplicatas (fight_id, fighter_id): "
        f"{len(duplicate_keys)}"
    )

    if not duplicate_keys.empty:
        issues.append(
            "Duplicatas de participação"
        )

    # ========================================================
    # HISTÓRICO = LUTAS ANTERIORES
    # ========================================================

    print("\n=== HISTÓRICO ===")

    expected_career_fights = (
        df.groupby("fighter_id")
        .cumcount()
    )

    mismatches = df[
        df["career_fights_before"]
        != expected_career_fights
    ]

    print(
        "career_fights_before consistente: "
        f"{len(mismatches) == 0}"
    )

    if not mismatches.empty:
        print(
            mismatches[
                [
                    "fighter_id",
                    "fighter_name",
                    "fight_id",
                    "career_fights_before",
                ]
            ].head(20).to_string(index=False)
        )

        issues.append(
            "career_fights_before inconsistente"
        )

    # ========================================================
    # CONTADORES
    # ========================================================

    print("\n=== CONTADORES ===")

    counter_sum = (
        df["wins_before"].fillna(0)
        + df["losses_before"].fillna(0)
        + df["draws_before"].fillna(0)
        + df["nc_before"].fillna(0)
    )

    counter_mismatches = df[
        counter_sum
        != df["career_fights_before"]
    ]

    print(
        "Contadores = carreira anterior: "
        f"{len(counter_mismatches) == 0}"
    )

    if not counter_mismatches.empty:
        issues.append(
            "Contadores históricos inconsistentes"
        )

    # ========================================================
    # PRIMEIRA LUTA
    # ========================================================

    print("\n=== ESTREIAS ===")

    debuts = df[
        df["career_fights_before"] == 0
    ]

    debut_issues = debuts[
        (
            debuts["wins_before"] != 0
        )
        |
        (
            debuts["losses_before"] != 0
        )
        |
        (
            debuts["draws_before"] != 0
        )
        |
        (
            debuts["nc_before"] != 0
        )
    ]

    print(
        "Estreias sem histórico: "
        f"{len(debut_issues) == 0}"
    )

    if not debut_issues.empty:
        issues.append(
            "Estreias possuem histórico"
        )

    # ========================================================
    # WIN RATE
    # ========================================================

    print("\n=== WIN RATE ===")

    expected_win_rate = (
        df["wins_before"]
        / df["career_fights_before"]
    )

    expected_win_rate = expected_win_rate.where(
        df["career_fights_before"] > 0,
        0,
    )

    win_rate_diff = (
        df["win_rate_before"]
        - expected_win_rate
    ).abs()

    win_rate_issues = df[
        win_rate_diff > 1e-10
    ]

    print(
        "Win rate consistente: "
        f"{len(win_rate_issues) == 0}"
    )

    if not win_rate_issues.empty:
        issues.append(
            "Win rate inconsistente"
        )

    # ========================================================
    # ESTATÍSTICAS HISTÓRICAS
    # ========================================================

    print("\n=== ESTATÍSTICAS ===")

    metric_columns = [
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

    statistic_issues = []

    for metric in metric_columns:

        sum_column = (
            f"sum_{metric}_before"
        )

        avg_column = (
            f"avg_{metric}_before"
        )

        if (
            sum_column not in df.columns
            or avg_column not in df.columns
        ):
            statistic_issues.append(
                metric
            )
            continue

        # Estreias não podem possuir histórico.
        debut_sum = debuts[
            sum_column
        ].fillna(0)

        if not (debut_sum == 0).all():
            statistic_issues.append(
                f"{metric}: debut sum"
            )

    print(
        "Estatísticas ausentes nas estreias: "
        f"{len(statistic_issues) == 0}"
    )

    if statistic_issues:
        print(
            "\nProblemas:"
        )

        for issue in statistic_issues:
            print(f" - {issue}")

        issues.append(
            "Estatísticas históricas "
            "inconsistentes"
        )

    # ========================================================
    # RESULTADOS DA PRÓPRIA LUTA
    # ========================================================

    print("\n=== LEAKAGE ===")

    # As features históricas não devem depender
    # do resultado da luta atual.

    history_columns = [
        "career_fights_before",
        "wins_before",
        "losses_before",
        "draws_before",
        "nc_before",
        "win_rate_before",
    ]

    leakage = []

    for column in history_columns:

        if column not in df.columns:
            leakage.append(column)

    print(
        "Colunas históricas presentes: "
        f"{len(leakage) == 0}"
    )

    if leakage:
        print(
            f"Ausentes: {leakage}"
        )
        issues.append(
            "Colunas históricas ausentes"
        )

    # ========================================================
    # RESULTADO
    # ========================================================

    print("\n=== RESULTADO ===")

    if issues:

        print(
            f"Histórico inválido: "
            f"{len(issues)} problema(s)."
        )

        for issue in issues:
            print(f" - {issue}")

        raise RuntimeError(
            "Validação temporal falhou."
        )

    print(
        "Histórico válido: "
        "nenhum problema encontrado."
    )

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()