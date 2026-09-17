from pathlib import Path
import re

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "round_stats.csv"
FEATURES_DIR = ROOT_DIR / "data" / "features"
OUTPUT_FILE = FEATURES_DIR / "round_stats_numeric.csv"


# ============================================================
# PARSERS
# ============================================================

def parse_landed_attempted(value):
    """
    Converte valores no formato 'X of Y'.

    Exemplos:
        '5 of 10' -> (5, 10)
        '0 of 0'  -> (0, 0)
        NaN       -> (NaN, NaN)
    """

    if pd.isna(value):
        return pd.NA, pd.NA

    value = str(value).strip()

    match = re.fullmatch(r"(\d+)\s+of\s+(\d+)", value)

    if not match:
        raise ValueError(f"Formato inválido de landed/attempted: {value!r}")

    landed = int(match.group(1))
    attempted = int(match.group(2))

    return landed, attempted


def parse_percentage(value):
    """
    Converte:
        '50%' -> 50
        '0%'  -> 0
        '---' -> NaN
        NaN   -> NaN
    """

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value in {"---", "--", ""}:
        return pd.NA

    match = re.fullmatch(r"(\d+(?:\.\d+)?)%", value)

    if not match:
        raise ValueError(f"Formato inválido de porcentagem: {value!r}")

    return float(match.group(1))


def parse_control_time(value):
    """
    Converte tempo de controle para segundos.

    Exemplos:
        '0:00' -> 0
        '1:35' -> 95
        '--'   -> NaN
        NaN    -> NaN
    """

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value in {"--", "---", ""}:
        return pd.NA

    match = re.fullmatch(r"(\d+):([0-5]\d)", value)

    if not match:
        raise ValueError(f"Formato inválido de control_time: {value!r}")

    minutes = int(match.group(1))
    seconds = int(match.group(2))

    return minutes * 60 + seconds


def parse_numeric(value):
    """
    Converte campos numéricos simples.
    """

    if pd.isna(value):
        return pd.NA

    return float(value)


# ============================================================
# TRANSFORMAÇÃO
# ============================================================

LANDED_ATTEMPTED_COLUMNS = {
    "significant_strikes": "significant_strikes",
    "total_strikes": "total_strikes",
    "takedowns": "takedowns",
    "head_strikes": "head_strikes",
    "body_strikes": "body_strikes",
    "leg_strikes": "leg_strikes",
    "distance_strikes": "distance_strikes",
    "clinch_strikes": "clinch_strikes",
    "ground_strikes": "ground_strikes",
}


PERCENTAGE_COLUMNS = [
    "significant_strikes_pct",
    "takedown_pct",
]


NUMERIC_COLUMNS = [
    "knockdowns",
    "submission_attempts",
    "reversals",
]


def build_numeric_round_stats(df):
    """
    Converte round_stats para uma estrutura numérica.
    """

    result = df[
        [
            "fight_id",
            "fighter_id",
            "round",
            "fighter_name",
        ]
    ].copy()

    # --------------------------------------------------------
    # Campos X of Y
    # --------------------------------------------------------

    for source_column, prefix in LANDED_ATTEMPTED_COLUMNS.items():

        parsed = df[source_column].apply(parse_landed_attempted)

        result[f"{prefix}_landed"] = parsed.apply(
            lambda x: x[0]
        ).astype("Int64")

        result[f"{prefix}_attempted"] = parsed.apply(
            lambda x: x[1]
        ).astype("Int64")

    # --------------------------------------------------------
    # Porcentagens
    # --------------------------------------------------------

    for column in PERCENTAGE_COLUMNS:
        result[column] = (
            df[column]
            .apply(parse_percentage)
            .astype("Float64")
        )

    # --------------------------------------------------------
    # Campos numéricos simples
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        result[column] = (
            df[column]
            .apply(parse_numeric)
            .astype("Float64")
        )

    # --------------------------------------------------------
    # Tempo de controle
    # --------------------------------------------------------

    result["control_time_seconds"] = (
        df["control_time"]
        .apply(parse_control_time)
        .astype("Int64")
    )

    return result


# ============================================================
# VALIDAÇÃO
# ============================================================

def validate_numeric_features(original, result):
    """
    Valida se a transformação preservou os dados esperados.
    """

    errors = []

    # --------------------------------------------------------
    # Quantidade de linhas
    # --------------------------------------------------------

    if len(original) != len(result):
        errors.append(
            f"Quantidade de linhas alterada: "
            f"{len(original)} -> {len(result)}"
        )

    # --------------------------------------------------------
    # Identidade dos registros
    # --------------------------------------------------------

    original_keys = original[
        ["fight_id", "fighter_id", "round"]
    ].astype(str)

    result_keys = result[
        ["fight_id", "fighter_id", "round"]
    ].astype(str)

    if not original_keys.reset_index(drop=True).equals(
        result_keys.reset_index(drop=True)
    ):
        errors.append(
            "As chaves (fight_id, fighter_id, round) "
            "não foram preservadas."
        )

    # --------------------------------------------------------
    # Duplicatas
    # --------------------------------------------------------

    duplicates = result.duplicated(
        subset=["fight_id", "fighter_id", "round"]
    ).sum()

    if duplicates != 0:
        errors.append(
            f"Foram encontradas {duplicates} "
            f"duplicatas de (fight_id, fighter_id, round)."
        )

    # --------------------------------------------------------
    # Validação X of Y
    # --------------------------------------------------------

    for column in LANDED_ATTEMPTED_COLUMNS.values():

        landed = result[f"{column}_landed"]
        attempted = result[f"{column}_attempted"]

        # landed nunca pode ser maior que attempted
        invalid = (
            landed.notna()
            & attempted.notna()
            & (landed > attempted)
        ).sum()

        if invalid:
            errors.append(
                f"{column}: {invalid} registros com "
                f"landed > attempted."
            )

        # valores negativos
        negative = (
            landed.lt(0).fillna(False)
            | attempted.lt(0).fillna(False)
        ).sum()

        if negative:
            errors.append(
                f"{column}: {negative} registros com "
                f"valores negativos."
            )

    # --------------------------------------------------------
    # Validação das porcentagens
    # --------------------------------------------------------

    for column in PERCENTAGE_COLUMNS:

        invalid = (
            result[column].notna()
            & (
                (result[column] < 0)
                | (result[column] > 100)
            )
        ).sum()

        if invalid:
            errors.append(
                f"{column}: {invalid} valores fora "
                f"do intervalo 0-100."
            )

    # --------------------------------------------------------
    # Validação de control_time
    # --------------------------------------------------------

    invalid_control = (
        result["control_time_seconds"].notna()
        & (result["control_time_seconds"] < 0)
    ).sum()

    if invalid_control:
        errors.append(
            f"control_time_seconds: {invalid_control} "
            f"valores negativos."
        )

    # --------------------------------------------------------
    # Validação de round
    # --------------------------------------------------------

    invalid_rounds = (
        result["round"].isna()
        | (result["round"] <= 0)
    ).sum()

    if invalid_rounds:
        errors.append(
            f"{invalid_rounds} rounds inválidos."
        )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    if errors:
        print("\n=== ERROS DE VALIDAÇÃO ===")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "\nA validação falhou. "
            "O arquivo não será salvo."
        )

    print("\n=== VALIDAÇÃO ===")
    print("Linhas preservadas:", len(result))
    print("Chaves preservadas: OK")
    print("Duplicatas: 0")
    print("Landed <= Attempted: OK")
    print("Porcentagens: OK")
    print("Control time: OK")
    print("Rounds: OK")
    print("Dataset numérico válido.")


# ============================================================
# RESUMO
# ============================================================

def print_summary(original, result):
    print("\n=== RESUMO ===")

    print(f"Registros originais: {len(original)}")
    print(f"Registros processados: {len(result)}")

    print("\nColunas geradas:")

    for column in result.columns:
        print(f"  - {column}")

    print("\nValores ausentes:")

    nulls = result.isna().sum()

    for column, count in nulls.items():
        if count > 0:
            print(f"  {column}: {count}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=== BUILD NUMERIC ROUND STATS ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_FILE}"
        )

    print(f"\nLendo: {INPUT_FILE}")

    original = pd.read_csv(INPUT_FILE)

    print(f"Registros carregados: {len(original)}")

    print("\nConvertendo dados...")

    result = build_numeric_round_stats(original)

    validate_numeric_features(original, result)

    print_summary(original, result)

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nArquivo salvo em:")
    print(OUTPUT_FILE)

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()