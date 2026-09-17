from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

FEATURES_DIR = ROOT_DIR / "data" / "features"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ROUND_STATS_FILE = FEATURES_DIR / "round_stats_numeric.csv"
FIGHTS_FILE = PROCESSED_DIR / "fights.csv"

FIGHTER_FEATURES_FILE = FEATURES_DIR / "fight_fighter_features.csv"
OUTPUT_FILE = FEATURES_DIR / "fight_features.csv"


# ============================================================
# COLUNAS
# ============================================================

SUM_COLUMNS = [
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
]


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def safe_percentage(landed, attempted):
    """
    Calcula percentual a partir de landed / attempted.

    Quando não houve tentativa, retorna NaN.
    """

    if attempted == 0:
        return pd.NA

    return (landed / attempted) * 100


def build_fighter_features(round_stats):
    """
    Agrega os rounds para uma linha por lutador em cada luta.
    """

    group_columns = [
        "fight_id",
        "fighter_id",
        "fighter_name",
    ]

    result = (
        round_stats
        .groupby(group_columns, as_index=False)[SUM_COLUMNS]
        .sum()
    )

    # Número de rounds disponíveis para aquele lutador
    rounds = (
        round_stats
        .groupby(group_columns)["round"]
        .nunique()
        .reset_index(name="rounds_with_stats")
    )

    result = result.merge(
        rounds,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Eficiência
    # --------------------------------------------------------

    result["significant_strikes_pct"] = (
        result.apply(
            lambda row: safe_percentage(
                row["significant_strikes_landed"],
                row["significant_strikes_attempted"],
            ),
            axis=1,
        )
        .astype("Float64")
    )

    result["takedown_pct"] = (
        result.apply(
            lambda row: safe_percentage(
                row["takedowns_landed"],
                row["takedowns_attempted"],
            ),
            axis=1,
        )
        .astype("Float64")
    )

    return result


def build_fight_features(fights, fighter_features):
    """
    Junta os dois lutadores de cada luta em uma única linha.
    """

    # --------------------------------------------------------
    # Seleciona somente lutas com exatamente dois lutadores
    # --------------------------------------------------------

    fighter_counts = (
        fighter_features
        .groupby("fight_id")["fighter_id"]
        .nunique()
    )

    valid_fight_ids = fighter_counts[
        fighter_counts == 2
    ].index

    fighter_features = fighter_features[
        fighter_features["fight_id"].isin(valid_fight_ids)
    ].copy()

    # --------------------------------------------------------
    # Junta com a tabela de lutas
    # --------------------------------------------------------

    result = fights.merge(
        fighter_features,
        left_on=["fight_id", "fighter_1_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
        suffixes=("", "_fighter_1"),
        validate="one_to_one",
    )

    result = result.rename(
        columns={
            "fighter_name": "stats_fighter_1_name",
        }
    )

    result = result.drop(
        columns=["fighter_id"]
    )

    # --------------------------------------------------------
    # Segundo lutador
    # --------------------------------------------------------

    fighter_2 = fighter_features.copy()

    fighter_2 = fighter_2.rename(
        columns={
            column: f"{column}_fighter_2"
            for column in fighter_2.columns
            if column not in ["fight_id", "fighter_id"]
        }
    )

    result = result.merge(
        fighter_2,
        left_on=["fight_id", "fighter_2_id"],
        right_on=["fight_id", "fighter_id"],
        how="left",
        validate="one_to_one",
    )

    result = result.drop(
        columns=["fighter_id"]
    )

    # --------------------------------------------------------
    # Renomeia as features do primeiro lutador
    # --------------------------------------------------------

    rename_fighter_1 = {}

    for column in SUM_COLUMNS:
        rename_fighter_1[column] = f"{column}_fighter_1"

    rename_fighter_1["rounds_with_stats"] = "rounds_with_stats_fighter_1"
    rename_fighter_1["significant_strikes_pct"] = "significant_strikes_pct_fighter_1"
    rename_fighter_1["takedown_pct"] = "takedown_pct_fighter_1"

    result = result.rename(columns=rename_fighter_1)

    # --------------------------------------------------------
    # Diferenças entre os lutadores
    # --------------------------------------------------------

    difference_columns = [
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

    for column in difference_columns:

        c1 = f"{column}_fighter_1"
        c2 = f"{column}_fighter_2"

        result[f"{column}_diff"] = (
            result[c1] - result[c2]
        )

    # Eficiência também pode ter diferença
    result["significant_strikes_pct_diff"] = (
        result["significant_strikes_pct_fighter_1"]
        - result["significant_strikes_pct_fighter_2"]
    )

    result["takedown_pct_diff"] = (
        result["takedown_pct_fighter_1"]
        - result["takedown_pct_fighter_2"]
    )

    return result


# ============================================================
# VALIDAÇÃO
# ============================================================

def validate_fighter_features(round_stats, fighter_features):
    errors = []

    # --------------------------------------------------------
    # Cada combinação luta + lutador deve ser única
    # --------------------------------------------------------

    duplicates = fighter_features.duplicated(
        subset=["fight_id", "fighter_id"]
    ).sum()

    if duplicates:
        errors.append(
            f"{duplicates} duplicatas em "
            f"(fight_id, fighter_id)."
        )

    # --------------------------------------------------------
    # Cada lutador deve ter pelo menos um round
    # --------------------------------------------------------

    if (fighter_features["rounds_with_stats"] <= 0).any():
        errors.append(
            "Existem lutadores com rounds_with_stats <= 0."
        )

    # --------------------------------------------------------
    # As somas não podem ser negativas
    # --------------------------------------------------------

    for column in SUM_COLUMNS:

        if (fighter_features[column] < 0).any():
            errors.append(
                f"{column}: existem valores negativos."
            )

    # --------------------------------------------------------
    # Landed <= Attempted
    # --------------------------------------------------------

    paired_columns = [
        ("significant_strikes_landed", "significant_strikes_attempted"),
        ("total_strikes_landed", "total_strikes_attempted"),
        ("takedowns_landed", "takedowns_attempted"),
        ("head_strikes_landed", "head_strikes_attempted"),
        ("body_strikes_landed", "body_strikes_attempted"),
        ("leg_strikes_landed", "leg_strikes_attempted"),
        ("distance_strikes_landed", "distance_strikes_attempted"),
        ("clinch_strikes_landed", "clinch_strikes_attempted"),
        ("ground_strikes_landed", "ground_strikes_attempted"),
    ]

    for landed, attempted in paired_columns:

        invalid = (
            fighter_features[landed]
            > fighter_features[attempted]
        ).sum()

        if invalid:
            errors.append(
                f"{landed}: {invalid} registros "
                f"com landed > attempted."
            )

    # --------------------------------------------------------
    # Percentuais
    # --------------------------------------------------------

    for column in [
        "significant_strikes_pct",
        "takedown_pct",
    ]:

        invalid = (
            fighter_features[column].notna()
            & (
                (fighter_features[column] < 0)
                | (fighter_features[column] > 100)
            )
        ).sum()

        if invalid:
            errors.append(
                f"{column}: {invalid} valores "
                f"fora de 0-100."
            )

    if errors:

        print("\n=== ERROS DE VALIDAÇÃO ===")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "\nValidação das features por lutador falhou."
        )

    print("\n=== VALIDAÇÃO — LUTADOR/LUTA ===")
    print("Combinações duplicadas: 0")
    print("Rounds válidos: OK")
    print("Valores negativos: 0")
    print("Landed <= Attempted: OK")
    print("Percentuais: OK")


def validate_fight_features(
    fights,
    fighter_features,
    fight_features,
):
    errors = []

    # --------------------------------------------------------
    # Uma linha por luta
    # --------------------------------------------------------

    duplicates = fight_features.duplicated(
        subset=["fight_id"]
    ).sum()

    if duplicates:
        errors.append(
            f"{duplicates} duplicatas de fight_id."
        )

    # --------------------------------------------------------
    # Todas as lutas originais devem permanecer
    # --------------------------------------------------------

    if len(fight_features) != len(fights):
        errors.append(
            f"Quantidade de lutas alterada: "
            f"{len(fights)} -> {len(fight_features)}."
        )

    original_ids = set(fights["fight_id"])
    feature_ids = set(fight_features["fight_id"])

    missing_ids = original_ids - feature_ids
    extra_ids = feature_ids - original_ids

    if missing_ids:
        errors.append(
            f"{len(missing_ids)} fight_ids desapareceram."
        )

    if extra_ids:
        errors.append(
            f"{len(extra_ids)} fight_ids inexistentes "
            f"foram adicionados."
        )

    # --------------------------------------------------------
    # Lutas com estatísticas
    # --------------------------------------------------------

    stats_by_fight = (
        fighter_features
        .groupby("fight_id")["fighter_id"]
        .nunique()
    )

    fights_with_stats = (
        (stats_by_fight == 2)
        .sum()
    )

    fights_without_stats = (
        ~fight_features["fight_id"].isin(
            stats_by_fight.index
        )
    ).sum()

    expected_with_stats = 8866
    expected_without_stats = 21

    if fights_with_stats != expected_with_stats:
        errors.append(
            f"Lutas com stats: {fights_with_stats} "
            f"(esperado: {expected_with_stats})."
        )

    if fights_without_stats != expected_without_stats:
        errors.append(
            f"Lutas sem stats: {fights_without_stats} "
            f"(esperado: {expected_without_stats})."
        )

    # --------------------------------------------------------
    # Nenhuma luta pode ter somente um lutador nas stats
    # --------------------------------------------------------

    incomplete_stats = (
        stats_by_fight[
            stats_by_fight == 1
        ]
    )

    if len(incomplete_stats):
        errors.append(
            f"{len(incomplete_stats)} lutas possuem "
            f"estatísticas de apenas um lutador."
        )

    # --------------------------------------------------------
    # Verifica os fighters das lutas
    # --------------------------------------------------------

    missing_fighter_1 = (
        fight_features["fighter_1_id"].isna()
    ).sum()

    missing_fighter_2 = (
        fight_features["fighter_2_id"].isna()
    ).sum()

    if missing_fighter_1:
        errors.append(
            f"{missing_fighter_1} lutas sem fighter_1."
        )

    if missing_fighter_2:
        errors.append(
            f"{missing_fighter_2} lutas sem fighter_2."
        )

    # --------------------------------------------------------
    # Features estatísticas
    #
    # Ausência é permitida SOMENTE nas 21 lutas
    # sem round stats.
    # --------------------------------------------------------

    feature_columns = [
        "significant_strikes_landed",
        "takedowns_landed",
        "knockdowns",
        "control_time_seconds",
    ]

    for column in feature_columns:

        c1 = f"{column}_fighter_1"
        c2 = f"{column}_fighter_2"

        missing_1 = fight_features[c1].isna()
        missing_2 = fight_features[c2].isna()

        # Uma luta sem stats deve ter ambos os lutadores
        # sem stats.
        asymmetric = missing_1 != missing_2

        if asymmetric.any():
            errors.append(
                f"{column}: existem lutas com ausência "
                f"de stats em apenas um lutador."
            )

    # --------------------------------------------------------
    # Verifica que as 21 lutas sem stats são exatamente
    # as que esperávamos.
    # --------------------------------------------------------

    missing_stats_mask = (
        fight_features[
            "significant_strikes_landed_fighter_1"
        ].isna()
    )

    missing_stats_ids = set(
        fight_features.loc[
            missing_stats_mask,
            "fight_id"
        ]
    )

    expected_missing_stats_ids = set(
        fights[
            ~fights["fight_id"].isin(
                fighter_features["fight_id"]
            )
        ]["fight_id"]
    )

    if missing_stats_ids != expected_missing_stats_ids:
        errors.append(
            "As lutas sem stats no resultado não "
            "correspondem às lutas sem stats na "
            "base intermediária."
        )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    if errors:

        print("\n=== ERROS DE VALIDAÇÃO — LUTAS ===")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "\nValidação das features de luta falhou."
        )

    print("\n=== VALIDAÇÃO — LUTA ===")
    print("Fight IDs duplicados: 0")
    print(f"Lutas preservadas: {len(fight_features)}")
    print(f"Lutas com stats: {fights_with_stats}")
    print(f"Lutas sem stats: {fights_without_stats}")
    print("Fighters presentes: OK")
    print("Ausência de stats consistente: OK")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=== BUILD FIGHT FEATURES ===")

    # --------------------------------------------------------
    # Leitura
    # --------------------------------------------------------

    print("\nLendo round_stats_numeric.csv...")

    round_stats = pd.read_csv(
        ROUND_STATS_FILE
    )

    print(
        f"Round stats carregados: "
        f"{len(round_stats)}"
    )

    print("\nLendo fights.csv...")

    fights = pd.read_csv(
        FIGHTS_FILE
    )

    print(
        f"Lutas carregadas: "
        f"{len(fights)}"
    )

    # --------------------------------------------------------
    # Agregação por lutador
    # --------------------------------------------------------

    print("\nAgregando rounds por lutador...")

    fighter_features = build_fighter_features(
        round_stats
    )

    print(
        f"Combinações luta/lutador: "
        f"{len(fighter_features)}"
    )

    validate_fighter_features(
        round_stats,
        fighter_features,
    )

    # --------------------------------------------------------
    # Salva camada intermediária
    # --------------------------------------------------------

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fighter_features.to_csv(
        FIGHTER_FEATURES_FILE,
        index=False,
    )

    print(
        f"\nArquivo intermediário salvo em:"
    )
    print(FIGHTER_FEATURES_FILE)

    # --------------------------------------------------------
    # Agregação por luta
    # --------------------------------------------------------

    print("\nMontando uma linha por luta...")

    fight_features = build_fight_features(
        fights,
        fighter_features,
    )

    validate_fight_features(
        fights,
        fighter_features,
        fight_features,
    )

    # --------------------------------------------------------
    # Salva resultado final
    # --------------------------------------------------------

    fight_features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nArquivo final salvo em:"
    )
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------

    print("\n=== RESUMO ===")
    print(
        f"Round stats:        {len(round_stats)}"
    )
    print(
        f"Lutador/luta:       {len(fighter_features)}"
    )
    print(
    f"Lutas preservadas:  {len(fight_features)}"
    )

    print(
        f"Lutas com stats:    {fighter_features['fight_id'].nunique()}"
    )

    print(
        f"Lutas sem stats:    "
        f"{len(fight_features) - fighter_features['fight_id'].nunique()}"
    )

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()