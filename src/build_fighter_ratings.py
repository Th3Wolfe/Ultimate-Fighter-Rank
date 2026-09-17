from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"

FIGHTS_FILE = PROCESSED_DIR / "fights.csv"
EVENTS_FILE = PROCESSED_DIR / "events.csv"
OUTPUT_FILE = FEATURES_DIR / "fighter_ratings.csv"


# ============================================================
# Parâmetros do Elo.
#
# Estes valores são um ponto de partida, não uma escolha
# validada. Devem ser comparados empiricamente com outras
# configurações via avaliação temporal (ver
# docs/rating_methodology.md, seções 4.2, 23 e 24) antes de
# serem considerados definitivos.
# ============================================================

INITIAL_RATING = 1500.0
BASE_K = 32.0
BOOSTED_K = 64.0
SCALE = 400.0

# Número de lutas, após a estreia ou após uma mudança de
# divisão, em que o rating é considerado de baixa confiança
# e recebe um K-factor maior para se recalibrar mais rápido.
PROVISIONAL_FIGHTS = 3

# Categorias que não representam uma divisão de peso padrão.
# Nunca contam como "mudança de divisão" ao comparar com a
# divisão anterior do lutador.
NON_STANDARD_DIVISIONS = {"Catch Weight", "Open Weight"}

# Divisões conhecidas, usadas para extrair a divisão real de
# rótulos "sujos" de weight_class (que misturam divisão com
# tipo de luta: "Lightweight Bout", "UFC Lightweight Title
# Bout", "Ultimate Fighter 21 Welterweight Tournament Title
# Bout" representam todos a mesma divisão: Lightweight).
#
# Eventos anteriores à existência de categorias de peso no UFC
# (ex.: "UFC 2 Tournament Title Bout") não correspondem a
# nenhuma divisão conhecida e resultam em divisão = None. Isso
# é esperado e correto: essas lutas realmente não tinham
# categoria de peso.
KNOWN_DIVISIONS = [
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Women's Featherweight",
    "Strawweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Catch Weight",
    "Open Weight",
]


def normalize_division(weight_class):
    """
    Extrai a divisão real a partir do rótulo bruto de
    weight_class.

    Retorna None quando nenhuma divisão conhecida é
    encontrada no rótulo (ex.: lutas de torneio anteriores à
    existência de categorias de peso no UFC).
    """

    if pd.isna(weight_class):
        return None

    label = str(weight_class)

    # Nomes compostos e divisões femininas precisam ser
    # checados antes dos nomes simples que são substring
    # deles (ex.: "Light Heavyweight" contém "Heavyweight";
    # "Women's Flyweight" contém "Flyweight").
    for division in sorted(KNOWN_DIVISIONS, key=len, reverse=True):
        if division in label:
            return division

    return None


def validate_input(fights, events):
    errors = []

    required_fights = [
        "fight_id",
        "event_id",
        "fighter_1_id",
        "fighter_2_id",
        "fighter_1_result",
        "fighter_2_result",
        "weight_class",
    ]

    missing = [
        column for column in required_fights
        if column not in fights.columns
    ]

    if missing:
        errors.append(
            f"Colunas obrigatórias ausentes em fights.csv: {missing}"
        )

    if "event_id" not in events.columns or "date" not in events.columns:
        errors.append("events.csv não possui event_id/date.")

    duplicate_fights = fights.duplicated(subset=["fight_id"])

    if duplicate_fights.any():
        errors.append(
            f"fight_id duplicado em fights.csv: "
            f"{duplicate_fights.sum()}."
        )

    valid_results = {"W", "L", "D", "NC"}

    invalid_results = ~(
        fights["fighter_1_result"].isin(valid_results)
        & fights["fighter_2_result"].isin(valid_results)
    )

    if invalid_results.any():
        errors.append(
            f"Resultados fora do esperado (W/L/D/NC): "
            f"{invalid_results.sum()}."
        )

    if errors:
        for error in errors:
            print(f"- {error}")

        raise ValueError("Entrada inválida.")


def expected_score(rating_a, rating_b, scale=SCALE):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / scale))


def build_ratings(fights):
    """
    Processa as lutas em ordem cronológica e calcula o Elo de
    cada lutador antes e depois de cada luta.

    Decisões metodológicas aplicadas (registradas em
    docs/rating_methodology.md, seção "37. Decisões
    registradas"):

    - Rating único por lutador, cross-division. Não existe um
      Elo separado por categoria de peso: a divisão é mantida
      apenas como metadado de contexto. Isso implica que, ao
      mudar de divisão, o lutador carrega consigo o rating que
      tinha antes da mudança.
    - No Contest não atualiza o rating de nenhum dos dois
      lutadores (a luta é tratada como se não tivesse ocorrido
      para fins de força competitiva), mas conta normalmente
      na contagem de lutas de cada um.
    - Empate atualiza os dois ratings com resultado 0.5 / 0.5.
    - A estreia de um lutador e as PROVISIONAL_FIGHTS lutas
      seguintes a uma mudança de divisão usam um K-factor maior
      (BOOSTED_K), para permitir uma recalibração mais rápida
      em cenários de alta incerteza. Isso é um parâmetro a ser
      validado, não uma regra definitiva.
    """

    fights = fights.sort_values(
        ["date", "fight_id"]
    ).reset_index(drop=True)

    rating = {}
    fights_count = {}
    fights_since_switch = {}
    current_division = {}

    def ensure_known(fighter_id):
        rating.setdefault(fighter_id, INITIAL_RATING)
        fights_count.setdefault(fighter_id, 0)
        fights_since_switch.setdefault(fighter_id, 0)
        current_division.setdefault(fighter_id, None)

    def is_provisional(fighter_id):
        return (
            fights_count[fighter_id] < PROVISIONAL_FIGHTS
            or fights_since_switch[fighter_id] < PROVISIONAL_FIGHTS
        )

    def k_factor(fighter_id):
        return BOOSTED_K if is_provisional(fighter_id) else BASE_K

    rows = []

    for fight in fights.itertuples(index=False):
        fighter_1_id = fight.fighter_1_id
        fighter_2_id = fight.fighter_2_id

        ensure_known(fighter_1_id)
        ensure_known(fighter_2_id)

        division = normalize_division(fight.weight_class)

        division_change = {}

        for fighter_id in (fighter_1_id, fighter_2_id):
            previous_division = current_division[fighter_id]

            changed = (
                division is not None
                and previous_division is not None
                and division not in NON_STANDARD_DIVISIONS
                and previous_division not in NON_STANDARD_DIVISIONS
                and division != previous_division
            )

            division_change[fighter_id] = changed

            if changed:
                fights_since_switch[fighter_id] = 0

        is_debut = {
            fighter_1_id: fights_count[fighter_1_id] == 0,
            fighter_2_id: fights_count[fighter_2_id] == 0,
        }

        rating_1_before = rating[fighter_1_id]
        rating_2_before = rating[fighter_2_id]

        expected_1 = expected_score(rating_1_before, rating_2_before)
        expected_2 = 1.0 - expected_1

        result_1 = fight.fighter_1_result
        result_2 = fight.fighter_2_result

        is_no_contest = result_1 == "NC" or result_2 == "NC"

        k_1 = k_factor(fighter_1_id)
        k_2 = k_factor(fighter_2_id)

        if is_no_contest:
            rating_1_after = rating_1_before
            rating_2_after = rating_2_before
        else:
            if result_1 == "W":
                actual_1, actual_2 = 1.0, 0.0
            elif result_1 == "L":
                actual_1, actual_2 = 0.0, 1.0
            elif result_1 == "D":
                actual_1, actual_2 = 0.5, 0.5
            else:
                raise ValueError(
                    f"Resultado inesperado na luta "
                    f"{fight.fight_id}: {result_1}."
                )

            rating_1_after = rating_1_before + k_1 * (
                actual_1 - expected_1
            )
            rating_2_after = rating_2_before + k_2 * (
                actual_2 - expected_2
            )

        for (
            fighter_id,
            opponent_id,
            result,
            rating_before,
            rating_after,
            expected,
            k_used,
        ) in (
            (
                fighter_1_id, fighter_2_id, result_1,
                rating_1_before, rating_1_after, expected_1, k_1,
            ),
            (
                fighter_2_id, fighter_1_id, result_2,
                rating_2_before, rating_2_after, expected_2, k_2,
            ),
        ):
            rows.append({
                "fight_id": fight.fight_id,
                "date": fight.date,
                "fighter_id": fighter_id,
                "opponent_id": opponent_id,
                "division": division,
                "result": result,
                "career_fights_before": fights_count[fighter_id],
                "is_debut": is_debut[fighter_id],
                "is_division_change": division_change[fighter_id],
                "k_factor": k_used,
                "rating_updated": not is_no_contest,
                "expected_score": expected,
                "rating_before": rating_before,
                "rating_after": rating_after,
            })

        rating[fighter_1_id] = rating_1_after
        rating[fighter_2_id] = rating_2_after

        for fighter_id in (fighter_1_id, fighter_2_id):
            fights_count[fighter_id] += 1
            fights_since_switch[fighter_id] += 1

            if division is not None:
                current_division[fighter_id] = division

    return pd.DataFrame(rows)


def validate_output(ratings, fights):
    errors = []

    expected_rows = len(fights) * 2

    if len(ratings) != expected_rows:
        errors.append(
            f"Quantidade de linhas inesperada: "
            f"{len(ratings)} (esperado {expected_rows})."
        )

    duplicates = ratings.duplicated(
        subset=["fight_id", "fighter_id"],
        keep=False,
    )

    if duplicates.any():
        errors.append(
            f"Participações duplicadas: {duplicates.sum()}."
        )

    # A primeira luta de cada lutador sempre parte do rating
    # inicial.
    first_fights = ratings["career_fights_before"] == 0

    invalid_first = (
        ratings.loc[first_fights, "rating_before"] != INITIAL_RATING
    )

    if invalid_first.any():
        errors.append(
            f"Estreias sem rating inicial correto: "
            f"{invalid_first.sum()}."
        )

    # Lutas sem NC devem alterar o rating (exceto no caso raro
    # de expected_score == actual_score, que não zera a
    # verificação abaixo pois comparamos com rating_updated).
    nc_changed = (
        ~ratings["rating_updated"]
        & (ratings["rating_before"] != ratings["rating_after"])
    )

    if nc_changed.any():
        errors.append(
            f"No Contest alterou o rating: {nc_changed.sum()}."
        )

    # Nenhum rating pode ser nulo ou não finito.
    invalid_rating = (
        ratings["rating_after"].isna()
        | ~ratings["rating_after"].apply(
            lambda value: value == value and abs(value) != float("inf")
        )
    )

    if invalid_rating.any():
        errors.append(
            f"Ratings inválidos (NaN/infinito): "
            f"{invalid_rating.sum()}."
        )

    # A ordem cronológica de rating_before de cada lutador deve
    # ser consistente com o rating_after da luta anterior.
    ordered = ratings.sort_values(
        ["fighter_id", "date", "fight_id"]
    )

    previous_after = ordered.groupby("fighter_id")["rating_after"].shift(1)

    mismatch = (
        previous_after.notna()
        & (previous_after != ordered["rating_before"])
    )

    if mismatch.any():
        errors.append(
            f"rating_before não corresponde ao rating_after "
            f"anterior: {mismatch.sum()}."
        )

    if errors:
        print("\n=== ERROS DE VALIDAÇÃO ===")

        for error in errors:
            print(f"- {error}")

        raise ValueError(
            "Ratings inválidos. Arquivo não será salvo."
        )

    print("\n=== VALIDAÇÃO ===")
    print("Linhas esperadas: OK")
    print("Participações duplicadas: 0")
    print("Ratings de estreia: OK")
    print("No Contest sem alteração de rating: OK")
    print("Ratings finitos: OK")
    print("Continuidade rating_before/rating_after: OK")


def main():
    print("=== BUILD FIGHTER RATINGS (Elo) ===")

    print("\nLendo fights.csv...")

    fights = pd.read_csv(FIGHTS_FILE)

    print(f"Lutas carregadas: {len(fights)}")

    print("\nLendo events.csv...")

    events = pd.read_csv(EVENTS_FILE)

    events["date"] = pd.to_datetime(events["date"])

    validate_input(fights, events)

    event_dates = events[["event_id", "date"]].drop_duplicates(
        subset=["event_id"]
    )

    fights = fights.merge(
        event_dates,
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    missing_dates = fights["date"].isna().sum()

    if missing_dates:
        print(
            f"\nAviso: {missing_dates} lutas não possuem "
            "data de evento."
        )

    print("\nCalculando ratings Elo...")

    ratings = build_ratings(fights)

    validate_output(ratings, fights)

    ratings = ratings.sort_values(
        ["date", "fight_id", "fighter_id"]
    ).reset_index(drop=True)

    ratings.to_csv(OUTPUT_FILE, index=False)

    print("\nArquivo salvo em:")
    print(OUTPUT_FILE)

    print("\n=== RESUMO ===")
    print(f"Participações:            {len(ratings)}")
    print(f"Lutadores:                {ratings['fighter_id'].nunique()}")
    print(f"Estreias:                 {ratings['is_debut'].sum()}")
    print(
        f"Mudanças de divisão:      "
        f"{ratings['is_division_change'].sum()}"
    )
    print(
        f"No Contest (sem update):  "
        f"{(~ratings['rating_updated']).sum()}"
    )
    print(
        f"Divisão não identificada: "
        f"{ratings['division'].isna().sum()}"
    )

    latest = (
        ratings.sort_values(["fighter_id", "date", "fight_id"])
        .groupby("fighter_id")
        .tail(1)
        .sort_values("rating_after", ascending=False)
        .head(10)
    )

    print("\nTop 10 ratings atuais:")

    for row in latest.itertuples(index=False):
        print(f"  {row.rating_after:8.1f}  {row.fighter_id}")

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()
