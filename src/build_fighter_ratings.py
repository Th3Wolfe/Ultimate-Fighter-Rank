from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
# validada (ver docs/rating_methodology.md, seção 37.4). Para
# compará-los empiricamente com outras configurações, ver
# src/param_sweep.py (seção 53, item 5 do development_guideline.md),
# que reaproveita EloConfig/build_ratings/validate_output daqui
# em vez de duplicar esta lógica.
# ============================================================


@dataclass(frozen=True)
class EloConfig:
    """
    Agrupa os parâmetros livres do Elo v0.1, para permitir rodar
    `build_ratings()` com configurações diferentes (sweep de
    parâmetros) sem duplicar `build_fighter_ratings.py`.

    Os valores default reproduzem exatamente o baseline v0.1
    (docs/rating_methodology.md, seção 37.4).
    """

    initial_rating: float = 1500.0
    base_k: float = 32.0
    boosted_k: float = 64.0
    scale: float = 400.0

    # Número de lutas, após a estreia ou após uma mudança de
    # divisão, em que o rating é considerado de baixa confiança
    # e recebe um K-factor maior para se recalibrar mais rápido.
    provisional_fights: int = 3

    # Meia-vida (em anos) do decaimento por recência (docs/
    # rating_methodology.md, seção 39). None reproduz exatamente
    # o Elo v0.1 sem recência — é o default, para que
    # src/param_sweep.py e qualquer outro consumidor de EloConfig
    # continue reproduzindo o baseline v0.1 sem alterações. A
    # configuração de produção (PRODUCTION_CONFIG, mais abaixo)
    # é que ativa a recência.
    half_life_years: float | None = None

    @property
    def label(self):
        """Identificador curto e determinístico, usado como sufixo
        de arquivo em runs com configuração não-default (sweep)."""

        label = (
            f"init{self.initial_rating:.0f}"
            f"_base{self.base_k:.0f}"
            f"_boost{self.boosted_k:.0f}"
            f"_scale{self.scale:.0f}"
            f"_prov{self.provisional_fights}"
        )

        if self.half_life_years is not None:
            label += f"_half{self.half_life_years:.0f}y"

        return label


DEFAULT_CONFIG = EloConfig()

# ============================================================
# Configuração de produção (v0.2).
#
# Decisão registrada em docs/rating_methodology.md, seção 44
# ("Promoção para produção: Elo + Recência"): a extensão de
# recência (half-life = 4 anos) foi validada fold a fold em
# src/combined_experiment.py e vence o Elo v0.1 puro em 5 dos 6
# folds do walk-forward, com melhora simultânea em Log Loss,
# Brier, AUC e Accuracy no agregado. É a única mudança promovida
# para o cálculo de `rating_before`/`rating_after` nesta versão.
#
# A extensão de performance (M4) permanece fora do rating em si:
# ela mede algo conceitualmente diferente (seção 31 — "rating"
# vs. "modelo de previsão") e será tratada na Fase 4 do roadmap
# (Previsão), reaproveitando `src/combined_experiment.py`.
# ============================================================

PRODUCTION_CONFIG = EloConfig(half_life_years=4.0)

# Aliases mantidos para retrocompatibilidade (valores do baseline
# v0.1, iguais aos defaults de EloConfig).
INITIAL_RATING = DEFAULT_CONFIG.initial_rating
BASE_K = DEFAULT_CONFIG.base_k
BOOSTED_K = DEFAULT_CONFIG.boosted_k
SCALE = DEFAULT_CONFIG.scale
PROVISIONAL_FIGHTS = DEFAULT_CONFIG.provisional_fights

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


def apply_recency(rating, last_fight_date, current_date, initial_rating, half_life_years):
    """
    Decai `rating` em direção a `initial_rating` conforme os dias
    de inatividade desde `last_fight_date` (docs/rating_methodology.md,
    seção 39.1):

        R_efetivo = R_inicial + (R_histórico - R_inicial) * 2^(-Δt / H)

    Sem luta anterior (estreante), não há período de inatividade:
    retorna o rating inalterado e fator 1.0. Mesma implementação
    validada em src/recency_sweep.py e src/combined_experiment.py.
    """

    if last_fight_date is None or pd.isna(last_fight_date):
        return rating, 0, 1.0

    elapsed_days = max(0, (current_date - last_fight_date).days)
    half_life_days = half_life_years * 365.25
    factor = 2 ** (-elapsed_days / half_life_days)
    effective_rating = initial_rating + (rating - initial_rating) * factor

    return effective_rating, elapsed_days, factor


def build_ratings(fights, config=DEFAULT_CONFIG):
    """
    Processa as lutas em ordem cronológica e calcula o Elo de
    cada lutador antes e depois de cada luta.

    `config` (EloConfig) carrega os parâmetros livres do Elo
    (initial_rating, base_k, boosted_k, scale, provisional_fights,
    half_life_years). O default (DEFAULT_CONFIG) reproduz
    exatamente o baseline v0.1, sem recência. Passar uma
    `EloConfig` diferente permite comparar configurações (ver
    src/param_sweep.py) sem duplicar esta função.

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
    - A estreia de um lutador e as `config.provisional_fights`
      lutas seguintes a uma mudança de divisão usam um K-factor
      maior (`config.boosted_k`), para permitir uma recalibração
      mais rápida em cenários de alta incerteza. Isso é um
      parâmetro a ser validado, não uma regra definitiva (seção
      37.4) — ver src/param_sweep.py para a comparação empírica.
    - Recência (seção 39): quando `config.half_life_years` não é
      None, o rating histórico de cada lutador é decaído em
      direção a `initial_rating` proporcionalmente à inatividade
      desde a última luta, ANTES de calcular `expected_score` e
      a atualização da luta atual. O decaimento nunca é reescrito
      retroativamente: `rating_after` desta luta é que passa a
      ser o novo rating histórico para a próxima (seção 2.1 —
      nenhuma informação futura contamina o passado).
    """

    fights = fights.sort_values(
        ["date", "fight_id"]
    ).reset_index(drop=True)

    historical_rating = {}
    fights_count = {}
    fights_since_switch = {}
    current_division = {}
    last_fight_date = {}

    def ensure_known(fighter_id):
        historical_rating.setdefault(fighter_id, config.initial_rating)
        fights_count.setdefault(fighter_id, 0)
        fights_since_switch.setdefault(fighter_id, 0)
        current_division.setdefault(fighter_id, None)
        last_fight_date.setdefault(fighter_id, None)

    def is_provisional(fighter_id):
        return (
            fights_count[fighter_id] < config.provisional_fights
            or fights_since_switch[fighter_id] < config.provisional_fights
        )

    def k_factor(fighter_id):
        return config.boosted_k if is_provisional(fighter_id) else config.base_k

    rows = []

    for fight in fights.itertuples(index=False):
        fighter_1_id = fight.fighter_1_id
        fighter_2_id = fight.fighter_2_id

        ensure_known(fighter_1_id)
        ensure_known(fighter_2_id)

        current_date = pd.Timestamp(fight.date)

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

        if config.half_life_years is None:
            rating_1_before, days_since_1, factor_1 = (
                historical_rating[fighter_1_id], 0, 1.0
            )
            rating_2_before, days_since_2, factor_2 = (
                historical_rating[fighter_2_id], 0, 1.0
            )
        else:
            rating_1_before, days_since_1, factor_1 = apply_recency(
                historical_rating[fighter_1_id],
                last_fight_date[fighter_1_id],
                current_date,
                config.initial_rating,
                config.half_life_years,
            )
            rating_2_before, days_since_2, factor_2 = apply_recency(
                historical_rating[fighter_2_id],
                last_fight_date[fighter_2_id],
                current_date,
                config.initial_rating,
                config.half_life_years,
            )

        expected_1 = expected_score(
            rating_1_before, rating_2_before, scale=config.scale
        )
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
            days_since,
            recency_factor,
            expected,
            k_used,
        ) in (
            (
                fighter_1_id, fighter_2_id, result_1,
                rating_1_before, rating_1_after, days_since_1, factor_1,
                expected_1, k_1,
            ),
            (
                fighter_2_id, fighter_1_id, result_2,
                rating_2_before, rating_2_after, days_since_2, factor_2,
                expected_2, k_2,
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
                "days_since_last_fight": days_since,
                "recency_factor": recency_factor,
                "expected_score": expected,
                "rating_before": rating_before,
                "rating_after": rating_after,
            })

        # `rating_after` (calculado a partir do rating EFETIVO,
        # pós-recência) passa a ser o novo rating histórico. O
        # decaimento nunca é uma perda permanente: ele só é
        # reaplicado a partir daqui na próxima luta, proporcional
        # a uma nova janela de inatividade.
        historical_rating[fighter_1_id] = rating_1_after
        historical_rating[fighter_2_id] = rating_2_after

        for fighter_id in (fighter_1_id, fighter_2_id):
            fights_count[fighter_id] += 1
            fights_since_switch[fighter_id] += 1
            last_fight_date[fighter_id] = current_date

            if division is not None:
                current_division[fighter_id] = division

    return pd.DataFrame(rows)


def validate_output(ratings, fights, initial_rating=DEFAULT_CONFIG.initial_rating, half_life_years=None):
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
        ratings.loc[first_fights, "rating_before"] != initial_rating
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
    #
    # Sem recência, rating_before == rating_after anterior,
    # exatamente. Com recência (half_life_years definido),
    # rating_before é o rating_after anterior DECAÍDO pela
    # inatividade (seção 39.1) — a igualdade exata só vale para
    # a primeira luta de cada lutador (sem luta anterior, sem
    # decaimento) ou lutas no mesmo dia (Δt = 0, fator = 1.0).
    ordered = ratings.sort_values(
        ["fighter_id", "date", "fight_id"]
    ).copy()

    ordered["date"] = pd.to_datetime(ordered["date"])

    previous_after = ordered.groupby("fighter_id")["rating_after"].shift(1)
    previous_date = ordered.groupby("fighter_id")["date"].shift(1)

    if half_life_years is None:
        expected_before = previous_after
    else:
        elapsed_days = (
            (ordered["date"] - previous_date).dt.days.clip(lower=0)
        )
        half_life_days = half_life_years * 365.25
        factor = 2 ** (-elapsed_days / half_life_days)
        expected_before = initial_rating + (previous_after - initial_rating) * factor

    mismatch = (
        previous_after.notna()
        & ~np.isclose(
            expected_before.astype(float),
            ordered["rating_before"].astype(float),
            atol=1e-6,
        )
    )

    if mismatch.any():
        errors.append(
            f"rating_before não corresponde ao rating_after "
            f"anterior (considerando recência): {mismatch.sum()}."
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


def load_and_merge_fights():
    """Lê fights.csv/events.csv, valida e devolve fights já com a
    coluna `date` do evento mesclada. Reaproveitado por `main()` e
    por `src/param_sweep.py`, para não duplicar o carregamento de
    dados a cada configuração testada."""

    fights = pd.read_csv(FIGHTS_FILE)
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

    return fights


def run_pipeline(fights, config=DEFAULT_CONFIG, output_file=OUTPUT_FILE, verbose=True):
    """
    Calcula os ratings de Elo para `fights` com `config`, valida e
    salva em `output_file`.

    Extraído de `main()` para ser reaproveitado por
    `src/param_sweep.py` (seção 53, item 5 do
    development_guideline.md), rodando a mesma lógica com
    configurações diferentes sem duplicar o script.
    """

    if verbose:
        print(f"\nCalculando ratings Elo (config={config.label})...")

    ratings = build_ratings(fights, config=config)

    validate_output(
        ratings,
        fights,
        initial_rating=config.initial_rating,
        half_life_years=config.half_life_years,
    )

    ratings = ratings.sort_values(
        ["date", "fight_id", "fighter_id"]
    ).reset_index(drop=True)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output_file, index=False)

    if verbose:
        print("\nArquivo salvo em:")
        print(output_file)

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

    return ratings


def main():
    print("=== BUILD FIGHTER RATINGS (Elo) ===")

    print("\nLendo fights.csv...")

    fights = load_and_merge_fights()

    print(f"Lutas carregadas: {len(fights)}")

    print(
        f"\nConfiguração de produção: {PRODUCTION_CONFIG.label} "
        "(Elo v0.1 + recência, ver docs/rating_methodology.md seção 44)"
    )

    run_pipeline(fights, config=PRODUCTION_CONFIG, output_file=OUTPUT_FILE, verbose=True)

    print("\n=== CONCLUÍDO ===")


if __name__ == "__main__":
    main()
