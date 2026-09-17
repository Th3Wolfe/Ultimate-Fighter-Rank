from pathlib import Path
import re

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


# ============================================================
# UTILITÁRIOS
# ============================================================

def normalize_text(value):
    """
    Normaliza textos para comparação.

    Exemplos:
        "  John Doe  " -> "john doe"
        "John   Doe"   -> "john doe"
    """
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    return value


def extract_ufcstats_id(url):
    """
    Extrai o ID de 16 caracteres presente nas URLs do UFCStats.
    """

    if pd.isna(url):
        return None

    match = re.search(
        r"/([a-f0-9]{16})(?:$|/)",
        str(url).strip(),
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).lower()

    return None


def synthetic_event_id(event_name):
    """
    Gera um ID determinístico para eventos que não possuem
    registro em ufc_event_details.csv.

    O ID é prefixado para deixar claro que é sintético.
    """

    normalized = normalize_text(event_name)

    import hashlib

    digest = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:16]

    return f"synthetic_{digest}"


# ============================================================
# ALIASES DE EVENTOS
# ============================================================

EVENT_ALIASES = {
    normalize_text(
        "UFC Fight Night: Lopes vs. Silva"
    ): "Noche UFC: Lopes vs. Silva",

    normalize_text(
        "UFC Fight Night: Grasso vs. Shevchenko 2"
    ): "Noche UFC: Grasso vs. Shevchenko 2",
}

# ============================================================
# METADADOS DE EVENTOS AUSENTES
# ============================================================

MISSING_EVENT_METADATA = {
    normalize_text("UFC - Road to UFC 4.6"): {
        "date": "August 22, 2025",
        "location": "Shanghai, Hebei, China",
    },
}


def canonical_event_name(event_name):
    """
    Retorna o nome canônico do evento.
    """

    if pd.isna(event_name):
        return ""

    original = str(event_name).strip()

    normalized = normalize_text(original)

    return EVENT_ALIASES.get(
        normalized,
        original,
    )


# ============================================================
# ALIASES DE LUTADORES
# ============================================================

FIGHTER_ALIASES = {
    normalize_text(
        "Bibulatov Magomed"
    ): "Magomed Bibulatov",

    normalize_text(
        "Kai Kamaka"
    ): "Kai Kamaka III",

    normalize_text(
        "Patricio Freire"
    ): "Patricio Pitbull",

    normalize_text(
        "Rafael Cerquiera"
    ): "Rafael Cerqueira",
}


# ============================================================
# LUTADORES
# ============================================================

def build_fighter_name_map(fighters):
    """
    Cria mapa de nomes normalizados -> fighter_id.

    Inclui:
    - FIRST + LAST
    - LAST + FIRST
    - FIRST
    - aliases explícitos conhecidos
    """

    fighter_name_map = {}

    for _, row in fighters.iterrows():
        fighter_id = row["fighter_id"]

        first = (
            ""
            if pd.isna(row["FIRST"])
            else str(row["FIRST"]).strip()
        )

        last = (
            ""
            if pd.isna(row["LAST"])
            else str(row["LAST"]).strip()
        )

        full_name = f"{first} {last}".strip()

        candidates = [
            full_name,
            f"{last} {first}".strip(),
        ]

        for candidate in candidates:
            normalized = normalize_text(candidate)

            if normalized:
                fighter_name_map.setdefault(
                    normalized,
                    fighter_id,
                )

    # Aliases explícitos
    for alias, canonical_name in FIGHTER_ALIASES.items():
        canonical_normalized = normalize_text(
            canonical_name
        )

        if canonical_normalized in fighter_name_map:
            fighter_name_map[alias] = (
                fighter_name_map[canonical_normalized]
            )

    return fighter_name_map


def resolve_fighter_id(
    fighter_name,
    fighter_name_map,
):
    """
    Resolve fighter_id a partir do nome usado nas estatísticas.
    """

    normalized = normalize_text(fighter_name)

    if not normalized:
        return None

    return fighter_name_map.get(
        normalized
    )


def build_fighters(fighters_raw):
    """
    Constrói tabela processada de lutadores.
    """

    fighters = fighters_raw.copy()

    fighters["fighter_id"] = fighters[
        "URL"
    ].apply(extract_ufcstats_id)

    fighters["first_name"] = fighters[
        "FIRST"
    ]

    fighters["last_name"] = fighters[
        "LAST"
    ]

    fighters["nickname"] = fighters[
        "NICKNAME"
    ]

    fighters["name"] = (
        fighters["FIRST"]
        .fillna("")
        .astype(str)
        .str.strip()
        + " "
        + fighters["LAST"]
        .fillna("")
        .astype(str)
        .str.strip()
    ).str.strip()

    fighters = fighters[
        [
            "fighter_id",
            "name",
            "first_name",
            "last_name",
            "nickname",
        ]
    ].copy()

    fighters = fighters.dropna(
        subset=["fighter_id"]
    )

    fighters = fighters.drop_duplicates(
        subset=["fighter_id"],
        keep="first",
    )

    return fighters


# ============================================================
# EVENTOS
# ============================================================

def build_events(
    events_raw,
    fight_details_raw,
):
    """
    Constrói tabela de eventos.

    Inclui eventos presentes nos detalhes de lutas mas ausentes
    no arquivo de eventos.

    Para:
        UFC - Road to UFC 4.6

    é criado um ID sintético determinístico.
    """

    events = events_raw.copy()
    details = fight_details_raw.copy()

    events["event_name"] = (
        events["EVENT"]
        .astype(str)
        .str.strip()
        .apply(canonical_event_name)
    )

    events["event_id"] = events[
        "URL"
    ].apply(extract_ufcstats_id)

    events["date"] = events[
        "DATE"
    ]

    events["location"] = events[
        "LOCATION"
    ]

    events = events[
        [
            "event_id",
            "event_name",
            "date",
            "location",
        ]
    ].copy()

    events = events.dropna(
        subset=["event_id"]
    )

    events = events.drop_duplicates(
        subset=["event_name"],
        keep="first",
    )

    # --------------------------------------------------------
    # Eventos utilizados pelas lutas
    # --------------------------------------------------------

    detail_events = (
        details["EVENT"]
        .astype(str)
        .str.strip()
        .apply(canonical_event_name)
        .drop_duplicates()
    )

    existing_events = set(
        events["event_name"]
    )

    missing_events = [
        event
        for event in detail_events
        if event not in existing_events
    ]

    for event_name in missing_events:
        metadata = MISSING_EVENT_METADATA.get(
            normalize_text(event_name),
            {}
        )

        events.loc[len(events)] = {
            "event_id": synthetic_event_id(
                event_name
            ),
            "event_name": event_name,
            "date": metadata.get("date"),
            "location": metadata.get("location"),
        }

    events = events.drop_duplicates(
        subset=["event_name"],
        keep="first",
    )

    return events


# ============================================================
# FIGHTS
# ============================================================

def build_fights(
    fight_details_raw,
    fight_results_raw,
    events,
    fighter_name_map,
):
    """
    Constrói tabela de lutas.

    Uma luta é identificada pelo fight_id da URL do UFCStats.

    Os arquivos de detalhes/resultados possuem 25 duplicatas
    de fights sob nomes alternativos de eventos. Elas são
    removidas pelo fight_id.
    """

    details = fight_details_raw.copy()
    results = fight_results_raw.copy()

    # --------------------------------------------------------
    # Normalização
    # --------------------------------------------------------

    details["event_name"] = (
        details["EVENT"]
        .astype(str)
        .str.strip()
        .apply(canonical_event_name)
    )

    details["bout"] = (
        details["BOUT"]
        .astype(str)
        .str.strip()
    )

    details["fight_id"] = details[
        "URL"
    ].apply(extract_ufcstats_id)

    results["event_name"] = (
        results["EVENT"]
        .astype(str)
        .str.strip()
        .apply(canonical_event_name)
    )

    results["bout"] = (
        results["BOUT"]
        .astype(str)
        .str.strip()
    )

    results["fight_id"] = results[
        "URL"
    ].apply(extract_ufcstats_id)

    # --------------------------------------------------------
    # Remove duplicatas por fight_id
    # --------------------------------------------------------

    details = details.dropna(
        subset=["fight_id"]
    )

    details = details.drop_duplicates(
        subset=["fight_id"],
        keep="first",
    )

    results = results.dropna(
        subset=["fight_id"]
    )

    results = results.drop_duplicates(
        subset=["fight_id"],
        keep="first",
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    fights = details[
        [
            "fight_id",
            "event_name",
            "bout",
        ]
    ].copy()

    result_columns = [
        "fight_id",
        "OUTCOME",
        "WEIGHTCLASS",
        "METHOD",
        "ROUND",
        "TIME",
        "TIME FORMAT",
        "REFEREE",
        "DETAILS",
    ]

    fights = fights.merge(
        results[result_columns],
        on="fight_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Event ID
    # --------------------------------------------------------

    event_map = (
        events[
            [
                "event_id",
                "event_name",
            ]
        ]
        .drop_duplicates(
            subset=["event_name"]
        )
        .set_index("event_name")[
            "event_id"
        ]
        .to_dict()
    )

    fights["event_id"] = fights[
        "event_name"
    ].map(event_map)

    # --------------------------------------------------------
    # Fighters
    # --------------------------------------------------------
    #
    # BOUT normalmente:
    #
    # Fighter A vs. Fighter B
    #
    # Não usamos split ingênuo em todos os casos sem validar
    # o formato.
    # --------------------------------------------------------

    fighter_1 = []
    fighter_2 = []

    for bout in fights["bout"]:
        parts = re.split(
            r"\s+vs\.?\s+",
            str(bout),
            maxsplit=1,
            flags=re.IGNORECASE,
        )

        if len(parts) == 2:
            fighter_1.append(
                parts[0].strip()
            )
            fighter_2.append(
                parts[1].strip()
            )
        else:
            fighter_1.append(None)
            fighter_2.append(None)

    fights["fighter_1_name"] = fighter_1
    fights["fighter_2_name"] = fighter_2

    fights["fighter_1_id"] = fights[
        "fighter_1_name"
    ].apply(
        lambda name: resolve_fighter_id(
            name,
            fighter_name_map,
        )
    )

    fights["fighter_2_id"] = fights[
        "fighter_2_name"
    ].apply(
        lambda name: resolve_fighter_id(
            name,
            fighter_name_map,
        )
    )

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    outcome_parts = (
        fights["OUTCOME"]
        .fillna("")
        .astype(str)
        .str.split("/")
    )

    fights["fighter_1_result"] = (
        outcome_parts
        .apply(
            lambda x: x[0]
            if len(x) > 0
            else None
        )
    )

    fights["fighter_2_result"] = (
        outcome_parts
        .apply(
            lambda x: x[1]
            if len(x) > 1
            else None
        )
    )

    # --------------------------------------------------------
    # Colunas finais
    # --------------------------------------------------------

    fights = fights[
        [
            "fight_id",
            "event_id",
            "event_name",
            "bout",
            "fighter_1_id",
            "fighter_2_id",
            "fighter_1_name",
            "fighter_2_name",
            "fighter_1_result",
            "fighter_2_result",
            "WEIGHTCLASS",
            "METHOD",
            "ROUND",
            "TIME",
            "TIME FORMAT",
            "REFEREE",
            "DETAILS",
        ]
    ].copy()

    fights = fights.rename(
        columns={
            "WEIGHTCLASS": "weight_class",
            "METHOD": "method",
            "ROUND": "ending_round",
            "TIME": "ending_time",
            "TIME FORMAT": "time_format",
            "REFEREE": "referee",
            "DETAILS": "details",
        }
    )

    return fights


# ============================================================
# ROUND STATS
# ============================================================

def build_round_stats(
    round_stats_raw,
    fight_details_raw,
    fights,
    fighter_name_map,
):
    """
    Constrói estatísticas por lutador e round.

    Existem três situações importantes:

    1. EVENT+BOUT -> um fight_id
       Mapeamento direto.

    2. EVENT+BOUT aparece em nomes de eventos diferentes
       que representam a mesma luta.
       Depois do mapeamento, as linhas redundantes são removidas.

    3. EVENT+BOUT -> múltiplos fight_ids.
       Caso histórico observado em UFC - Ultimate Japan:
           Kazushi Sakuraba vs. Marcus Silveira

       Nesse caso, os blocos de estatísticas são associados
       aos fight_ids na mesma ordem em que aparecem nos
       fight_details.
    """

    stats = round_stats_raw.copy()
    details = fight_details_raw.copy()

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    stats["EVENT_NORMALIZED"] = (
        stats["EVENT"]
        .astype(str)
        .str.strip()
        .apply(canonical_event_name)
    )

    details["EVENT_NORMALIZED"] = (
        details["EVENT"]
        .astype(str)
        .str.strip()
        .apply(canonical_event_name)
    )

    stats["BOUT_NORMALIZED"] = (
        stats["BOUT"]
        .astype(str)
        .str.strip()
        .apply(normalize_text)
    )

    details["BOUT_NORMALIZED"] = (
        details["BOUT"]
        .astype(str)
        .str.strip()
        .apply(normalize_text)
    )

    # --------------------------------------------------------
    # REMOVE PLACEHOLDERS VAZIOS
    #
    # O arquivo possui 42 linhas sem FIGHTER/ROUND.
    # São placeholders e não representam estatísticas.
    # --------------------------------------------------------

    stats = stats[
        stats["FIGHTER"].notna()
    ].copy()

    # --------------------------------------------------------
    # ORDEM ORIGINAL
    # --------------------------------------------------------

    stats["_stats_order"] = range(
        len(stats)
    )

    details["_details_order"] = range(
        len(details)
    )

    # --------------------------------------------------------
    # FIGHT ID
    # --------------------------------------------------------

    details["fight_id"] = details[
        "URL"
    ].apply(extract_ufcstats_id)

    details = details[
        details["fight_id"].notna()
    ].copy()

    details = details.sort_values(
        "_details_order"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # MAPA EVENT+BOUT -> FIGHTS
    # --------------------------------------------------------

    fight_groups = (
        details[
            [
                "EVENT_NORMALIZED",
                "BOUT_NORMALIZED",
                "fight_id",
                "_details_order",
            ]
        ]
        .drop_duplicates(
            subset=[
                "EVENT_NORMALIZED",
                "BOUT_NORMALIZED",
                "fight_id",
            ]
        )
        .groupby(
            [
                "EVENT_NORMALIZED",
                "BOUT_NORMALIZED",
            ],
            sort=False,
        )
    )

    stats["fight_id"] = pd.NA

    # --------------------------------------------------------
    # RESOLUÇÃO
    # --------------------------------------------------------

    for (
        event_name,
        bout_name,
    ), group in fight_groups:

        stat_indices = stats.index[
            (
                stats["EVENT_NORMALIZED"]
                == event_name
            )
            & (
                stats["BOUT_NORMALIZED"]
                == bout_name
            )
        ].tolist()

        fight_ids = group[
            "fight_id"
        ].tolist()

        # ----------------------------------------------------
        # CASO NORMAL
        # ----------------------------------------------------

        if len(fight_ids) == 1:

            stats.loc[
                stat_indices,
                "fight_id",
            ] = fight_ids[0]

            continue

        # ----------------------------------------------------
        # CASO AMBÍGUO
        #
        # O mesmo EVENT+BOUT possui mais de uma luta.
        #
        # Exemplo:
        #
        #   ec1bda... -> Submission 3:44
        #   2750ac... -> NC 1:51
        #
        # Os blocos aparecem na mesma ordem nos arquivos.
        # ----------------------------------------------------

        subset = (
            stats.loc[
                stat_indices,
                [
                    "_stats_order",
                    "ROUND",
                    "FIGHTER",
                ],
            ]
            .sort_values(
                "_stats_order"
            )
        )

        total_rows = len(subset)

        if (
            total_rows % len(fight_ids)
            != 0
        ):
            raise RuntimeError(
                "Não foi possível separar "
                "múltiplos fights para "
                f"EVENT+BOUT: "
                f"{event_name} | "
                f"{bout_name}. "
                f"Stats={total_rows}, "
                f"fights={len(fight_ids)}."
            )

        rows_per_fight = (
            total_rows // len(fight_ids)
        )

        for position, fight_id in enumerate(
            fight_ids
        ):

            start = (
                position
                * rows_per_fight
            )

            end = (
                start
                + rows_per_fight
            )

            selected_indices = (
                subset.iloc[
                    start:end
                ].index
            )

            stats.loc[
                selected_indices,
                "fight_id",
            ] = fight_id

    # --------------------------------------------------------
    # FIGHTER ID
    # --------------------------------------------------------

    stats["fighter_id"] = stats[
        "FIGHTER"
    ].apply(
        lambda name: resolve_fighter_id(
            name,
            fighter_name_map,
        )
    )

    # --------------------------------------------------------
    # ROUND
    # --------------------------------------------------------

    stats["round"] = (
        stats["ROUND"]
        .astype(str)
        .str.extract(
            r"(\d+)"
        )[0]
    )

    stats["round"] = pd.to_numeric(
        stats["round"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # VALIDAÇÕES LOCAIS
    # --------------------------------------------------------

    unresolved_fights = stats[
        stats["fight_id"].isna()
    ]

    if not unresolved_fights.empty:
        raise RuntimeError(
            "Existem round stats sem "
            f"fight_id resolvido: "
            f"{len(unresolved_fights)}"
        )

    unresolved_fighters = stats[
        stats["fighter_id"].isna()
    ]

    if not unresolved_fighters.empty:
        names = (
            unresolved_fighters[
                "FIGHTER"
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Existem lutadores não "
            "resolvidos nos round stats: "
            f"{names}"
        )

    invalid_rounds = stats[
        stats["round"].isna()
    ]

    if not invalid_rounds.empty:
        raise RuntimeError(
            "Existem round stats com "
            f"round inválido: "
            f"{len(invalid_rounds)}"
        )

    # --------------------------------------------------------
    # DETECTA DUPLICATAS CONFLITANTES
    #
    # A identidade correta é:
    #
    #   fight_id + fighter_id + round
    #
    # Não usamos EVENT+BOUT porque existem duas lutas
    # historicamente com a mesma chave textual.
    # --------------------------------------------------------

    identity_columns = [
        "fight_id",
        "fighter_id",
        "round",
    ]

    stat_columns = [
        "KD",
        "SIG.STR.",
        "SIG.STR. %",
        "TOTAL STR.",
        "TD",
        "TD %",
        "SUB.ATT",
        "REV.",
        "CTRL",
        "HEAD",
        "BODY",
        "LEG",
        "DISTANCE",
        "CLINCH",
        "GROUND",
    ]

    duplicate_check = (
        stats.groupby(
            identity_columns,
            dropna=False,
        )[stat_columns]
        .nunique(
            dropna=False
        )
        .max(axis=1)
    )

    conflicting_duplicates = (
        duplicate_check[
            duplicate_check > 1
        ]
    )

    if not conflicting_duplicates.empty:

        print(
            "\nERRO — DUPLICATAS COM "
            "ESTATÍSTICAS CONFLITANTES:"
        )

        print(
            conflicting_duplicates
        )

        raise RuntimeError(
            "Existem round stats "
            "duplicados com valores "
            "diferentes."
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATAS REDUNDANTES
    # --------------------------------------------------------

    before_dedup = len(stats)

    stats = stats.drop_duplicates(
        subset=identity_columns,
        keep="first",
    ).copy()

    removed_duplicates = (
        before_dedup
        - len(stats)
    )

    print(
        "\nDuplicatas redundantes "
        "removidas: "
        f"{removed_duplicates}"
    )

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    stats = stats[
        [
            "fight_id",
            "fighter_id",
            "round",
            "FIGHTER",
            "KD",
            "SIG.STR.",
            "SIG.STR. %",
            "TOTAL STR.",
            "TD",
            "TD %",
            "SUB.ATT",
            "REV.",
            "CTRL",
            "HEAD",
            "BODY",
            "LEG",
            "DISTANCE",
            "CLINCH",
            "GROUND",
        ]
    ].copy()

    stats = stats.rename(
        columns={
            "FIGHTER": "fighter_name",
            "KD": "knockdowns",
            "SIG.STR.": "significant_strikes",
            "SIG.STR. %": "significant_strikes_pct",
            "TOTAL STR.": "total_strikes",
            "TD": "takedowns",
            "TD %": "takedown_pct",
            "SUB.ATT": "submission_attempts",
            "REV.": "reversals",
            "CTRL": "control_time",
            "HEAD": "head_strikes",
            "BODY": "body_strikes",
            "LEG": "leg_strikes",
            "DISTANCE": "distance_strikes",
            "CLINCH": "clinch_strikes",
            "GROUND": "ground_strikes",
        }
    )

    return stats


# ============================================================
# VALIDAÇÃO
# ============================================================

def validate_data(
    events,
    fighters,
    fights,
    round_stats,
):
    """
    Validação geral do dataset processado.
    """

    issues = []

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    print("\n=== DATASET ===")

    print(
        f"Events:       {len(events)}"
    )

    print(
        f"Fighters:     {len(fighters)}"
    )

    print(
        f"Fights:       {len(fights)}"
    )

    print(
        f"Round stats:  {len(round_stats)}"
    )

    # --------------------------------------------------------
    # IDS DOS FIGHTERS
    # --------------------------------------------------------

    unresolved_fighter_1 = fights[
        fights["fighter_1_id"].isna()
    ]

    unresolved_fighter_2 = fights[
        fights["fighter_2_id"].isna()
    ]

    print(
        "\nFight fighter 1 unresolved: "
        f"{len(unresolved_fighter_1)}"
    )

    print(
        "Fight fighter 2 unresolved: "
        f"{len(unresolved_fighter_2)}"
    )

    if not unresolved_fighter_1.empty:
        issues.append(
            "Fight fighter 1 unresolved"
        )

    if not unresolved_fighter_2.empty:
        issues.append(
            "Fight fighter 2 unresolved"
        )

    # --------------------------------------------------------
    # ROUND STATS
    # --------------------------------------------------------

    unresolved_stats_fighters = (
        round_stats[
            round_stats["fighter_id"].isna()
        ]
    )

    print(
        "Round fighters unresolved: "
        f"{len(unresolved_stats_fighters)}"
    )

    if not unresolved_stats_fighters.empty:
        issues.append(
            "Round fighters unresolved"
        )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    result_counts = (
        fights.groupby(
            [
                "fighter_1_result",
                "fighter_2_result",
            ]
        )
        .size()
    )

    w_l = result_counts.get(
        ("W", "L"),
        0,
    )

    l_w = result_counts.get(
        ("L", "W"),
        0,
    )

    d_d = result_counts.get(
        ("D", "D"),
        0,
    )

    nc_nc = result_counts.get(
        ("NC", "NC"),
        0,
    )

    print(
        f"\nW/L: {w_l}"
    )

    print(
        f"L/W: {l_w}"
    )

    print(
        f"D/D: {d_d}"
    )

    print(
        f"NC/NC: {nc_nc}"
    )

    # --------------------------------------------------------
    # DUPLICATE FIGHTS
    # --------------------------------------------------------

    duplicate_fights = fights[
        fights["fight_id"].duplicated(
            keep=False
        )
    ]

    print(
        "\nDuplicate fights: "
        f"{len(duplicate_fights)}"
    )

    if not duplicate_fights.empty:
        issues.append(
            "Duplicate fights"
        )

    # --------------------------------------------------------
    # FIGHTS SEM EVENT
    # --------------------------------------------------------

    fights_without_event = fights[
        fights["event_id"].isna()
    ]

    print(
        "Fights without event: "
        f"{len(fights_without_event)}"
    )

    if not fights_without_event.empty:
        issues.append(
            "Fights without event"
        )

    # --------------------------------------------------------
    # STATS SEM FIGHT
    # --------------------------------------------------------

    fight_ids = set(
        fights["fight_id"]
    )

    stats_without_fight = round_stats[
        ~round_stats["fight_id"].isin(
            fight_ids
        )
    ]

    print(
        "Stats without fight: "
        f"{len(stats_without_fight)}"
    )

    if not stats_without_fight.empty:
        issues.append(
            "Stats without fight"
        )

    # --------------------------------------------------------
    # DUPLICIDADE DE IDENTIDADE
    # --------------------------------------------------------

    identity_columns = [
        "fight_id",
        "fighter_id",
        "round",
    ]

    duplicate_stats = round_stats[
        round_stats.duplicated(
            subset=identity_columns,
            keep=False,
        )
    ]

    print(
        "Duplicate stat rows: "
        f"{len(duplicate_stats)}"
    )

    if not duplicate_stats.empty:
        issues.append(
            "Duplicate round-stat identities"
        )

    # --------------------------------------------------------
    # IDS NULOS
    # --------------------------------------------------------

    null_fight_ids = fights[
        fights["fight_id"].isna()
    ]

    if not null_fight_ids.empty:
        issues.append(
            "Null fight IDs"
        )

    null_fighter_ids = fighters[
        fighters["fighter_id"].isna()
    ]

    if not null_fighter_ids.empty:
        issues.append(
            "Null fighter IDs"
        )

    null_event_ids = events[
        events["event_id"].isna()
    ]

    if not null_event_ids.empty:
        issues.append(
            "Null event IDs"
        )

    # --------------------------------------------------------
    # RELACIONAMENTO FIGHT -> EVENT
    # --------------------------------------------------------

    event_ids = set(
        events["event_id"]
    )

    invalid_event_refs = fights[
        ~fights["event_id"].isin(
            event_ids
        )
    ]

    if not invalid_event_refs.empty:
        issues.append(
            "Invalid event references"
        )

    # --------------------------------------------------------
    # RELACIONAMENTO FIGHT -> FIGHTER
    # --------------------------------------------------------

    fighter_ids = set(
        fighters["fighter_id"]
    )

    invalid_fighter_refs = fights[
        (
            ~fights["fighter_1_id"].isin(
                fighter_ids
            )
        )
        |
        (
            ~fights["fighter_2_id"].isin(
                fighter_ids
            )
        )
    ]

    if not invalid_fighter_refs.empty:
        issues.append(
            "Invalid fighter references"
        )

    # --------------------------------------------------------
    # RELACIONAMENTO STATS -> FIGHTER
    # --------------------------------------------------------

    invalid_stats_fighters = round_stats[
        ~round_stats["fighter_id"].isin(
            fighter_ids
        )
    ]

    if not invalid_stats_fighters.empty:
        issues.append(
            "Invalid round-stat fighter references"
        )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    print("\n=== DATA QUALITY REPORT ===")

    if issues:

        print(
            f"Dataset invalid: "
            f"{len(issues)} issue(s)."
        )

        for issue in issues:
            print(
                f" - {issue}"
            )

        return False

    print(
        "Dataset valid: "
        "no integrity issues found."
    )

    return True


# ============================================================
# SALVAMENTO
# ============================================================

def save_processed_data(
    events,
    fighters,
    fights,
    round_stats,
):
    """
    Salva os datasets processados somente depois
    da validação.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        PROCESSED_DIR / "events.csv",
        index=False,
    )

    fighters.to_csv(
        PROCESSED_DIR / "fighters.csv",
        index=False,
    )

    fights.to_csv(
        PROCESSED_DIR / "fights.csv",
        index=False,
    )

    round_stats.to_csv(
        PROCESSED_DIR / "round_stats.csv",
        index=False,
    )

    print(
        "\nArquivos processados salvos em:"
    )

    print(
        PROCESSED_DIR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=== ULTIMATE FIGHTER RANK ==="
    )

    print(
        "Construindo dataset..."
    )

    # --------------------------------------------------------
    # LEITURA
    # --------------------------------------------------------

    events_raw = pd.read_csv(
        RAW_DIR / "ufc_event_details.csv"
    )

    fighters_raw = pd.read_csv(
        RAW_DIR / "ufc_fighter_details.csv"
    )

    fight_details_raw = pd.read_csv(
        RAW_DIR / "ufc_fight_details.csv"
    )

    fight_results_raw = pd.read_csv(
        RAW_DIR / "ufc_fight_results.csv"
    )

    round_stats_raw = pd.read_csv(
        RAW_DIR / "ufc_fight_stats.csv"
    )

    print("\n=== RAW ===")

    print(
        f"Events raw:       {len(events_raw)}"
    )

    print(
        f"Fighters raw:     {len(fighters_raw)}"
    )

    print(
        f"Fight details:    {len(fight_details_raw)}"
    )

    print(
        f"Fight results:    {len(fight_results_raw)}"
    )

    print(
        f"Round stats raw:  {len(round_stats_raw)}"
    )

    # --------------------------------------------------------
    # FIGHTERS
    # --------------------------------------------------------

    fighters = build_fighters(
        fighters_raw
    )

    fighter_name_map = (
        build_fighter_name_map(
            fighters_raw.assign(
                fighter_id=
                fighters_raw[
                    "URL"
                ].apply(
                    extract_ufcstats_id
                )
            )
        )
    )

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    events = build_events(
        events_raw,
        fight_details_raw,
    )

    # --------------------------------------------------------
    # FIGHTS
    # --------------------------------------------------------

    fights = build_fights(
        fight_details_raw,
        fight_results_raw,
        events,
        fighter_name_map,
    )

    # --------------------------------------------------------
    # ROUND STATS
    # --------------------------------------------------------

    round_stats = build_round_stats(
        round_stats_raw,
        fight_details_raw,
        fights,
        fighter_name_map,
    )

    # --------------------------------------------------------
    # VALIDAÇÃO
    # --------------------------------------------------------

    is_valid = validate_data(
        events,
        fighters,
        fights,
        round_stats,
    )

    if not is_valid:

        raise RuntimeError(
            "\nDataset contém problemas "
            "de integridade. "
            "Os arquivos processados "
            "não foram sobrescritos."
        )

    # --------------------------------------------------------
    # SALVA SOMENTE SE VÁLIDO
    # --------------------------------------------------------

    save_processed_data(
        events,
        fighters,
        fights,
        round_stats,
    )

    print(
        "\n=== CONCLUÍDO ==="
    )


if __name__ == "__main__":
    main()