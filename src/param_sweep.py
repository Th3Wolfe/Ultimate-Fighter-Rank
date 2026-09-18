"""
Sweep de parâmetros do Elo v0.1 (roadmap, seção 53 do
development_guideline.md, item 5).

Contexto (docs/rating_methodology.md, seção 37.4): INITIAL_RATING,
BASE_K, BOOSTED_K, SCALE e PROVISIONAL_FIGHTS do baseline v0.1 são
um ponto de partida arbitrário, não uma escolha validada. Este
script testa configurações alternativas seguindo o princípio de
experimentação (seção 57):

    Hipótese → Feature/Modelo → Backtest temporal → Métrica →
    Comparação → Conclusão

Nenhuma lógica nova de Elo ou de avaliação é criada aqui: o script
apenas chama `build_fighter_ratings.build_ratings()` com uma
`EloConfig` diferente por rodada, e reaproveita o harness de
`evaluate_ratings.py` (walk-forward + split simples) para comparar
Log Loss / Brier / ROC AUC / Accuracy lado a lado — o mesmo que já
foi feito para Win Rate vs. Elo (seção 38 do rating_methodology.md).

Hipóteses testadas (cada uma isola UM parâmetro por vez, mantendo os
demais no valor do baseline v0.1, para que a comparação seja
interpretável):

1. K menor (mais conservador) melhora calibração, mesmo perdendo um
   pouco de capacidade de reação: BASE_K/BOOSTED_K em (16,32) e
   (24,48), contra o baseline (32,64).
2. BOOSTED_K para estreantes/mudança de divisão está alto (ou baixo)
   demais: testa BOOSTED_K = BASE_K (sem boost), e BOOSTED_K = 96
   (boost maior).
3. SCALE=400 (padrão do xadrez) não é necessariamente o melhor para
   MMA: testa SCALE=200 (mais sensível a diferenças de rating) e
   SCALE=600 (menos sensível).
4. PROVISIONAL_FIGHTS=3 pode ser curto ou longo demais: testa 1 e 5.

Saídas (não sobrescrevem os artefatos de produção
data/features/fighter_ratings.csv nem data/evaluation/metrics_summary.csv):
- data/features/param_sweep/fighter_ratings__<config>.csv (um por config)
- data/evaluation/param_sweep/summary.csv (métricas de todas as
  configs, todos os folds, lado a lado)

A conclusão (qual config generalizou melhor e por quê, seguindo a
seção 26 sobre evitar overfitting a um único período) deve ser
registrada em docs/rating_methodology.md depois de rodar este script
e inspecionar data/evaluation/param_sweep/summary.csv — isso não é
feito automaticamente pelo script.
"""

from pathlib import Path

import pandas as pd

import src.build_fighter_ratings as bfr
import src.evaluate_ratings as evr

ROOT_DIR = Path(__file__).resolve().parents[1]
SWEEP_FEATURES_DIR = ROOT_DIR / "data" / "features" / "param_sweep"
SWEEP_EVALUATION_DIR = ROOT_DIR / "data" / "evaluation" / "param_sweep"
SUMMARY_OUTPUT_FILE = SWEEP_EVALUATION_DIR / "summary.csv"

BASELINE = bfr.DEFAULT_CONFIG  # init1500_base32_boost64_scale400_prov3


def build_candidate_configs():
    """
    Uma config por hipótese (nomeada), variando um parâmetro por
    vez a partir do baseline v0.1. Retorna uma lista de tuplas
    (nome_da_hipotese, EloConfig).
    """

    return [
        ("baseline_v0.1", BASELINE),

        # Hipótese 1: K menor melhora calibração.
        ("k_menor_16_32", bfr.EloConfig(base_k=16.0, boosted_k=32.0)),
        ("k_menor_24_48", bfr.EloConfig(base_k=24.0, boosted_k=48.0)),

        # Hipótese 2: BOOSTED_K para estreantes está descalibrado.
        ("sem_boost_k_igual_base", bfr.EloConfig(boosted_k=BASELINE.base_k)),
        ("boost_k_maior_96", bfr.EloConfig(boosted_k=96.0)),

        # Hipótese 3: SCALE=400 (padrão do xadrez) não é ideal para MMA.
        ("scale_200", bfr.EloConfig(scale=200.0)),
        ("scale_600", bfr.EloConfig(scale=600.0)),

        # Hipótese 4: janela de PROVISIONAL_FIGHTS curta/longa demais.
        ("provisional_1", bfr.EloConfig(provisional_fights=1)),
        ("provisional_3", bfr.EloConfig(provisional_fights=3)),
        ("provisional_5", bfr.EloConfig(provisional_fights=5)),
        ("provisional_7", bfr.EloConfig(provisional_fights=7)),
        ("provisional_9", bfr.EloConfig(provisional_fights=9)),
    ]


def run_sweep():
    print("=== PARAM SWEEP: Elo v0.1 — teste de parâmetros ===")

    SWEEP_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLendo fights.csv / events.csv / fighter_history.csv...")

    # fights mesclado com a data do evento (necessário para
    # build_ratings, que processa as lutas em ordem cronológica).
    merged_fights = bfr.load_and_merge_fights()

    # Insumos crus, reaproveitados pelo harness de avaliação
    # (evaluate_ratings.build_evaluation_dataset faz seu próprio
    # merge com events.csv).
    raw_fights = pd.read_csv(bfr.FIGHTS_FILE)
    events = pd.read_csv(evr.EVENTS_FILE)
    events["date"] = pd.to_datetime(events["date"])
    history = pd.read_csv(evr.HISTORY_FILE)

    configs = build_candidate_configs()

    all_metrics = []

    for name, config in configs:
        print(f"\n--- Config: {name} ({config.label}) ---")

        ratings = bfr.run_pipeline(
            merged_fights,
            config=config,
            output_file=SWEEP_FEATURES_DIR / f"fighter_ratings__{name}.csv",
            verbose=False,
        )

        print(f"Ratings calculados: {len(ratings)} participações.")

        metrics = evr.evaluate_elo_config(
            raw_fights,
            events,
            history,
            ratings_file=SWEEP_FEATURES_DIR / f"fighter_ratings__{name}.csv",
            verbose=False,
        )

        metrics.insert(0, "config", name)
        metrics.insert(1, "base_k", config.base_k)
        metrics.insert(2, "boosted_k", config.boosted_k)
        metrics.insert(3, "scale", config.scale)
        metrics.insert(4, "provisional_fights", config.provisional_fights)

        all_metrics.append(metrics)

        # Feedback rápido: métricas do Elo desta config no fold
        # agregado do walk-forward (o mais robusto contra
        # overfitting a um único período — seção 26).
        elo_pooled = metrics[
            (metrics["model"] == "elo") & (metrics["fold"] == "pooled test")
        ]

        if not elo_pooled.empty:
            row = elo_pooled.iloc[0]
            print(
                f"  elo | pooled test | log_loss={row['log_loss']:.4f} "
                f"brier={row['brier_score']:.4f} auc={row['roc_auc']:.4f} "
                f"accuracy={row['accuracy']:.4f}"
            )

    summary = pd.concat(all_metrics, ignore_index=True)
    summary.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    print(f"\nResumo completo (todos os folds, todas as configs) salvo em:")
    print(SUMMARY_OUTPUT_FILE)

    print_comparison_table(summary)

    return summary


def print_comparison_table(summary):
    """
    Compara as configs pelo fold mais robusto contra overfitting a
    um único período: o agregado do walk-forward ("pooled test"),
    olhando só o modelo Elo (Win Rate é idêntico em todas as
    configs, pois não depende dos parâmetros do Elo).
    """

    elo_pooled = summary[
        (summary["model"] == "elo") & (summary["fold"] == "pooled test")
    ].sort_values("log_loss")

    print("\n=== COMPARAÇÃO (Elo, walk-forward agregado, ordenado por Log Loss) ===")

    header = (
        f"{'config':<24}{'base_k':>8}{'boosted_k':>10}{'scale':>8}"
        f"{'prov':>6}{'log_loss':>12}{'brier':>10}{'auc':>10}{'accuracy':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in elo_pooled.itertuples(index=False):
        print(
            f"{row.config:<24}{row.base_k:>8.0f}{row.boosted_k:>10.0f}"
            f"{row.scale:>8.0f}{row.provisional_fights:>6}"
            f"{row.log_loss:>12.4f}{row.brier_score:>10.4f}"
            f"{row.roc_auc:>10.4f}{row.accuracy:>10.4f}"
        )

    print(
        "\nLembrete (seção 26 do development_guideline.md): não escolher a "
        "config vencedora só por este agregado — inspecionar também "
        "data/evaluation/param_sweep/summary.csv fold a fold, para "
        "confirmar que o resultado não vem de um único período."
    )


def main():
    run_sweep()


if __name__ == "__main__":
    main()
