from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


FILES = {
    "events": "ufc_event_details.csv",
    "fighter_details": "ufc_fighter_details.csv",
    "fighter_tott": "ufc_fighter_tott.csv",
    "fight_details": "ufc_fight_details.csv",
    "fight_results": "ufc_fight_results.csv",
    "fight_stats": "ufc_fight_stats.csv",
}


def load_raw():
    return {
        key: pd.read_csv(RAW / filename)
        for key, filename in FILES.items()
    }


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    data = load_raw()

    print("Arquivos carregados:")
    for name, df in data.items():
        print(f"  {name:15} {df.shape}")

    print("\nETL ainda não implementado.")
    print("Próximo passo: construir identity resolution e as tabelas normalizadas.")


if __name__ == "__main__":
    main()
