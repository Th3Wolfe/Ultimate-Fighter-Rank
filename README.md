# Ultimate Fighter Rank

Projeto de análise histórica e Machine Learning aplicado a dados do UFC.

## Objetivo

Construir uma base histórica confiável de lutadores, eventos, lutas e estatísticas por round para desenvolver um sistema de **avaliação histórica e ranking de lutadores**.

A ideia é acompanhar a evolução de cada lutador ao longo da carreira e utilizar seu histórico de desempenho para analisar confrontos e desenvolver modelos capazes de estimar resultados futuros.

## Fonte dos dados

Os dados utilizados como ponto de partida foram coletados pelo projeto **scrape_ufc_stats**, desenvolvido por **Greco1899**.

**Repositório:** https://github.com/Greco1899/scrape_ufc_stats

O **Ultimate Fighter Rank** é um projeto independente. O código de coleta não faz parte deste repositório; os dados são utilizados como entrada para as etapas próprias de limpeza, normalização, análise e Machine Learning.

## Estrutura

```text
ultimate-fighter-rank/

├── data/
│   ├── raw/
│   └── processed/
├── src/
├── tests/
├── notebooks/
├── docs/
├── README.md
└── .gitignore
```

## Pipeline

```text
Dados brutos
    ↓
Resolução de identidade
    ↓
Limpeza e normalização
    ↓
Base histórica
    ↓
Features temporais
    ↓
Fighter Rating
    ↓
Dataset de Machine Learning
    ↓
Modelo de previsão
    ↓
Backtest e avaliação
```

## Dados

A base local contém informações sobre:

* Eventos
* Lutadores
* Lutas
* Resultados
* Estatísticas por round
* Características físicas
* Métodos de vitória
* Classes de peso

Os dados brutos não são versionados no Git.

## Desenvolvimento

O projeto será desenvolvido de forma incremental, começando pela construção e validação da base histórica e posteriormente avançando para o sistema de rating e os modelos de Machine Learning.