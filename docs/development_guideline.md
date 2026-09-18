# Ultimate Fighter Rank — Development Guideline

**Projeto:** Ultimate Fighter Rank
**Repositório:** https://github.com/Th3Wolfe/Ultimate-Fighter-Rank
**Tipo:** Projeto de Data Engineering + Machine Learning + Web Application
**Status:** Em desenvolvimento
**Documento:** Guia completo de desenvolvimento e continuidade
**Versão:** 1.0

---

# 1. Propósito deste documento

Este documento é o guia técnico central para o desenvolvimento do **Ultimate Fighter Rank**.

Ele existe para permitir que qualquer desenvolvedor, colaborador ou sistema de IA consiga continuar o projeto sem depender do histórico desta conversa.

Antes de alterar qualquer parte do projeto:

1. Ler este documento.
2. Ler `docs/rating_methodology.md`.
3. Inspecionar a estrutura atual do repositório.
4. Verificar o estado do Git.
5. Validar quais etapas já foram concluídas.
6. Não refazer etapas concluídas sem motivo técnico.
7. Não alterar decisões metodológicas importantes sem registrar a mudança.

Este documento deve ser atualizado sempre que uma decisão estrutural importante for tomada.

---

# 2. Visão do projeto

O **Ultimate Fighter Rank** é um projeto de análise histórica do UFC baseado em dados públicos de lutas.

O objetivo é construir uma infraestrutura confiável capaz de:

- armazenar o histórico de lutadores;
- armazenar eventos;
- armazenar lutas;
- armazenar estatísticas por round;
- reconstruir o histórico de cada lutador;
- estimar a força competitiva dos lutadores ao longo do tempo;
- prever resultados de confrontos;
- identificar características de estilo;
- estudar interações entre estilos;
- gerar rankings;
- permitir análises históricas;
- futuramente simular confrontos.

O projeto não deve ser tratado apenas como um CRUD de UFC.

A intenção é construir um sistema de dados e ML que permita responder perguntas como:

> "Qual era a força estimada deste lutador naquele momento?"

> "Como a avaliação dele mudou após determinada luta?"

> "Quais características definem seu estilo?"

> "Como o estilo de um lutador interage com o estilo de outro?"

> "Qual era a probabilidade estimada de A vencer B utilizando apenas informações disponíveis antes da luta?"

---

# 3. Visão geral da arquitetura

A arquitetura conceitual do projeto é:

```text
                        UFC DATA
                           │
                           ▼
                   DATA ENGINEERING
                           │
                           ▼
                    PROCESSED DATA
                           │
                           ▼
                      FEATURES
                           │
                           ▼
                  HISTORICAL STATE
                  ┌────────┼────────┐
                  │        │        │
                  ▼        ▼        ▼
               RATING   METRICS   STYLE
                  │        │        │
                  └────────┼────────┘
                           ▼
                     MATCHUP MODEL
                           │
                           ▼
                      PREDICTION
                    ┌──────┼──────┐
                    ▼      ▼      ▼
                 RANKING PROFILE SIMULATOR
```

A implementação deve preservar essa separação conceitual.

---

# 4. Princípio arquitetural principal

O projeto deve seguir uma pipeline em camadas:

```text
RAW
 ↓
PROCESSED
 ↓
FEATURES
 ↓
MODELS
 ↓
RESULTS
 ↓
APPLICATION
```

Cada camada possui responsabilidade própria.

**RAW**

Dados originais.

Não devem ser modificados manualmente.

**PROCESSED**

Dados limpos e normalizados.

Representam entidades confiáveis do domínio:

- eventos;
- lutadores;
- lutas;
- estatísticas por round.

**FEATURES**

Variáveis derivadas dos dados processados.

Exemplos:

- estatísticas numéricas;
- features por luta;
- features por lutador;
- histórico pré-luta.

**MODELS**

Modelos estatísticos e de Machine Learning.

Exemplos:

- Win Rate;
- Elo;
- Glicko;
- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- modelos de matchup.

**RESULTS**

Resultados dos experimentos.

Exemplos:

- métricas;
- previsões;
- rankings;
- comparações entre modelos;
- gráficos.

**APPLICATION**

Interface final do produto.

Futuramente:

- website;
- busca;
- perfis;
- rankings;
- histórico;
- simulador.

---

# 5. Estrutura atual do projeto

A estrutura esperada atualmente é:

```text
ultimate-fighter-rank/
│
├── data/
│   ├── raw/
│   │   ├── ufc_event_details.csv
│   │   ├── ufc_fighter_details.csv
│   │   ├── ufc_fight_details.csv
│   │   ├── ufc_fight_results.csv
│   │   ├── ufc_fight_stats.csv
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   ├── events.csv
│   │   ├── fighters.csv
│   │   ├── fights.csv
│   │   └── round_stats.csv
│   │
│   ├── features/
│   │   ├── round_stats_numeric.csv
│   │   ├── fight_fighter_features.csv
│   │   ├── fight_features.csv
│   │   ├── fighter_fight_features.csv
│   │   ├── fighter_history.csv
│   │   ├── fighter_ratings.csv          # produção — Elo v0.1
│   │   ├── param_sweep/                 # experimento: variantes de K/SCALE/PROVISIONAL_FIGHTS
│   │   └── recency/                     # experimento: ratings com decaimento por inatividade
│   │
│   ├── evaluation/
│   │   ├── evaluation_dataset.csv       # y, p_win_rate, p_elo por luta
│   │   ├── metrics_summary.csv          # Win Rate vs. Elo v0.1
│   │   ├── calibration_tables.csv
│   │   ├── param_sweep/summary.csv
│   │   ├── recency/summary.csv
│   │   ├── recency/fold_metrics.csv
│   │   └── glicko2/
│   │
│   └── results/
│       ├── glicko2_temporal_results.csv
│       ├── glicko2_predictions.csv
│       ├── opponent_quality_temporal_results.csv
│       ├── opponent_quality_predictions.csv
│       ├── performance_temporal_results.csv
│       ├── performance_predictions.csv
│       └── performance_coefficients.csv
│
├── src/
│   ├── build_dataset.py
│   ├── build_features.py
│   ├── build_fight_features.py
│   ├── build_fighter_fight_features.py
│   ├── build_fighter_history.py
│   ├── build_fighter_ratings.py         # produção — Elo v0.1
│   ├── evaluate_ratings.py              # avaliação temporal Win Rate vs. Elo
│   ├── validate_history.py
│   ├── param_sweep.py                   # experimento — K, SCALE, PROVISIONAL_FIGHTS
│   ├── recency_sweep.py                 # experimento — decaimento por inatividade
│   ├── glicko2_experiment.py            # experimento — Glicko-2
│   ├── opponent_quality_experiment.py   # experimento — Elo + Strength of Schedule
│   └── performance_experiment.py        # experimento — Elo + striking/grappling/dominância
│
├── tests/
│   └── __init__.py                      # ainda sem testes automatizados (seção 44)
│
├── notebooks/                           # ainda vazio
│
├── docs/
│   ├── development_guideline.md
│   ├── rating_methodology.md
│   ├── data_model.md
│   └── identity_resolution.md
│
├── .gitignore
├── requirements.txt
├── README.md
└── ...
```

A estrutura pode evoluir.

Novos módulos devem ser adicionados respeitando a separação de responsabilidades.

Um script em `src/` que não sobrescreve os artefatos de produção (`data/features/fighter_ratings.csv`, `data/evaluation/metrics_summary.csv`) e grava seus próprios resultados em uma subpasta dedicada de `data/features/`, `data/evaluation/` ou `data/results/` é, por definição, um **experimento** (seção 45) — não uma mudança de produção. Todos os scripts marcados como "experimento" acima seguem essa regra.

---

# 6. Ambiente de desenvolvimento

O projeto atualmente é desenvolvido em Windows.

Ambiente Python:

```text
.\.venv\Scripts\Activate.ps1
```

Para sair do ambiente:

```text
deactivate
```

O projeto deve utilizar um ambiente virtual próprio.

Não instalar dependências globalmente quando elas puderem ser instaladas no ambiente virtual.

---

# 7. Git

O repositório oficial é:

```text
https://github.com/Th3Wolfe/Ultimate-Fighter-Rank
```

Branch principal:

```text
main
```

Antes de trabalhar:

```text
git status
git pull
```

Depois de alterações:

```text
git status
git diff
```

Após validar:

```text
git add .
git commit -m "tipo: descrição"
git push
```

Preferir mensagens de commit em português.

Exemplos:

```text
feat: adiciona baseline de Elo
feat: adiciona avaliação temporal
fix: corrige associação de estatísticas por luta
refactor: separa cálculo de rating da avaliação
test: adiciona validações do histórico
docs: atualiza metodologia do ranking
```

---

# 8. Regra de ouro do Git

Nunca fazer commit de arquivos gerados ou temporários sem verificar se eles pertencem ao repositório.

Antes de qualquer commit:

```text
git status
```

Verificar especialmente:

- arquivos de dados muito grandes;
- bancos locais;
- ambientes virtuais;
- caches;
- arquivos temporários;
- outputs experimentais.

O `.gitignore` deve proteger arquivos que não precisam ser versionados.

---

# 9. Dados de origem

A fonte inicial utilizada pelo projeto é o UFCStats.

Os arquivos brutos atualmente utilizados são:

```text
ufc_event_details.csv
ufc_fighter_details.csv
ufc_fight_details.csv
ufc_fight_results.csv
ufc_fight_stats.csv
```

Esses arquivos representam diferentes níveis do domínio.

---

# 10. Entidades principais

**Event**

Representa um evento.

Campos processados:

```text
event_id
event_name
date
location
```

**Fighter**

Representa um lutador.

Campos:

```text
fighter_id
name
first_name
last_name
nickname
```

**Fight**

Representa uma luta.

Campos principais:

```text
fight_id
event_id
event_name
bout
fighter_1_id
fighter_2_id
fighter_1_name
fighter_2_name
fighter_1_result
fighter_2_result
weight_class
method
ending_round
ending_time
time_format
referee
details
```

**Round Stats**

Representa as estatísticas de um lutador em um determinado round.

Campos:

```text
fight_id
fighter_id
round
fighter_name
knockdowns
significant_strikes
significant_strikes_pct
total_strikes
takedowns
takedown_pct
submission_attempts
reversals
control_time
head_strikes
body_strikes
leg_strikes
distance_strikes
clinch_strikes
ground_strikes
```

---

# 11. Estado atual dos dados

O ETL atual produz:

```text
Events:       789
Fighters:     4618
Fights:       8887
Round stats:  41748
```

Resultados das lutas:

```text
W/L  5588
L/W  3144
NC/NC 90
D/D   65
```

Validações atuais:

```text
Fight fighter 1 unresolved: 0
Fight fighter 2 unresolved: 0
Round fighters unresolved: 0

Duplicate fights: 0
Fights without event: 0
Stats without fight: 0
Duplicate stat rows: 0
```

O dataset processado foi considerado válido pelos testes atuais.

---

# 12. ETL

O principal pipeline de construção do dataset é:

```text
src/build_dataset.py
```

Responsabilidades:

1. Ler os arquivos RAW.
2. Normalizar nomes.
3. Resolver identificadores.
4. Construir eventos.
5. Construir lutadores.
6. Construir lutas.
7. Construir estatísticas por round.
8. Validar integridade.
9. Salvar os dados processados somente quando válidos.

---

# 13. Identidade de lutadores

A identidade dos lutadores é crítica.

Não assumir que o nome exibido pelo UFCStats é sempre consistente.

O pipeline possui aliases conhecidos.

Atualmente existem correções para casos como:

```text
Bibulatov Magomed → Magomed Bibulatov
Kai Kamaka → Kai Kamaka III
Patricio Freire → Patricio Pitbull
Rafael Cerquiera → Rafael Cerqueira
```

Qualquer novo conflito de identidade deve ser investigado antes de adicionar um alias.

Nunca corrigir nomes silenciosamente sem registrar o motivo.

---

# 14. Identidade de eventos

Existem eventos que aparecem com nomes diferentes entre arquivos.

Aliases conhecidos:

```text
UFC Fight Night: Grasso vs. Shevchenko 2
→ Noche UFC: Grasso vs. Shevchenko 2

UFC Fight Night: Lopes vs. Silva
→ Noche UFC: Lopes vs. Silva
```

Existe também:

```text
UFC - Road to UFC 4.6
```

que não possuía correspondência no arquivo original de detalhes de eventos.

Foi preservado com ID sintético e metadata registrada no ETL.

Dados utilizados:

```text
Date: August 22, 2025
Location: Shanghai, Hebei, China
```

IDs sintéticos devem ser determinísticos.

---

# 15. Caso especial Sakuraba vs. Marcus Silveira

Existe um caso histórico importante no dataset:

```text
UFC - Ultimate Japan
Kazushi Sakuraba vs. Marcus Silveira
```

Existem duas lutas distintas:

```text
1.
URL: ec1bda9a4c2aab42
Resultado: W/L
Método: Submission
Round: 1
Tempo: 3:44

2.
URL: 2750ac5854e8b28
Resultado: NC/NC
Método: Overturned
Round: 1
Tempo: 1:51
```

Essas lutas não devem ser tratadas como duplicatas.

O ETL possui lógica específica para preservar ambas.

Qualquer alteração na associação entre:

```text
EVENT + BOUT
```

e:

```text
fight_id
```

deve preservar esse caso.

---

# 16. Estatísticas sem aplicação

O dataset possui lutas antigas sem estatísticas de round.

Atualmente:

```text
8887 lutas
8866 com estatísticas
21 sem estatísticas
```

As 21 lutas sem estatísticas devem continuar existindo.

Não converter ausência de estatística em zero.

Exemplo:

```text
0 of 0
```

não significa necessariamente:

```text
0%
```

Valores ausentes devem preservar seu significado.

---

# 17. Features numéricas

Script:

```text
src/build_features.py
```

Converte as estatísticas textuais para formato numérico.

Exemplos:

```text
X of Y
```

vira:

```text
landed
attempted
```

Percentuais são convertidos para valores numéricos.

Controle é convertido para segundos.

O script valida:

- quantidade de linhas;
- chaves;
- duplicatas;
- landed <= attempted;
- percentuais entre 0 e 100;
- controle não negativo;
- rounds válidos.

---

# 18. Features por luta

Script:

```text
src/build_fight_features.py
```

Produz:

```text
fight_fighter_features.csv
fight_features.csv
```

A primeira representação possui uma linha por lutador em cada luta.

A segunda representa a luta completa.

As estatísticas dos rounds são agregadas para o nível da luta.

Eficiências devem ser recalculadas no nível agregado quando apropriado.

Não calcular simplesmente a média das porcentagens dos rounds quando a métrica correta for:

```text
total_landed / total_attempted
```

---

# 19. Representação fighter-fight

Script:

```text
src/build_fighter_fight_features.py
```

Transforma cada luta em duas observações:

```text
fight_id + fighter_id
```

Cada linha representa um lutador dentro de uma luta.

Além das próprias estatísticas, possui:

- opponent_id;
- opponent_name;
- opponent_result;
- resultado do lutador;
- duração da luta.

Essa tabela é uma das principais bases para construir o histórico temporal.

---

# 20. Histórico pré-luta

Script:

```text
src/build_fighter_history.py
```

Constrói o estado histórico dos lutadores.

As features históricas são calculadas usando:

```text
shift(1)
```

antes da agregação cumulativa.

Isso é essencial para evitar leakage.

Exemplo:

```text
Luta 1
career_fights_before = 0

Luta 2
career_fights_before = 1

Luta 3
career_fights_before = 2
```

Nunca usar a própria luta para construir o estado utilizado na previsão daquela luta.

---

# 21. Features históricas atuais

O histórico possui:

```text
career_fights_before
wins_before
losses_before
draws_before
nc_before
win_rate_before
```

Também possui históricos de:

```text
significant_strikes_landed
significant_strikes_attempted
total_strikes_landed
total_strikes_attempted
takedowns_landed
takedowns_attempted
knockdowns
submission_attempts
reversals
control_time_seconds
```

Para essas métricas existem:

```text
sum_<metric>_before
avg_<metric>_before
```

Também existem médias históricas para:

```text
significant_strikes_pct
takedown_pct
```

---

# 22. Validação do histórico

Script:

```text
src/validate_history.py
```

Valida atualmente:

- duplicidade de (fight_id, fighter_id);
- consistência de career_fights_before;
- contadores históricos;
- estreias;
- win rate;
- estatísticas históricas;
- existência das colunas esperadas;
- propriedades temporais básicas.

Resultado atual:

```text
Histórico válido: nenhum problema encontrado.
```

Essa validação deve ser executada sempre que a construção do histórico for modificada.

---

# 23. Problema conhecido: lutas no mesmo dia

A ordenação histórica atual utiliza:

```text
date
fight_id
fighter_id
```

Isso fornece uma ordenação determinística.

Entretanto, alguns períodos antigos do UFC possuem torneios nos quais um lutador pode lutar mais de uma vez no mesmo evento/dia.

Essa situação deve ser investigada antes de considerar a ordenação temporal absolutamente perfeita.

Nunca inventar uma ordem de luta.

Se uma sequência confiável puder ser derivada da fonte, ela deverá substituir a ordenação arbitrária.

---

# 24. Rating — objetivo

O Fighter Rank deve representar:

```text
A força competitiva estimada de um lutador naquele momento.
```

Não deve ser simplesmente:

```text
número de vitórias
```

ou:

```text
win rate
```

ou:

```text
número de finalizações
```

O resultado da luta é o sinal principal, mas contexto e performance podem fornecer informação adicional.

---

# 25. Metodologia do rating

A especificação conceitual está em:

```text
docs/rating_methodology.md
```

Esse arquivo deve ser lido antes de modificar o sistema de rating.

Ele define:

- temporalidade;
- prevenção de leakage;
- baselines;
- Elo;
- recência;
- qualidade do adversário;
- performance;
- método;
- experiência;
- small sample;
- estilo;
- matchup;
- avaliação temporal;
- métricas de avaliação;
- rankings;
- roadmap metodológico.

---

# 26. Regra fundamental do rating

Não criar uma fórmula arbitrária simplesmente porque ela parece intuitiva.

Evitar decisões como:

```text
KO = +20
Submission = +15
Decision = +5
```

sem experimentação.

A metodologia deve testar hipóteses.

O objetivo é descobrir empiricamente quais informações melhoram a estimativa.

---

# 27. Baselines obrigatórios

Antes de modelos complexos, implementar:

1. Win Rate
2. Elo

Depois testar extensões:

3. Elo + recência
4. Elo + qualidade do adversário
5. Elo + performance
6. outros modelos de rating

Os resultados devem ser comparados temporalmente.

---

# 28. Avaliação temporal

Não utilizar apenas:

```text
train_test_split(random_state=...)
```

para avaliar o sistema principal.

O UFC é um processo temporal.

O modelo deve aprender com o passado e ser testado no futuro.

Exemplo:

```text
Treino:
1994 → 2010

Teste:
2011 → 2013
```

Depois:

```text
Treino:
1994 → 2013

Teste:
2014 → 2016
```

E assim por diante.

Essa abordagem é conhecida como walk-forward validation.

---

# 29. Métricas de ML

A avaliação deve considerar:

```text
Log Loss
Brier Score
ROC AUC
Accuracy
Calibration
```

Accuracy não deve ser a única métrica.

Para um sistema probabilístico, a qualidade das probabilidades é especialmente importante.

---

# 30. Rating vs. Prediction

Manter os conceitos separados.

**Rating**

Pergunta:

```text
Qual é a força competitiva estimada do lutador?
```

**Prediction**

Pergunta:

```text
Qual é a probabilidade de A vencer B?
```

Um rating pode alimentar um modelo de previsão.

Mas os dois componentes não precisam ser o mesmo algoritmo.

---

# 31. Estilo

O projeto deverá construir posteriormente um perfil multidimensional dos lutadores.

Possíveis dimensões:

```text
Striking
Wrestling
Grappling
Submission
Control
Defense
Aggression
Pace
Finishing
Cardio
```

Essas categorias são conceituais.

As métricas exatas ainda devem ser definidas e testadas.

---

# 32. Descoberta de estilo

Não assumir que um lutador pertence simplesmente a uma categoria como:

```text
Striker
Wrestler
Grappler
```

O objetivo é permitir representações mais contínuas.

Por exemplo:

```text
Striking: 0.82
Wrestling: 0.64
Submission: 0.71
Control: 0.38
Pace: 0.77
```

Os valores acima são apenas ilustrativos.

Não devem ser implementados como números fixos.

---

# 33. Matchup

O matchup deve estudar a interação entre lutadores.

A pergunta é:

```text
Como as características de A interagem com as características de B?
```

Características possíveis:

```text
rating
striking
wrestling
grappling
defense
pace
control
submission
```

O modelo deverá buscar interações nos dados.

Evitar regras manuais do tipo:

```text
Wrestler sempre vence Striker
```

MMA não deve ser reduzido a um sistema de relações determinísticas.

---

# 34. Features de matchup

Para uma luta entre A e B, poderão ser criadas:

```text
feature_A
feature_B
feature_A - feature_B
feature_A / feature_B
feature_A * feature_B
```

As interações devem ser selecionadas com validação.

---

# 35. Categorias de peso

O sistema deve possuir rankings por divisão.

Mudanças de categoria devem ser preservadas.

Não misturar automaticamente:

```text
Heavyweight
Welterweight
Flyweight
```

como se fossem exatamente o mesmo contexto competitivo.

A categoria da luta deve ser considerada no modelo.

---

# 36. Pound-for-Pound

O P4P será desenvolvido separadamente.

Não utilizar simplesmente:

```text
rating global de todas as categorias
```

sem normalização.

O sistema deverá considerar contexto de divisão.

Possíveis informações:

- força relativa dentro da divisão;
- qualidade dos adversários;
- performance relativa;
- rating normalizado;
- resultados entre divisões.

---

# 37. Rankings especializados

O projeto deverá permitir rankings baseados em dimensões específicas.

Exemplos:

```text
Finalização
Striking
Wrestling
Controle
Knockdowns
Defesa
Pace
```

Esses rankings não devem ser confundidos com o ranking de força competitiva geral.

---

# 38. Ranking de finalizadores

A ideia inicial é combinar:

```text
Finishing Rate
Finishing Volume
Opponent Quality
Activity
```

Mas os pesos não devem ser definidos arbitrariamente.

Devem ser testados.

É necessário controlar:

- amostras pequenas;
- qualidade dos adversários;
- atividade;
- diferenças de número de lutas.

---

# 39. Simulador

O simulador é uma etapa futura.

Não deve ser implementado antes de existir uma base preditiva suficientemente validada.

A arquitetura desejada:

```text
Fighter A
    +
Fighter B
    ↓
Historical State
    ↓
Matchup Features
    ↓
Prediction Model
    ↓
Outcome Probabilities
```

Possíveis saídas:

```text
P(A vence)
P(B vence)
P(KO/TKO)
P(Submission)
P(Decision)
```

Essas probabilidades devem ser tratadas como estimativas do modelo, não como certezas.

---

# 40. Dados novos

Quando novos dados forem adicionados:

1. Nunca alterar os RAW manualmente.
2. Reexecutar o ETL.
3. Reexecutar as validações.
4. Reexecutar a construção de features.
5. Reexecutar o histórico.
6. Verificar mudanças de cardinalidade.
7. Verificar novas inconsistências.
8. Só então atualizar os modelos.

Pipeline:

```text
RAW
 ↓
build_dataset.py
 ↓
PROCESSED
 ↓
build_features.py
 ↓
build_fight_features.py
 ↓
build_fighter_fight_features.py
 ↓
build_fighter_history.py
 ↓
validate_history.py
 ↓
MODEL
```

---

# 41. Regra contra corrupção silenciosa

Nunca aceitar automaticamente:

```text
script executou sem erro
```

como equivalente a:

```text
dados corretos
```

Todo pipeline deve possuir validações.

Sempre verificar:

- número de linhas;
- chaves;
- duplicatas;
- foreign keys;
- valores inválidos;
- valores ausentes;
- cardinalidade;
- consistência temporal.

---

# 42. Regra contra dados inventados

Nunca preencher dados desconhecidos com valores inventados.

Exemplos proibidos:

```text
missing percentage → 0
missing control time → 0
missing statistics → estimativa manual
missing event → nome inventado
```

Quando uma informação não existe, preservar a ausência ou utilizar uma metodologia explicitamente documentada.

---

# 43. Alterações no schema

Qualquer mudança de schema deve verificar todos os consumidores da tabela.

Exemplo:

Se mudar:

```text
fighter_history.csv
```

verificar:

- scripts de validação;
- modelos;
- notebooks;
- futuros endpoints;
- frontend;
- documentação.

Não alterar nomes de colunas silenciosamente.

---

# 44. Testes

Cada nova etapa importante deve possuir validações automatizadas.

Exemplos:

```text
assert row_count_expected
assert no_duplicate_keys
assert no_invalid_foreign_keys
assert no_future_information
assert valid_ranges
```

Quanto mais crítica a transformação, mais forte deve ser a validação.

---

# 45. Experimentos

Experimentos de ML devem ser reproduzíveis.

Registrar:

- dataset utilizado
- período de treino
- período de teste
- features
- modelo
- hiperparâmetros
- métricas
- resultado

Evitar escolher um modelo apenas olhando o resultado de uma única divisão temporal.

---

# 46. Estrutura futura para ML

A estrutura pode evoluir para:

```text
src/
├── data/
├── features/
├── models/
├── evaluation/
└── utils/
```

Exemplo:

```text
src/models/
├── elo.py
├── glicko.py
├── win_rate.py
└── prediction.py

src/evaluation/
├── temporal_split.py
├── metrics.py
└── backtest.py
```

Não criar essa estrutura prematuramente se ainda não for necessária.

---

# 47. Separação entre experimento e produção

Durante pesquisa:

```text
experiments/
```

pode conter:

- notebooks;
- análises;
- testes;
- gráficos;
- comparações.

Código reutilizável deve ficar em:

```text
src/
```

A aplicação final não deve depender de notebooks.

---

# 48. Documentação

Documentos importantes:

```text
docs/
├── development_guideline.md    # este documento
├── rating_methodology.md       # metodologia e resultados de todos os experimentos de rating
├── data_model.md               # schema das tabelas processadas, de features e de resultados
└── identity_resolution.md      # como fighter_id é resolvido e exceções auditadas
```

Novas decisões relevantes devem ser documentadas.

Exemplos:

- nova fonte de dados;
- mudança no método de rating;
- novo modelo;
- nova feature;
- mudança na estratégia de avaliação;
- decisão sobre P4P;
- tratamento de casos históricos especiais.

---

# 49. Como outra IA deve assumir o projeto

Se uma nova IA receber este repositório, ela deve seguir:

1. Ler development_guideline.md
2. Ler rating_methodology.md
3. Inspecionar README.md
4. Executar git status
5. Inspecionar estrutura do projeto
6. Verificar scripts existentes
7. Verificar dados existentes
8. Executar validações relevantes
9. Identificar o próximo item do roadmap
10. Só então propor alterações

A IA não deve:

- reescrever o projeto inteiro;
- substituir a arquitetura sem justificativa;
- apagar dados;
- refazer o ETL sem necessidade;
- criar fórmulas arbitrárias;
- ignorar casos especiais conhecidos;
- usar informação futura em features históricas.

---

# 50. Como solicitar ajuda de outra IA

Ao continuar o projeto em outra IA, fornecer:

```text
Este é o projeto Ultimate Fighter Rank.

Leia primeiro:
docs/development_guideline.md
docs/rating_methodology.md

Não altere arquitetura ou metodologia sem justificar.

Estado atual:
[resultado de git status]

Tarefa atual:
[descrever tarefa]

Objetivo:
[descrever resultado esperado]
```

Se necessário, também fornecer:

- README.md
- estrutura do projeto
- logs de validação
- resultado dos experimentos

---

# 51. Regra de continuidade

O projeto deve sempre possuir uma noção clara de:

```text
CONCLUÍDO
EM DESENVOLVIMENTO
PRÓXIMO
FUTURO
```

Estado atual:

**CONCLUÍDO**

Data Engineering
- ETL inicial
- Eventos
- Lutadores
- Lutas
- Estatísticas por round
- Resolução de identidades
- Tratamento de aliases
- Tratamento de eventos ausentes
- Preservação das duas lutas Sakuraba vs. Marcus
- Validação de foreign keys
- Validação de duplicatas

Feature Engineering
- Estatísticas numéricas
- Features por luta
- Features fighter-fight
- Histórico temporal
- Features pré-luta
- Validação do histórico

Documentação
- rating_methodology.md
- development_guideline.md

Fighter Rating
- Baseline de Elo v0.1 (`src/build_fighter_ratings.py`)
  - rating único por lutador, cross-division (seção 37.2 do rating_methodology.md)
  - normalização de `weight_class` → divisão real (seção 37.3)
  - No Contest sem atualização de rating, mas contando na experiência (seção 37.1)
  - K-factor provisório pós-estreia e pós-mudança de divisão (seção 37.4)
  - validação automática de estreia, continuidade e ausência de update em NC
- Baseline de Win Rate como modelo de previsão + Avaliação temporal (`src/evaluate_ratings.py`)
  - Win Rate convertido em `P(A vence)` por normalização direta, sem fórmula inventada (seção 38.1 do rating_methodology.md)
  - Elo reaproveita `expected_score` já calculado pelo v0.1, sem recomputar nada (seção 38.2)
  - Draw e No Contest excluídos da avaliação, com justificativa registrada (seção 38.3)
  - split temporal simples (2019+) e walk-forward expansivo (janelas de 3 anos a partir de 2011), seção 38.4
  - Log Loss, Brier, ROC AUC, Accuracy e Calibration calculados lado a lado para os dois modelos (seção 38.5)
  - resultado: nenhum modelo domina em todas as métricas — Win Rate leva em Accuracy/AUC, Elo é muito mais estável em Log Loss por não produzir probabilidades degeneradas (0.0/1.0); detalhes e implicação para o roadmap em rating_methodology.md, seção 38.5
- Sweep de parâmetros do Elo (`src/param_sweep.py`) — PROVISIONAL_FIGHTS, K-factor e SCALE testados isoladamente via walk-forward (rating_methodology.md, seções 37.5 e 37.6)
  - resultado: nenhuma variação de K ou SCALE superou o baseline em Log Loss/Brier; `PROVISIONAL_FIGHTS=5` é candidato experimental com melhor Log Loss agregado, mas não uniforme entre períodos — baseline v0.1 (K=32/64, SCALE=400, PROVISIONAL_FIGHTS=3) permanece oficial
- Experimento de recência (`src/recency_sweep.py`) — decaimento do rating por inatividade, meia-vida testada de 1 a 5 anos (rating_methodology.md, seção 39)
  - resultado: meias-vidas de 2 a 5 anos melhoram Log Loss, Brier, AUC e Accuracy simultaneamente em relação ao Elo v0.1 — a única extensão isolada com esse comportamento além da seção seguinte; ainda não incorporada à produção
- Experimento Glicko-2 (`src/glicko2_experiment.py`, sem aging por calendário) — rating_methodology.md, seção 40
  - resultado: AUC e Accuracy melhores que o Elo v0.1, mas Log Loss e Brier piores — não adotado
- Experimento de qualidade do adversário / Strength of Schedule (`src/opponent_quality_experiment.py`) — rating_methodology.md, seção 41
  - resultado: Log Loss piora fortemente (0.68 → 0.78 pooled), sobretudo no fold mais escasso em histórico (2011-2014) — resultado negativo registrado, não adotado
- Experimento de performance — striking/grappling/dominância (`src/performance_experiment.py`, modelos M0-M5) — rating_methodology.md, seção 42
  - resultado: M4 (Elo + todas as features de performance) melhora Log Loss, Brier, AUC e Accuracy simultaneamente em relação ao Elo v0.1 isolado; performance sem Elo (M5) é o pior modelo do grupo — ainda não incorporado à produção
- Síntese comparativa das cinco extensões acima, mesmo protocolo walk-forward (rating_methodology.md, seção 43)

---

# 52. EM DESENVOLVIMENTO

A próxima grande etapa continua sendo:

```text
Fighter Rating
```

Concluído:

```text
Elo baseline v0.1
Win Rate baseline (como modelo de previsão)
Avaliação temporal (split simples + walk-forward)
Comparação formal Win Rate vs. Elo v0.1
Sweep de parâmetros do Elo (K, SCALE, PROVISIONAL_FIGHTS)
Extensão de recência (decaimento por inatividade)
Extensão de qualidade do adversário (Strength of Schedule)
Extensão de performance (striking, grappling, dominância)
Experimento Glicko-2 (sem aging)
Síntese comparativa das extensões acima
```

Todos os itens 5-9 do roadmap original (seção 53) foram **experimentalmente concluídos**: cada extensão foi implementada, avaliada com o mesmo harness temporal e comparada contra o Elo v0.1. Dois resultados de sinal positivo e consistente em todas as métricas emergiram — recência (`rating_methodology.md`, seção 39) e Elo + performance/M4 (seção 42) — e dois de sinal negativo ou misto — Glicko-2 sem aging (seção 40) e Elo + SoS (seção 41).

**O que falta antes de avançar para o item 10 (Fighter Style Profile):** nenhuma das extensões testadas foi promovida a `src/build_fighter_ratings.py`. Falta (a) validar fold a fold a estabilidade dos dois resultados positivos (recência e performance), do mesmo jeito que a seção 37.5 fez para `PROVISIONAL_FIGHTS`; (b) decidir se recência e performance devem ser combinadas ou tratadas como alternativas; e (c) só então atualizar o rating de produção. Essa decisão é o item em aberto mais próximo no roadmap.

---

# 53. PRÓXIMOS PASSOS

A sequência recomendada é:

```text
1. Criar baseline de Win Rate                   [CONCLUÍDO — v0.1]
        ↓
2. Criar baseline de Elo                        [CONCLUÍDO — v0.1]
        ↓
3. Criar avaliação temporal                     [CONCLUÍDO — v0.1]
        ↓
4. Comparar Win Rate vs Elo                     [CONCLUÍDO — v0.1]
        ↓
5. Testar parâmetros do Elo                     [CONCLUÍDO — experimental, seções 37.5/37.6]
        ↓
6. Adicionar recência                           [CONCLUÍDO — experimental, seção 39; positivo]
        ↓
7. Adicionar qualidade do adversário             [CONCLUÍDO — experimental, seção 41; negativo]
        ↓
8. Adicionar performance                        [CONCLUÍDO — experimental, seção 42; positivo]
        ↓
9. Comparar modelos                             [CONCLUÍDO — seção 43]
        ↓
9.5 Validar estabilidade e promover extensão(ões) vencedora(s) para produção  ← próximo passo
        ↓
10. Criar Fighter Style Profile
        ↓
11. Criar Matchup Model
        ↓
12. Criar rankings
        ↓
13. Criar P4P
        ↓
14. Criar simulador
        ↓
15. Construir aplicação web
```

Detalhes de cada experimento dos itens 5-9, incluindo tabelas de resultado e decisões metodológicas registradas, estão em `docs/rating_methodology.md`, seções 37.5 a 43.

---

# 54. O que NÃO fazer agora

Não começar ainda por:

- frontend completo;
- banco PostgreSQL;
- sistema de login;
- deployment;
- simulador visual;
- ranking P4P;
- dashboard complexo;
- dezenas de métricas arbitrárias;
- deep learning;
- LLM.

A prioridade atual é validar cientificamente o núcleo:

```text
Dados
 ↓
Histórico
 ↓
Rating
 ↓
Previsão
```

---

# 55. Critério de qualidade do projeto

Uma implementação não deve ser considerada boa apenas porque:

```text
funciona
```

Ela deve ser:

```text
correta
+
reproduzível
+
validada
+
temporalmente consistente
+
explicável
+
extensível
```

---

# 56. Princípio de simplicidade

Começar com modelos simples.

Se:

```text
Elo
```

for competitivo com um modelo muito mais complexo, isso é uma informação importante.

Complexidade deve ser adicionada somente quando houver evidência de benefício.

Não utilizar Machine Learning complexo apenas para tornar o projeto mais impressionante.

---

# 57. Princípio de experimentação

Toda hipótese relevante deve seguir:

```text
Hipótese
    ↓
Feature / Modelo
    ↓
Backtest temporal
    ↓
Métrica
    ↓
Comparação
    ↓
Conclusão
```

Exemplo:

```text
"Recência melhora a estimativa de força."
```

Não assumir que isso é verdade.

Testar:

```text
Elo
vs.
Elo + Recência
```

e medir a diferença em períodos futuros.

---

# 58. Princípio de transparência

Sempre que possível, o sistema deve conseguir explicar:

```text
Por que o rating mudou?
```

Exemplo:

```text
Rating antes: 1512
Resultado: vitória
Adversário: rating 1580
Rating depois: 1541
```

Futuramente:

```text
Rating antes
+
resultado
+
qualidade do adversário
+
performance
+
recência
=
novo estado
```

A explicabilidade será importante para o produto final.

---

# 59. Visão do produto final

O produto final deverá permitir que o usuário:

**Pesquise**
- lutadores;
- eventos;
- lutas;
- períodos históricos.

**Consulte rankings**
- geral;
- divisões;
- P4P;
- finalização;
- striking;
- wrestling;
- controle;
- defesa;
- outros especializados.

**Consulte perfis**

Cada lutador poderá apresentar:

- rating histórico;
- evolução do rating;
- resultados;
- adversários;
- estatísticas;
- estilo;
- forma recente;
- desempenho por categoria.

**Compare lutadores**

Exemplo:

```text
Fighter A
vs.
Fighter B
```

Com:

- rating;
- histórico;
- estilo;
- estatísticas;
- matchup.

**Simule confrontos**

O usuário poderá futuramente fornecer:

```text
A vs B
```

e receber estimativas probabilísticas baseadas no modelo.

---

# 60. Visão de longo prazo

A visão final do Ultimate Fighter Rank é evoluir de:

```text
Dataset histórico do UFC
```

para:

```text
Sistema analítico completo de MMA
```

com:

```text
Dados históricos
      ↓
Modelagem temporal
      ↓
Rating
      ↓
Performance
      ↓
Estilo
      ↓
Matchup
      ↓
Previsão
      ↓
Ranking
      ↓
Simulação
```

O projeto deve permanecer orientado por dados durante toda essa evolução.

---

# 61. Checklist antes de qualquer alteração

Antes de implementar:

- [ ] Entendi qual camada estou modificando?
- [ ] Li a metodologia relevante?
- [ ] Verifiquei o estado atual do Git?
- [ ] Verifiquei se a funcionalidade já existe?
- [ ] A mudança pode causar leakage?
- [ ] A mudança pode alterar cardinalidade?
- [ ] Existem casos históricos especiais afetados?
- [ ] Existem dados ausentes que precisam ser preservados?
- [ ] Preciso adicionar uma validação?
- [ ] Preciso atualizar a documentação?

---

# 62. Checklist antes do commit

```text
git status
git diff
```

Depois:

- [ ] Scripts executam sem erro.
- [ ] Validações passam.
- [ ] Número de linhas esperado.
- [ ] Nenhuma duplicata inesperada.
- [ ] Nenhum foreign key quebrado.
- [ ] Nenhuma informação futura utilizada.
- [ ] Dados ausentes tratados corretamente.
- [ ] Documentação atualizada quando necessário.

Então:

```text
git add .
git commit -m "tipo: descrição"
git push
```

---

# 63. Regra final para qualquer IA ou desenvolvedor

O objetivo não é apenas fazer o código funcionar.

O objetivo é construir um sistema historicamente confiável.

Portanto:

- Não inventar dados.
- Não esconder inconsistências.
- Não ignorar leakage.
- Não criar pesos arbitrários sem testar.
- Não apagar casos históricos difíceis.
- Não trocar simplicidade por complexidade sem evidência.
- Não quebrar etapas anteriores.
- Não assumir que uma previsão plausível significa que o modelo está correto.

A ordem de prioridade é:

```text
CORREÇÃO
    ↓
INTEGRIDADE
    ↓
REPRODUTIBILIDADE
    ↓
VALIDAÇÃO
    ↓
MODELAGEM
    ↓
COMPLEXIDADE
    ↓
INTERFACE
```

O Ultimate Fighter Rank deve ser construído como um projeto de engenharia de dados e Machine Learning real, no qual cada camada pode ser auditada, reproduzida e substituída sem comprometer as demais.
