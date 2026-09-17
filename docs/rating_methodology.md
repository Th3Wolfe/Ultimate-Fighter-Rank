# Fighter Rank — Metodologia de Rating

**Projeto:** Ultimate Fighter Rank
**Documento:** Metodologia do sistema de rating
**Versão:** 0.1
**Status:** Definição metodológica inicial

---

# 1. Objetivo

O Fighter Rank tem como objetivo estimar a força competitiva de cada lutador ao longo da história do UFC.

A pergunta central do sistema é:

> **Dado o estado histórico disponível em determinado momento, qual é a força competitiva estimada de cada lutador e qual seria a probabilidade de ele vencer outro lutador?**

O rating não deve representar simplesmente:

- número de vitórias;
- número de finalizações;
- percentual de vitórias;
- popularidade;
- posição oficial do UFC.

Ele deve representar uma estimativa dinâmica da força competitiva do lutador naquele momento.

O sistema será reconstruído cronologicamente, permitindo observar como a avaliação de cada lutador evolui após cada luta.

---

# 2. Princípios fundamentais

## 2.1 Temporalidade

Todas as informações utilizadas para avaliar uma luta devem estar disponíveis antes daquela luta acontecer.

Para uma luta `F` ocorrida no instante `t`, o modelo deve utilizar somente informações disponíveis em:

```text
t - 1
```

Nunca serão utilizadas informações da própria luta para determinar a previsão daquela luta.

Isso inclui:

- resultado;
- método;
- estatísticas dos rounds;
- duração;
- performance do adversário naquela luta;
- qualquer informação derivada posteriormente.

## 2.2 Ausência de data leakage

O sistema deve impedir que informações futuras contaminem o histórico.

Exemplo:

Se Jon Jones enfrentou um adversário em 2015, o rating usado para prever essa luta não pode considerar:

- lutas de Jones de 2016;
- ranking de Jones em 2020;
- estatísticas acumuladas de toda a carreira;
- resultado de lutas posteriores.

Cada luta deve enxergar apenas o passado.

## 2.3 Resultado continua sendo o principal sinal

O resultado da luta é o principal sinal de atualização da força competitiva.

- Uma vitória deve aumentar a avaliação do lutador.
- Uma derrota deve diminuir sua avaliação.
- Um empate deve produzir uma atualização específica.
- Um No Contest não deve ser tratado automaticamente como vitória ou derrota.

Performance e contexto podem modificar a magnitude da atualização, mas não devem transformar uma derrota em uma vitória.

## 2.4 Avaliação baseada em evidências

Pesos e regras importantes não devem ser definidos arbitrariamente quando puderem ser aprendidos ou avaliados empiricamente.

Em vez de assumir:

```text
KO = X pontos
Submission = Y pontos
Decision = Z pontos
```

serão testadas diferentes representações e modelos.

A escolha deverá considerar desempenho fora da amostra e validação temporal.

---

# 3. Unidade fundamental: estado do lutador

Antes de cada luta, cada lutador possui um estado histórico.

Conceitualmente:

```text
FighterState(t) =
    Rating
    Experience
    HistoricalPerformance
    RecentForm
    OpponentQuality
    Activity
    StyleProfile
```

Nem todas essas informações precisam entrar diretamente no rating.

Algumas serão utilizadas para:

- rating;
- previsão de resultado;
- análise de estilo;
- matchup;
- rankings especializados.

---

# 4. Rating

## 4.1 Definição

O rating representa uma estimativa da força competitiva do lutador em determinado momento.

Um rating alto deve significar que, segundo o histórico observado até aquele momento, o lutador possui maior força competitiva estimada.

O rating deve ser:

- dinâmico;
- atualizado luta a luta;
- sensível à qualidade do adversário;
- sensível ao resultado;
- potencialmente sensível à performance;
- potencialmente sensível à recência.

## 4.2 Baseline

O primeiro modelo será baseado em um sistema de rating do tipo Elo.

A lógica básica será:

```text
Rating_A
Rating_B
      ↓
Diferença de rating
      ↓
Probabilidade esperada
      ↓
Resultado real
      ↓
Atualização dos ratings
```

A formulação inicial poderá utilizar:

```text
P(A vence) =
    1 / (1 + 10 ^ ((R_B - R_A) / S))
```

onde:

- R_A = rating de A;
- R_B = rating de B;
- S = escala do sistema.

A fórmula exata e os parâmetros serão definidos experimentalmente.

---

# 5. Modelos candidatos

O projeto não ficará limitado a um único algoritmo.

Serão avaliados diferentes modelos.

## 5.1 Baselines

**Baseline 1 — Win Rate**

Utilizar somente o percentual histórico de vitórias.

Serve como referência simples.

**Baseline 2 — Elo**

Rating baseado principalmente no resultado e na diferença de rating entre os lutadores.

**Baseline 3 — Elo + recência**

Adiciona informação temporal para reduzir a influência de resultados muito antigos.

**Baseline 4 — Elo + qualidade do adversário**

Avalia explicitamente o nível dos adversários enfrentados.

**Baseline 5 — Rating + performance**

Adiciona características estatísticas de performance.

**Baseline 6 — Modelos supervisionados**

Serão avaliados modelos como:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- outros modelos adequados aos dados.

O objetivo não é escolher o modelo mais complexo, mas encontrar uma abordagem que generalize bem para lutas futuras.

---

# 6. Qualidade do adversário

Uma vitória contra um adversário forte deve fornecer mais evidência sobre a força do lutador do que uma vitória contra um adversário de força estimada inferior.

Essa característica já é naturalmente contemplada por sistemas de rating como Elo.

Entretanto, também será investigada a utilização de:

- rating do adversário;
- histórico de vitórias;
- força média dos adversários;
- força dos últimos adversários;
- qualidade contextual da divisão.

A utilização desses sinais será avaliada empiricamente.

---

# 7. Performance

O resultado não é a única informação disponível.

As estatísticas de luta permitem medir como o lutador chegou ao resultado.

Entre os sinais disponíveis estão:

- significant strikes;
- total strikes;
- takedowns;
- takedown attempts;
- knockdowns;
- submission attempts;
- reversals;
- control time;
- distribuição dos golpes;
- eficiência;
- duração da luta.

Esses sinais podem representar diferentes dimensões da performance.

---

# 8. Volume vs. eficiência

O sistema deve diferenciar volume de eficiência.

Por exemplo:

```text
Strikes landed
```

representa volume. Enquanto:

```text
Strikes landed / Strikes attempted
```

representa eficiência.

Os dois sinais possuem significados diferentes e não devem ser tratados como equivalentes.

O mesmo princípio será aplicado a:

- striking;
- takedowns;
- controle;
- finalizações.

---

# 9. Performance em derrotas

Uma derrota continuará sendo uma derrota.

Entretanto, a forma como ela ocorreu pode conter informação adicional.

Exemplo conceitual:

```text
Luta A:
derrota por decisão dividida
+
boa performance estatística
```

versus:

```text
Luta B:
derrota por finalização no primeiro round
```

Essas derrotas podem representar evidências diferentes sobre a força futura do lutador.

O sistema deverá testar se informações de performance conseguem melhorar a estimativa após uma derrota sem apagar o efeito negativo do resultado.

---

# 10. Método de vitória/derrota

O método será tratado como uma característica potencialmente relevante.

Categorias incluem, entre outras:

- KO;
- TKO;
- Submission;
- Decision;
- Draw;
- No Contest.

A hipótese inicial do projeto é que vitórias por finalização ou KO/TKO podem fornecer informações diferentes de vitórias por decisão.

Entretanto, não será atribuído manualmente um valor fixo a cada método sem validação.

Serão testadas diferentes representações:

- Método como variável categórica

ou

- Método como característica de performance

ou

- Método incorporado à atualização do rating

A contribuição de cada abordagem será avaliada fora da amostra.

---

# 11. Recência

A força observada no passado pode não representar perfeitamente a força atual.

Portanto, a recência será investigada como componente do sistema.

Serão testadas diferentes estratégias:

**Sem recência**

Todos os resultados possuem influência histórica.

**Decay contínuo**

A influência de uma luta diminui progressivamente conforme o tempo aumenta.

**Janela temporal**

Somente determinadas lutas recentes recebem maior influência.

**Atividade**

Períodos longos sem lutar podem reduzir a confiança na estimativa atual.

A hipótese de que uma diferença superior a aproximadamente um ano deve favorecer o lutador mais ativo será tratada como hipótese experimental, e não como regra fixa.

---

# 12. Experiência

O número de lutas anteriores será considerado como medida de experiência.

Variáveis possíveis:

```text
career_fights_before
wins_before
losses_before
draws_before
nc_before
```

Experiência não deve ser confundida com força.

Um lutador com 20 lutas não é necessariamente mais forte que um lutador com 5.

Por isso, experiência será utilizada principalmente para:

- contextualização;
- confiança da estimativa;
- tratamento de amostras pequenas;
- análise de incerteza.

---

# 13. Small sample problem

Lutadores estreantes possuem pouca informação histórica.

Um lutador com:

```text
1 luta = 1 vitória
```

não deve ser considerado equivalente a outro com:

```text
20 lutas = 20 vitórias
```

O sistema deverá considerar mecanismos para lidar com baixa quantidade de observações.

Possíveis abordagens:

- rating inicial;
- regressão em direção à média;
- incerteza do rating;
- Glicko;
- Bayesian shrinkage;
- peso dependente da quantidade de lutas.

Essas abordagens serão comparadas experimentalmente.

---

# 14. Histórico do lutador

O arquivo:

```text
data/features/fighter_history.csv
```

representa o estado histórico pré-luta.

As variáveis históricas são calculadas usando somente lutas anteriores.

Exemplos:

```text
career_fights_before
wins_before
losses_before
win_rate_before

avg_significant_strikes_pct_before
avg_takedown_pct_before

avg_knockdowns_before
avg_submission_attempts_before
avg_control_time_seconds_before
```

Também existem medidas acumuladas:

```text
sum_significant_strikes_landed_before
sum_takedowns_landed_before
sum_knockdowns_before
...
```

Essas variáveis serão utilizadas como base para experimentos posteriores.

---

# 15. Forma recente

A média de toda a carreira pode esconder mudanças recentes.

Por isso, será investigada a utilização de janelas recentes:

- últimas 3 lutas
- últimas 5 lutas
- últimas N lutas

Também poderá ser testada uma combinação entre:

```text
forma histórica
+
forma recente
```

A janela não será escolhida apenas por intuição.

Será comparada através de validação temporal.

---

# 16. Estilo do lutador

O Fighter Rank não deve representar todos os aspectos do projeto.

O sistema também deverá construir um perfil de estilo.

Possíveis dimensões:

- Striking
- Grappling
- Wrestling
- Submission
- Control
- Aggression
- Defense
- Pace
- Finishing
- Cardio

Essas dimensões serão derivadas das estatísticas disponíveis.

O objetivo é descobrir características recorrentes do comportamento de cada lutador.

---

# 17. Matchup

Uma parte futura do sistema será dedicada à interação entre estilos.

A pergunta deixa de ser:

```text
Quem é melhor?
```

e passa a ser:

```text
Como o estilo de A interage com o estilo de B?
```

Exemplos conceituais:

```text
Striker A
    vs
Wrestler B
```

ou:

```text
Grappler A
    vs
Grappler B
```

O modelo poderá procurar relações como:

```text
A tem dificuldade contra wrestlers
B possui forte defesa de takedown
```

Essas relações não serão assumidas manualmente.

O objetivo é permitir que o modelo descubra padrões nos dados.

---

# 18. Diferenças entre lutadores

Para modelos supervisionados, as características podem ser representadas como diferenças:

```text
feature_A - feature_B
```

Exemplo:

```text
rating_diff
win_rate_diff
strike_accuracy_diff
takedown_rate_diff
control_time_diff
```

Isso permite que o modelo aprenda relações entre as características dos dois lutadores.

Também poderão ser testadas outras representações:

```text
A
B
A - B
A / B
A + B
```

A escolha será baseada em validação.

---

# 19. Categorias de peso

O ranking principal deverá respeitar a categoria de peso.

Um lutador não deve ser comparado diretamente com outro de uma categoria muito diferente sem considerar o contexto.

Serão produzidos rankings por divisão.

Exemplo conceitual:

```text
Heavyweight
Light Heavyweight
Middleweight
Welterweight
...
```

As categorias serão derivadas dos dados históricos disponíveis.

Mudanças de categoria deverão ser preservadas.

---

# 20. Pound-for-Pound

O ranking P4P será tratado separadamente.

O objetivo será estimar força relativa entre categorias.

Para isso, poderão ser utilizados:

- performance relativa à divisão;
- qualidade dos adversários;
- rating normalizado;
- métricas relativas à categoria;
- desempenho contra adversários de diferentes níveis.

O P4P não será simplesmente uma ordenação dos ratings de todas as categorias misturados.

Essa metodologia será definida em uma etapa posterior.

---

# 21. Rankings especializados

Além do ranking geral, o projeto poderá produzir rankings especializados.

Exemplos:

- Top Finalizadores
- Top Strikers
- Top Wrestlers
- Top Grapplers
- Top Control
- Top Knockdown Artists
- Top Submission Threats
- Top Defensive Fighters
- Top Pace

Esses rankings não representam necessariamente a força competitiva geral.

Eles representam uma dimensão específica do estilo ou da performance.

---

# 22. Ranking de finalizadores

O ranking de finalização não será baseado somente na quantidade absoluta de finalizações.

A ideia inicial é combinar:

```text
Finishing Rate
+
Finishing Volume
+
Opponent Quality
+
Activity
```

Porém, os pesos e a fórmula final deverão ser avaliados empiricamente.

O objetivo é evitar que:

- poucos combates distorçam a taxa;
- grande volume sem eficiência seja supervalorizado;
- adversários muito fracos gerem vantagem artificial.

---

# 23. Avaliação temporal

O sistema não será avaliado utilizando um split aleatório tradicional.

Isso poderia permitir que o modelo treinasse com informações de períodos futuros e fosse testado em períodos anteriores.

A divisão será temporal.

Exemplo:

```text
TREINO
───────────────────────────────>
1994       2005       2015

                         TESTE
                         ───────────────>
                         2016       2020
```

Posteriormente, poderá ser utilizada validação walk-forward:

```text
Treino → Teste
Treino maior → Teste seguinte
Treino maior → Teste seguinte
...
```

Isso reproduz melhor o cenário real do produto.

---

# 24. Métricas de avaliação

A avaliação não será baseada somente em accuracy.

Serão consideradas métricas como:

**Log Loss**

Avalia a qualidade das probabilidades produzidas pelo modelo.

**Brier Score**

Mede a qualidade das probabilidades previstas.

**ROC AUC**

Avalia a capacidade de discriminação entre os resultados.

**Accuracy**

Será utilizada como métrica complementar.

**Calibration**

Também será analisado se:

```text
70% de probabilidade
```

realmente corresponde aproximadamente a resultados favoráveis em torno de 70% em situações semelhantes.

---

# 25. Comparação dos modelos

Cada modelo será executado sobre exatamente o mesmo conjunto temporal de avaliação.

Exemplo:

```text
Modelos

Win Rate
Elo
Elo + Recency
Elo + Opponent Quality
Elo + Performance
Logistic Regression
Random Forest
Gradient Boosting
...
```

Cada modelo será avaliado por:

- Log Loss
- Brier Score
- AUC
- Accuracy
- Calibration

A escolha da abordagem final será baseada no desempenho fora da amostra e na estabilidade dos resultados.

---

# 26. Evitando overfitting

O sistema terá diversas características e será possível encontrar correlações acidentais.

Por isso:

- features serão selecionadas com cuidado;
- modelos simples serão mantidos como baselines;
- hiperparâmetros serão avaliados somente dentro do período de treinamento;
- resultados serão testados em períodos posteriores;
- não serão escolhidas features apenas porque melhoram um único período;
- alterações metodológicas serão registradas.

---

# 27. Reprodutibilidade

Todo resultado deverá poder ser reproduzido a partir dos dados processados e dos scripts do projeto.

A estrutura seguirá aproximadamente:

```text
data/
├── raw/
├── processed/
└── features/

src/
├── build_dataset.py
├── build_features.py
├── build_fight_features.py
├── build_fighter_fight_features.py
├── build_fighter_history.py
└── validate_history.py

docs/
└── rating_methodology.md
```

Novas etapas deverão manter essa separação entre:

```text
Dados brutos
    ↓
Dados processados
    ↓
Features
    ↓
Modelos
    ↓
Resultados
```

---

# 28. Dados ausentes

Dados ausentes não devem ser automaticamente convertidos em zero.

Exemplo:

```text
Takedown: 0 of 0
```

não significa necessariamente que:

```text
Takedown percentage = 0%
```

O percentual pode ser não aplicável.

Da mesma forma, lutas antigas sem estatísticas detalhadas não terão estatísticas inventadas.

Quando uma luta não possui dados de performance:

- resultado
- método
- oponente
- data

continuam disponíveis para modelos que utilizam essas informações.

As features estatísticas permanecem ausentes.

---

# 29. Lutas sem estatísticas

O dataset atualmente possui:

```text
8.887 lutas
```

e:

```text
8.866 lutas com estatísticas de round
21 lutas sem estatísticas de round
```

As 21 lutas sem estatísticas serão preservadas.

Não serão criados valores artificiais para essas lutas.

Isso permite que o sistema continue utilizando informações de resultado e contexto quando apropriado.

---

# 30. Atualização do rating

A arquitetura desejada é:

```text
Estado anterior
      ↓
Nova luta
      ↓
Resultado
      ↓
Atualização do rating
      ↓
Novo estado
```

Exemplo conceitual:

```text
Fighter A
Rating = 1520

Fighter B
Rating = 1480

        ↓

A vence B

        ↓

Atualização

        ↓

A = novo rating
B = novo rating
```

A magnitude da atualização dependerá do modelo selecionado.

---

# 31. Rating vs. previsão

Esses dois conceitos devem permanecer separados.

**Rating**

Responde:

```text
Qual é a força competitiva estimada desse lutador?
```

**Modelo de previsão**

Responde:

```text
Dado A e B, qual é a probabilidade estimada de A vencer?
```

É possível que o melhor sistema de rating não seja exatamente o melhor modelo preditivo.

Por isso, ambos serão avaliados separadamente.

---

# 32. Rating vs. estilo

Também serão mantidos dois conceitos distintos.

**Rating**

representa força competitiva.

**Style Profile**

representa características de luta.

Um lutador pode ter:

```text
Rating elevado
```

sem necessariamente possuir o maior valor em todas as métricas de estilo.

Essa separação será importante para construir posteriormente o mecanismo de matchup.

---

# 33. Arquitetura futura

A visão completa do sistema é:

```text
                    UFC DATA
                       │
                       ▼
               DATA ENGINEERING
                       │
                       ▼
              HISTORICAL STATE
                ┌──────┼──────┐
                │      │      │
                ▼      ▼      ▼
             RATING  METRICS STYLE
                │      │      │
                └──────┼──────┘
                       ▼
                 MATCHUP MODEL
                       │
                       ▼
                  PREDICTION
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
         RANKINGS   PROFILES  SIMULATOR
```

---

# 34. Roadmap metodológico

## Fase 1 — Dados

Concluído:

- ETL dos dados UFCStats;
- resolução de identidades;
- eventos;
- lutadores;
- lutas;
- estatísticas por round;
- validação de integridade.

## Fase 2 — Features

Concluído:

- conversão das estatísticas para formato numérico;
- agregação por luta;
- representação fighter-vs-fighter;
- histórico temporal dos lutadores;
- features pré-luta.

## Fase 3 — Rating

Próximas etapas:

1. Implementar baseline de Win Rate.
2. Implementar Elo.
3. Criar avaliação temporal.
4. Testar Elo com diferentes parâmetros.
5. Adicionar recência.
6. Adicionar qualidade do adversário.
7. Testar performance.
8. Comparar os modelos.

## Fase 4 — Previsão

Implementar e comparar:

- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- outros modelos relevantes.

## Fase 5 — Style Profile

Criar representação multidimensional do estilo dos lutadores.

## Fase 6 — Matchup

Investigar interações entre características dos dois lutadores.

## Fase 7 — Rankings

Construir:

- ranking geral;
- rankings por divisão;
- P4P;
- rankings especializados.

## Fase 8 — Simulator

Criar mecanismo para simular confrontos entre lutadores utilizando os modelos desenvolvidos.

---

# 35. Critérios para considerar o sistema confiável

O sistema não será considerado confiável simplesmente porque produz rankings plausíveis.

Será necessário demonstrar:

- ausência de leakage relevante;
- integridade dos dados;
- avaliação temporal;
- desempenho superior aos baselines relevantes;
- boa calibração das probabilidades;
- estabilidade em diferentes períodos;
- comportamento razoável para lutadores com poucos dados;
- tratamento correto de dados ausentes;
- reprodutibilidade;
- explicabilidade suficiente para compreender por que uma avaliação mudou.

---

# 36. Princípio final

O Ultimate Fighter Rank não pretende simplesmente criar uma fórmula que atribui pontos aos lutadores.

O objetivo é construir um sistema histórico capaz de aprender, a partir dos dados disponíveis, como diferentes fatores se relacionam com a força competitiva e com os resultados das lutas.

A metodologia deve seguir:

```text
Hipótese
   ↓
Implementação
   ↓
Validação temporal
   ↓
Comparação
   ↓
Análise
   ↓
Iteração
```

Em vez de:

```text
Intuição
   ↓
Fórmula fixa
   ↓
Ranking
```

O resultado esperado é um sistema que possa evoluir conforme novos dados e novos experimentos sejam incorporados, mantendo rastreabilidade e evitando decisões metodológicas arbitrárias.

---

# 37. Decisões registradas (v0.1 do baseline de Elo)

Esta seção documenta decisões metodológicas tomadas durante a implementação do primeiro baseline de Elo (`src/build_fighter_ratings.py`), conforme o princípio da seção 26 ("alterações metodológicas serão registradas"). Nenhuma delas é definitiva — todas devem ser revisadas quando a avaliação temporal (seção 23) estiver disponível.

## 37.1 Tratamento de No Contest

No Contest **não atualiza o rating** de nenhum dos dois lutadores — a luta é tratada como se não tivesse ocorrido para fins de força competitiva (`rating_after == rating_before` para ambos).

Entretanto, a luta **conta normalmente** na contagem de experiência (`career_fights_before`, `nc_before`), já que essas contagens são calculadas de forma independente em `build_fighter_history.py`.

## 37.2 Escopo do rating por categoria de peso

Foi adotado um **rating único por lutador, cross-division** — não existe um Elo separado por categoria de peso. A divisão de cada luta é mantida apenas como metadado de contexto (coluna `division`), usada para gerar rankings por divisão a partir do rating mais recente dos lutadores ativos naquela categoria.

Consequência direta desta escolha: ao mudar de divisão, o lutador carrega automaticamente o rating que tinha antes da mudança (não há reinício nem desconto). Isso equivale, na prática, a não ter Elos por divisão separados — só uma dimensão de metadado.

**Risco conhecido e não resolvido:** este desenho assume implicitamente que a escala de rating é comparável entre divisões (um 1600 no Heavyweight representa a mesma força competitiva que um 1600 no Lightweight). Essa suposição não foi validada e deve ser testada empiricamente comparando o desempenho preditivo do modelo em lutas que envolvem trocas de divisão.

**Hipótese explicitamente não incorporada ao baseline:** subir ou descer de divisão pode ter efeitos assimétricos sobre a performance (composição corporal, corte de peso, motivo da troca). O baseline atual trata as duas direções de forma idêntica. A direção da troca (`subiu` / `desceu`) é candidata a virar uma feature de contexto em uma fase posterior (Elo + contexto), a ser avaliada com validação temporal — não deve ser assumida a priori em nenhuma direção.

## 37.3 Pré-requisito de dados: normalização de divisão

A coluna `weight_class`, como vem do ETL, mistura a divisão com o tipo de luta (ex.: `"Lightweight Bout"` vs. `"UFC Lightweight Title Bout"` vs. `"Ultimate Fighter 21 Welterweight Tournament Title Bout"` representam todas a divisão Lightweight/Welterweight). `build_fighter_ratings.py` normaliza esse campo para uma divisão real antes de qualquer lógica de rating.

Lutas de categorias não padrão (`Catch Weight`, `Open Weight`) e lutas de eventos anteriores à existência de categorias de peso no UFC (ex.: torneios do UFC 1–10) não são tratadas como "mudança de divisão" e podem resultar em divisão não identificada (`division = None`). Isso é esperado, não um erro de dados.

## 37.4 K-factor provisório (baixa confiança)

A estreia de um lutador e as primeiras lutas seguintes a uma mudança de divisão usam um K-factor maior (`BOOSTED_K`, atualmente 64, contra `BASE_K` = 32), para permitir uma recalibração mais rápida do rating em cenários de alta incerteza — reaproveitando a mesma lógica de "small sample problem" (seção 13) para o caso de mudança de divisão.

Os valores de `INITIAL_RATING` (1500), `BASE_K` (32), `BOOSTED_K` (64), `SCALE` (400) e `PROVISIONAL_FIGHTS` (3) são pontos de partida arbitrários do baseline v0.1 e devem ser comparados empiricamente com outras configurações antes de serem considerados definitivos.

---

# 38. Decisões registradas (v0.1 da avaliação temporal: Win Rate vs. Elo)

Implementado em `src/evaluate_ratings.py`.

## 38.1 Win Rate como modelo de previsão

Win Rate (`win_rate_before`, já existente em `fighter_history.csv`) é uma taxa individual, não uma probabilidade de A vencer B. A conversão adotada é a normalização mais simples possível, sem inventar fórmula (seção 26):

```text
P(A vence) = win_rate_A / (win_rate_A + win_rate_B)
```

Dois casos recebem probabilidade neutra (0.5), por decisão explícita e não por acaso:

- Estreante (`win_rate_before` ausente) de qualquer um dos dois lados — análogo ao Elo tratar estreantes com `INITIAL_RATING` igual para todo mundo.
- Caso degenerado `win_rate_A + win_rate_B == 0` (os dois lutadores com 0% de aproveitamento até aquele momento).

## 38.2 Elo como modelo de previsão

Nenhum cálculo novo foi necessário: a coluna `expected_score` de `fighter_ratings.csv` já é `P(fighter_id vence)` antes da luta, calculada a partir de `rating_before`. A avaliação apenas reaproveita esse valor do ponto de vista do `fighter_1` de cada luta.

## 38.3 Exclusão de Draw e No Contest da avaliação

Log Loss, Brier Score, ROC AUC e Accuracy pressupõem um rótulo binário (vitória/derrota). Por isso:

- **No Contest** é excluído usando a mesma flag `rating_updated` que o Elo já usa para não atualizar rating em NC (seção 37.1) — nenhuma lógica nova foi criada, apenas reaproveitada.
- **Empate (Draw)** é excluído porque não há "vencedor": tratar y = 0.5 nesse caso misturaria dois tipos de incerteza diferentes (incerteza da previsão do modelo vs. resultado objetivamente sem vencedor) dentro da mesma métrica.

Isso reduz o conjunto avaliável de 8.887 para 8.732 lutas (65 draws + 90 NC excluídos).

## 38.4 Estrutura de avaliação temporal

Duas camadas, ambas implementadas:

1. **Split simples** (harness rápido): treino < 2019-01-01, teste ≥ 2019-01-01, conforme o exemplo da seção 53 do `development_guideline.md`.
2. **Walk-forward expansivo**: primeira janela de teste em 2011-01-01, passo de 3 anos, reproduzindo literalmente o exemplo da seção 23 (treino 1994→2010 / teste 2011-2013; depois treino 1994→2013 / teste 2014-2016; ...). O treino é usado apenas como contexto histórico — Win Rate e Elo v0.1 não têm parâmetros livres para ajustar, então "treino" aqui não significa "fit", e sim "período que já alimentou o rating/win rate acumulado até aquele ponto". Essa distinção passa a importar quando entrarem modelos com parâmetros livres (ex.: Logistic Regression sobre várias features), para os quais a mesma estrutura de fold já está pronta.

## 38.5 Resultado da comparação v0.1 (split simples, teste 2019+)

| Modelo    | Log Loss | Brier  | ROC AUC | Accuracy |
|-----------|---------:|-------:|--------:|---------:|
| Win Rate  |   1.1801 | 0.2626 |  0.5761 |   0.5873 |
| Elo v0.1  |   0.6825 | 0.2448 |  0.5559 |   0.5506 |

O padrão se repete de forma consistente em todas as janelas do walk-forward (ver `data/evaluation/metrics_summary.csv`).

**Nenhum dos dois modelos domina o outro em todas as métricas** — o que é, por si só, uma conclusão válida da seção 25 (a escolha final não pode se basear em uma métrica isolada):

- Win Rate tem melhor **Accuracy** e **ROC AUC**: como sinal bruto de "quem tem mais chance de ganhar", ele discrimina um pouco melhor.
- Elo tem **Log Loss** muito melhor (quase metade). A causa aparece na tabela de calibração (`data/evaluation/calibration_tables.csv`): a normalização do Win Rate produz probabilidades exatas de 0.0 e 1.0 sempre que um lutador tem 100% de aproveitamento e o outro 0%. Quando esse extremo erra, a penalização logarítmica é enorme. O Elo, por construção (função logística sobre a diferença de rating), nunca produz probabilidades exatamente 0 ou 1, então erros de previsão custam caro, mas não catastroficamente caro.
- A calibração do Win Rate no extremo inferior (p ≈ 0) é ruim: nesse bin, a taxa de vitória observada foi de ~52%, longe da probabilidade prevista de ~0%. O Elo não tem previsões nesse extremo para comparar (sua distribuição de probabilidades é mais concentrada perto de 0.5), o que já é um indício indireto de maior robustez.

**Implicação para os próximos passos (seção 53, itens 5-9):** o Elo v0.1 é a base mais segura para evoluir (recência, qualidade do adversário, performance), justamente por não ter esse problema de probabilidades degeneradas. Se o Win Rate normalizado for mantido como baseline de referência, qualquer melhoria futura nele (ex.: suavização tipo Laplace/Bayesian smoothing no numerador e denominador) deve ser tratada como uma hipótese a testar via backtest (seção 57), não como correção automática — para não repetir o erro descrito na seção 26 (fórmula ajustada por parecer intuitiva, sem experimentação).
