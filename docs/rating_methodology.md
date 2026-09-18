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

Os valores de INITIAL_RATING (1500), BASE_K (32), BOOSTED_K (64), SCALE (400) e PROVISIONAL_FIGHTS (3) foram definidos como pontos de partida arbitrários do baseline v0.1. O primeiro experimento de parâmetros avaliou especificamente PROVISIONAL_FIGHTS e mostrou que 5 apresenta melhor desempenho agregado, embora com variação relevante entre períodos; por isso, conforme registrado na seção 37.5, 3 permanece como configuração oficial do baseline até que um procedimento de seleção de hiperparâmetros com avaliação verdadeiramente fora da amostra seja adotado."

## 37.5 Avaliação experimental de PROVISIONAL_FIGHTS

O baseline v0.1 utilizava PROVISIONAL_FIGHTS = 3 de forma provisória. Para avaliar se a duração dessa janela influencia a qualidade preditiva do Elo, foi realizado um sweep temporal mantendo os demais parâmetros constantes:

INITIAL_RATING = 1500
BASE_K = 32
BOOSTED_K = 64
SCALE = 400

Foram comparados:

PROVISIONAL_FIGHTS = 1, 3, 5, 7, 9

A avaliação utilizou o mesmo harness de walk-forward temporal empregado na seção 38, com seis períodos de teste de três anos. O Log Loss foi utilizado como métrica principal, acompanhado de Brier Score, ROC AUC e Accuracy.

Resultado agregado
PROVISIONAL_FIGHTS	Log Loss	Brier	ROC AUC	Accuracy
1	0.6835	0.2453	0.5515	0.5530
3	0.6822	0.2446	0.5574	0.5597
5	0.6814	0.2442	0.5596	0.5619
7	0.6817	0.2444	0.5570	0.5614
9	0.6821	0.2446	0.5558	0.5604

O valor 5 apresentou o melhor desempenho agregado nas quatro métricas consideradas. Em relação ao baseline 3, apresentou melhora de aproximadamente 0.0008 em Log Loss, 0.0004 em Brier Score, 0.0022 em ROC AUC e 0.0022 em Accuracy.

Entretanto, a análise fold a fold mostrou que o resultado não é uniforme entre os períodos históricos. Considerando Log Loss, os valores que apresentaram o melhor resultado em cada janela foram:

2011–2014 → provisional_5
2014–2017 → provisional_9
2017–2020 → provisional_1
2020–2023 → provisional_9
2023–2026 → provisional_5
2026–2029 → provisional_1

Portanto, nenhum valor dominou todos os períodos. Os valores 1, 5 e 9 venceram dois folds cada, enquanto 3 e 7 não apresentaram o menor Log Loss em nenhum dos seis folds.

O resultado sugere que a duração da janela provisional influencia o desempenho do rating, mas que essa influência depende do período histórico. Não há evidência suficiente para afirmar que um único valor fixo seja universalmente correto.

Decisão metodológica

PROVISIONAL_FIGHTS = 5 será considerado candidato experimental, mas não será incorporado definitivamente ao baseline v0.1 neste momento.

O valor 3 permanece como configuração oficial do baseline enquanto novas etapas metodológicas são investigadas. O resultado do sweep será utilizado como evidência para a evolução futura do tratamento de baixa quantidade de observações.

A análise também reforça a hipótese apresentada na seção 13: o problema de amostras pequenas pode ser melhor tratado por uma representação de incerteza ou por um mecanismo dependente da quantidade e qualidade das observações, em vez de simplesmente escolher uma janela fixa por otimização sobre os mesmos folds de avaliação.

O experimento foi preservado em:

src/param_sweep.py
data/evaluation/param_sweep/summary.csv
data/features/param_sweep/

A escolha definitiva de parâmetros deverá utilizar um procedimento que mantenha períodos de avaliação realmente fora da seleção dos hiperparâmetros, evitando transformar repetidos backtests temporais no mesmo conjunto de ajuste.

## 37.6 Avaliação experimental de K-factor e SCALE

O mesmo script (`src/param_sweep.py`) também testou variações de `BASE_K`/`BOOSTED_K` e de `SCALE`, isolando um parâmetro por vez (seção 26). Resultado agregado (pooled, walk-forward completo 2011→2029, n=7.232, modelo Elo):

| Config                     | base_k | boosted_k | scale | Log Loss | Brier  | AUC    | Accuracy |
|----------------------------|-------:|----------:|------:|---------:|-------:|-------:|---------:|
| baseline_v0.1              |     32 |        64 |   400 |   0.6822 | 0.2446 | 0.5574 |   0.5597 |
| k_menor_16_32              |     16 |        32 |   400 |   0.6853 | 0.2461 | 0.5526 |   0.5567 |
| k_menor_24_48              |     24 |        48 |   400 |   0.6833 | 0.2451 | 0.5551 |   0.5578 |
| sem_boost_k_igual_base     |     32 |        32 |   400 |   0.6847 | 0.2458 | 0.5510 |   0.5610 |
| boost_k_maior_96           |     32 |        96 |   400 |   0.6836 | 0.2452 | 0.5594 |   0.5607 |
| scale_200                  |     32 |        64 |   200 |   0.6839 | 0.2453 | 0.5648 |   0.5653 |
| scale_600                  |     32 |        64 |   600 |   0.6839 | 0.2454 | 0.5542 |   0.5577 |

Nenhuma variação de `K` ou `SCALE` superou o baseline em Log Loss/Brier. `scale_200` apresentou o melhor AUC e Accuracy do grupo, mas piorou Log Loss e Brier — o mesmo padrão de "nenhum modelo domina todas as métricas" já visto na seção 38.5.

**Decisão metodológica:** `BASE_K=32`, `BOOSTED_K=64` e `SCALE=400` permanecem como configuração oficial do baseline v0.1. Não há evidência, nesta rodada, para alterar nenhum dos dois. Combinado com a seção 37.5, a única variação do Elo v0.1 com sinal de melhora consistente encontrada até aqui no `param_sweep.py` é `PROVISIONAL_FIGHTS=5` (candidato experimental, ainda não incorporado).

---

# 39. Experimento de recência

Implementado em `src/recency_sweep.py`, conforme item 6 do roadmap (seção 53 do `development_guideline.md`).

## 39.1 Metodologia

Recência é aplicada como um decaimento do rating em direção ao `INITIAL_RATING`, proporcional ao tempo de inatividade desde a última luta, **antes** de calcular o `expected_score` da próxima luta:

```text
R_efetivo = R_inicial + (R_histórico - R_inicial) * 2^(-Δt / H)
```

onde `Δt` é o número de dias desde a última luta do lutador e `H` é a meia-vida (half-life), em dias. O rating histórico já registrado (`rating_after` de cada luta) não é reescrito — o decaimento afeta apenas o `rating_before` efetivo usado na luta seguinte, preservando a temporalidade (seção 2.1).

Foram testadas cinco meias-vidas (1, 2, 3, 4 e 5 anos) contra o baseline sem recência (Elo v0.1), usando o mesmo protocolo walk-forward de seis folds (2011→2029, n pooled = 7.232).

## 39.2 Resultado agregado (pooled)

| Config        | Half-life | Log Loss | Brier  | AUC    | Accuracy |
|---------------|----------:|---------:|-------:|-------:|---------:|
| baseline      |         — |   0.6822 | 0.2446 | 0.5574 |   0.5597 |
| half_life_1y  |         1 |   0.6852 | 0.2460 | 0.5826 |   0.5622 |
| half_life_2y  |         2 |   0.6816 | 0.2443 | 0.5913 |   0.5736 |
| half_life_3y  |         3 |   0.6801 | 0.2435 | 0.5927 |   0.5796 |
| half_life_4y  |         4 |   0.6794 | 0.2432 | 0.5917 |   0.5810 |
| half_life_5y  |         5 |   0.6791 | 0.2431 | 0.5892 |   0.5783 |

## 39.3 Interpretação

Diferente de todas as extensões testadas até a seção 37.6, **meias-vidas entre 2 e 5 anos melhoram simultaneamente as quatro métricas** em relação ao baseline sem recência — não há o padrão de "cada modelo ganha em uma métrica diferente" observado em Win Rate vs. Elo (seção 38.5) ou no sweep de K/Scale (seção 37.6). Apenas `half_life_1y` (decaimento muito agressivo) piora Log Loss e Brier em relação ao baseline.

Dentro da faixa vencedora, o melhor Log Loss/Brier é de `half_life_5y`, o melhor Accuracy é de `half_life_4y`, e o melhor AUC é de `half_life_3y` — as diferenças entre 3, 4 e 5 anos são pequenas e não foi feita uma análise fold a fold para saber se algum desses três domina de forma consistente ao longo do tempo (o mesmo cuidado da seção 37.5 se aplica aqui: uma média pooled pode esconder inversões por período).

**Decisão metodológica:** a recência é o candidato mais forte encontrado até agora para evoluir o Elo v0.1 — é a única extensão com melhora simultânea em todas as métricas — mas **ainda não foi incorporada à produção** (`src/build_fighter_ratings.py` continua sem decaimento). Antes de promovê-la:

1. repetir a análise fold a fold da seção 37.5 para escolher a meia-vida com evidência robusta ao período, não apenas pooled;
2. decidir se a recência deve substituir o baseline v0.1 ou coexistir com ele como uma camada opcional.

Os artefatos do experimento estão em `data/evaluation/recency/summary.csv`, `data/evaluation/recency/fold_metrics.csv` e `data/features/recency/<config>/fighter_ratings.csv`.

---

# 40. Experimento Glicko-2

Implementado em `src/glicko2_experiment.py`, usando a biblioteca `glicko2==2.1.0`, conforme item "outros modelos de rating" da seção 27.

## 40.1 Metodologia

Glicko-2 é atualizado luta a luta, em ordem cronológica, com os valores padrão da biblioteca: `rating=1500`, `RD=350`, `volatility=0.06`. Cada luta é prevista **antes** de atualizar o rating dos dois lutadores, e NC/empates não entram na avaliação nem atualizam o rating — reaproveitando as mesmas regras já validadas para o Elo (seções 37.1 e 38.3). O mesmo protocolo walk-forward de seis folds (2011→2029) foi usado.

**Limitação conhecida e não resolvida:** esta implementação não aplica o "aging" por calendário do Glicko-2 completo (o aumento de RD — incerteza — durante períodos de inatividade). Isso significa que o principal mecanismo pelo qual o Glicko-2 trataria inatividade de forma nativa não está em uso neste experimento; os resultados abaixo descrevem apenas a atualização luta a luta sem esse componente.

## 40.2 Resultado (pooled, mesmo protocolo da seção 39)

| Modelo      | Log Loss | Brier  | AUC    | Accuracy |
|-------------|---------:|-------:|-------:|---------:|
| Elo v0.1    |   0.6822 | 0.2446 | 0.5574 |   0.5597 |
| Glicko-2    |   0.7037 | 0.2527 | 0.5684 |   0.5684 |

## 40.3 Interpretação

O Glicko-2 (sem aging) tem AUC e Accuracy melhores que o Elo v0.1, mas Log Loss e Brier claramente piores — o mesmo padrão de não-dominância já visto na comparação Win Rate vs. Elo (seção 38.5). Não foi investigada a causa exata da diferença de calibração (não é o mesmo tipo de problema de probabilidades degeneradas 0/1 do Win Rate, já que Glicko-2 também usa uma função logística), então essa explicação não deve ser assumida sem verificação.

**Decisão metodológica:** o Glicko-2, nesta implementação sem aging, não é adotado como substituto do Elo v0.1. Uma versão com aging por calendário — o diferencial conceitual real do Glicko-2 sobre o Elo — não foi testada e é candidata a um experimento futuro, especialmente por ser conceitualmente parecida com a recência da seção 39 (ambas tratam inatividade), mas por um mecanismo diferente (incerteza crescente vs. decaimento do rating).

Artefatos: `data/results/glicko2_temporal_results.csv`, `data/results/glicko2_predictions.csv`.

---

# 41. Experimento de qualidade do adversário (Strength of Schedule)

Implementado em `src/opponent_quality_experiment.py`, conforme item 7 do roadmap (seção 53).

## 41.1 Metodologia

Para cada lutador, calcula-se o Strength of Schedule (SoS): a média dos ratings Elo dos adversários enfrentados até aquele ponto. O modelo combina Elo e SoS via regressão logística:

```text
P(A) = sigmoid(intercept + beta_elo * (Elo_A - Elo_B) + beta_sos * (SoS_A - SoS_B))
```

Os pesos (`beta_elo`, `beta_sos`) são ajustados **apenas com dados anteriores ao período de teste** de cada fold do walk-forward — nunca com o próprio período de teste —, seguindo a mesma disciplina temporal das demais seções.

## 41.2 Resultado (pooled, n=7.232)

| Modelo         | Log Loss | Brier  | AUC    | Accuracy |
|----------------|---------:|-------:|-------:|---------:|
| Elo (só)       |   0.6824 | 0.2447 | 0.5568 |   0.5618 |
| Elo + SoS      |   0.7847 | 0.2758 | 0.5359 |   0.5787 |

A Accuracy melhora, mas o Log Loss piora drasticamente — mais de 0.1 acima do Elo isolado. Por fold, o problema é mais visível: no fold 2011→2014 (o mais escasso em histórico), o Log Loss do modelo Elo+SoS chega a 1.125, contra 0.684 do Elo isolado no mesmo fold — sugerindo previsões extremas e mal calibradas exatamente no período com menos dados por lutador para estimar SoS de forma confiável.

## 41.3 Interpretação

Este é um resultado negativo, não uma falha do experimento: seguindo o princípio da seção 26, o resultado deve ser registrado tal como saiu, sem tentar "consertar" a fórmula depois de ver que ela não funcionou. A hipótese mais provável é que a regressão logística ajustada com poucos exemplos (folds iniciais) produz coeficientes mal calibrados — o mesmo problema de amostra pequena discutido na seção 13, agora manifestado em um modelo de combinação em vez de em um rating individual.

**Decisão metodológica:** SoS via esta combinação logística **não é adotado**. Se o SoS for revisitado, candidatos a investigar antes de tentar novamente incluem: regularização da regressão logística, um SoS com menor variância (ex.: média ponderada pela recência das lutas do adversário) ou um ajuste único sobre todo o histórico pré-2011 em vez de refit por fold — mas nenhuma dessas alternativas foi testada, e nenhuma deve ser assumida como solução sem novo backtest.

Artefatos: `data/results/opponent_quality_temporal_results.csv`, `data/results/opponent_quality_predictions.csv`.

---

# 42. Experimento de performance (striking, grappling, dominância)

Implementado em `src/performance_experiment.py`, conforme item 8 do roadmap (seção 53).

## 42.1 Metodologia

Seis modelos, todos via regressão logística temporalmente ajustada (mesmo cuidado da seção 41.1 — fit somente com dados anteriores a cada fold de teste):

| Modelo | Composição |
|--------|------------|
| M0 | Elo (diferença de rating) |
| M1 | Elo + Striking (golpes significativos, totais, %) |
| M2 | Elo + Grappling (quedas, %) |
| M3 | Elo + Dominância (knockdowns, tempo de controle) |
| M4 | Elo + todas as features de performance acima |
| M5 | Apenas performance, sem Elo |

Princípios seguidos: nenhuma informação futura é usada; o histórico de performance de cada lutador é atualizado somente após cada luta; percentuais são recalculados a partir dos totais acumulados (não é a média dos percentuais por luta); ausência de estatística não é convertida em zero, e sim tratada como valor faltante explícito (com uma coluna indicadora `__missing`); scaler e imputer são ajustados somente no treino de cada fold, nunca no fold de teste inteiro.

## 42.2 Resultado (pooled, n=7.232 por modelo)

| Modelo                    | Log Loss | Brier  | AUC    | Accuracy |
|---------------------------|---------:|-------:|-------:|---------:|
| M4 — Elo + All Performance |   0.6806 | 0.2437 | 0.5931 |   0.5625 |
| M2 — Elo + Grappling       |   0.6813 | 0.2442 | 0.5862 |   0.5523 |
| M0 — Elo                   |   0.6818 | 0.2444 | 0.5830 |   0.5592 |
| M3 — Elo + Dominance       |   0.6821 | 0.2446 | 0.5826 |   0.5502 |
| M1 — Elo + Striking        |   0.6826 | 0.2447 | 0.5865 |   0.5589 |
| M5 — Performance Only      |   0.6884 | 0.2475 | 0.5634 |   0.5390 |

## 42.3 Interpretação

**M4 (Elo + todas as features de performance) é, entre todas as extensões testadas nas seções 37.6 a 42, a única configuração que melhora simultaneamente Log Loss, Brier, AUC e Accuracy em relação ao Elo v0.1 isolado** — não há trade-off entre métricas aqui, diferente de todos os outros experimentos deste documento (exceto a recência, seção 39, que também melhora tudo, mas isoladamente).

M5 (performance sem Elo) é o pior modelo do grupo em todas as métricas, inclusive pior que o Elo isolado — reforçando o princípio da seção 56: estatísticas de performance por si só carregam menos sinal preditivo que o histórico de resultados acumulado no Elo. Elas parecem funcionar como complemento ao Elo, não como substituto.

Os coeficientes por fold (`data/results/performance_coefficients.csv`) não foram analisados quanto à estabilidade entre períodos nesta rodada — antes de qualquer afirmação sobre quais estatísticas específicas (ex.: striking vs. grappling) pesam mais de forma consistente, seria necessário um exame fold a fold equivalente ao da seção 37.5.

**Decisão metodológica:** M4 é o candidato mais forte para produção encontrado até agora, mas **ainda não foi incorporado a `src/build_fighter_ratings.py`**. Antes de promovê-lo, falta: (a) validação fold a fold da estabilidade do ganho (mesmo cuidado da seção 39.3), e (b) uma decisão sobre como combiná-lo com a recência da seção 39, já que os dois experimentos foram avaliados de forma independente e nunca testados juntos.

Artefatos: `data/results/performance_temporal_results.csv`, `data/results/performance_predictions.csv`, `data/results/performance_coefficients.csv`.

---

# 43. Síntese comparativa das extensões do Elo v0.1

Todas as extensões abaixo foram avaliadas com o mesmo protocolo walk-forward (seis folds, 2011→2029, n pooled = 7.232), o que permite compará-las lado a lado:

| Extensão                         | Log Loss | Brier  | AUC    | Accuracy | Melhora em todas as métricas? |
|-----------------------------------|---------:|-------:|-------:|---------:|:---:|
| Elo v0.1 (baseline)                |   0.6822 | 0.2446 | 0.5574 |   0.5597 | — |
| K/Scale (melhor variante, scale_200) |  0.6839 | 0.2453 | 0.5648 |   0.5653 | Não (perde Log Loss/Brier) |
| Recência (half_life_4y)            |   0.6794 | 0.2432 | 0.5917 |   0.5810 | **Sim** |
| Glicko-2 (sem aging)               |   0.7037 | 0.2527 | 0.5684 |   0.5684 | Não (perde Log Loss/Brier) |
| Elo + SoS                          |   0.7847 | 0.2758 | 0.5359 |   0.5787 | Não (piora muito Log Loss) |
| Elo + Performance (M4)             |   0.6806 | 0.2437 | 0.5931 |   0.5625 | **Sim** |

Duas extensões — recência (seção 39) e performance (seção 42, modelo M4) — melhoram todas as quatro métricas em relação ao baseline, de forma independente uma da outra. Nenhuma combinação das duas foi testada até o momento.

**Estado da produção (histórico):** até esta seção, `src/build_fighter_ratings.py` continuava implementando apenas o Elo v0.1 puro (seção 37). Nenhuma das extensões desta seção 43 havia sido incorporada à produção. A seção 44, abaixo, registra a validação fold a fold e a decisão de promoção (item 9.5 do roadmap, seção 53 do `development_guideline.md`).

---

# 44. Validação fold a fold e promoção para produção (v0.2)

Implementado em `src/combined_experiment.py`, reaproveitando quase toda a lógica de `src/performance_experiment.py` (parsing, agregação de performance, pipeline logístico, métricas) e a fórmula de recência já validada em `src/recency_sweep.py` (seção 39.1).

## 44.1 Objetivo

A seção 43 identificou dois resultados de sinal positivo e consistente na média *pooled* — recência (seção 39) e Elo + performance / M4 (seção 42) — mas nenhum dos dois havia sido validado fold a fold, e os dois nunca haviam sido testados juntos. Este experimento resolve as duas pendências apontadas na seção 43 e no item "EM DESENVOLVIMENTO" do `development_guideline.md` (seção 52):

1. validar fold a fold a estabilidade de recência e performance, individualmente;
2. testar a combinação das duas (Elo + recência + performance) contra cada uma isolada e contra o Elo v0.1 puro;
3. decidir o que promover para produção.

Mesmo protocolo walk-forward de `performance_experiment.FOLDS` (seis folds, 2011→2029) em todas as comparações abaixo.

## 44.2 Validação fold a fold das extensões já existentes

Reanalisando os artefatos já existentes (`data/evaluation/recency/fold_metrics.csv` e `data/results/performance_temporal_results.csv`) fold a fold, e não apenas pela média pooled da seção 43:

**Recência:** `half_life_4y`/`half_life_5y` vencem em Log Loss 5 dos 6 folds; perdem apenas no fold 2014→2017, por margem pequena (0.0013–0.0020) frente ao baseline sem recência.

**Performance (M4):** M4 melhora o Log Loss frente ao Elo puro (M0) em 5 dos 6 folds; piora apenas no fold 2014→2017 (margem 0.0047). Isoladamente, M4 raramente é o vencedor absoluto do fold (outros modelos de performance parcial, como M2, vencem pontualmente), mas domina consistentemente o Elo puro.

**Conclusão:** as duas extensões têm evidência fold a fold, não apenas pooled — ambas vencem o Elo v0.1 puro em 5 de 6 janelas temporais, com o mesmo fold (2014→2017) como exceção nas duas. Isso é suficiente para atender ao critério da seção 37.5/39.3 ("não há evidência suficiente" antes disso).

## 44.3 Recência + Performance combinadas

`src/combined_experiment.py` recalcula o Elo internamente (mesma fórmula de decaimento da seção 39.1) e usa o rating efetivo (pós-recência) como `elo_difference` de entrada de uma regressão logística com as mesmas features de performance de M4 (seção 42), ajustada apenas com dados de treino de cada fold (mesma disciplina das seções 41.1/42.1).

Modelos comparados: `M0` (Elo puro), `Recency_only` (Elo + recência, sem performance), `M4` (Elo puro + performance), `M4_Recency` (Elo + recência + performance).

### Resultado agregado (pooled, n=7.232)

| Modelo             | Log Loss | Brier  | AUC    | Accuracy |
|--------------------|---------:|-------:|-------:|---------:|
| M0                 |   0.6818 | 0.2444 | 0.5824 |   0.5592 |
| M4                 |   0.6806 | 0.2437 | 0.5941 |   0.5625 |
| Recency_only (4y)  |   0.6794 | 0.2432 | 0.6058 |   0.5756 |
| Recency_only (5y)  |   0.6791 | 0.2430 | 0.6054 |   0.5757 |
| **M4_Recency (4y)**|   **0.6749** | **0.2409** | **0.6104** | **0.5801** |
| M4_Recency (5y)    |   0.6750 | 0.2410 | 0.6099 |   0.5796 |

`M4_Recency` (recência + performance combinadas) supera tanto cada extensão isolada quanto o Elo puro nas quatro métricas, com meia-vida de 4 e 5 anos praticamente empatadas (diferença de milésimos, sem vencedor consistente entre elas fold a fold).

### Validação fold a fold do modelo combinado (half-life 4 anos)

| Fold      | M0     | Recency_only | M4     | M4_Recency |
|-----------|-------:|-------------:|-------:|-----------:|
| 2011–2014 | 0.6823 | 0.6824       | 0.6803 | 0.6776     |
| 2014–2017 | 0.6785 | 0.6809       | 0.6832 | 0.6831     |
| 2017–2020 | 0.6894 | 0.6827       | 0.6863 | 0.6770     |
| 2020–2023 | 0.6808 | 0.6788       | 0.6779 | 0.6726     |
| 2023–2026 | 0.6794 | 0.6753       | 0.6777 | 0.6686     |
| 2026–2029 | 0.6784 | 0.6727       | 0.6730 | 0.6622     |

`M4_Recency` vence 5 dos 6 folds em Log Loss — o mesmo padrão de robustez das duas extensões isoladas (seção 44.2), mas agora com margens maiores frente ao Elo puro (até -0.0162 no fold 2026→2029, contra -0.0076/-0.0054 de recência/performance isoladas no mesmo fold). O único fold onde perde é, de novo, 2014→2017 (margem pequena, +0.0046 frente ao M0) — o mesmo fold "difícil" para ambas as extensões desde a seção 39.3/42.3. Nenhuma variação testada até agora domina esse fold específico; não foi investigada a causa (candidato a experimento futuro, não assumido sem novo backtest).

## 44.4 Decisão metodológica

Diferente de todas as decisões anteriores deste documento (K/Scale, Glicko-2, SoS — seções 37.6, 40, 41), este resultado passa nos dois critérios da seção 26 (validação fora da amostra) e do item 9.5 do roadmap (estabilidade fold a fold, não só pooled):

1. **Recência (half-life = 4 anos) é promovida ao rating de produção.** `src/build_fighter_ratings.py` ganhou um `PRODUCTION_CONFIG` (`EloConfig(half_life_years=4.0)`), usado por `main()`/`data/features/fighter_ratings.csv`. O `DEFAULT_CONFIG` (sem recência) permanece inalterado para não quebrar `src/param_sweep.py` e para preservar o Elo v0.1 puro como referência histórica. As colunas novas de `fighter_ratings.csv` — `rating_historical_before` (implícito: é o `rating_after` da luta anterior, sem decaimento), `days_since_last_fight`, `recency_factor` — tornam a recência auditável (seção 58 — explicabilidade).

   A meia-vida 5 anos não foi escolhida por não ter vantagem fold a fold consistente sobre 4 anos (seção 44.3); 4 anos também é o valor de melhor Accuracy pooled já registrado na seção 39.3.

2. **Performance (M4) NÃO é promovida ao rating em si.** Seguindo a distinção da seção 31 (rating vs. modelo de previsão), a regressão logística de M4/M4_Recency responde "qual a probabilidade de A vencer B", não "qual a força competitiva de A" — não há uma forma direta de embutir coeficientes de uma regressão logística multivariada dentro de um único número de Elo sem perder a interpretação de rating. A combinação `Elo (com recência) + performance`, validada nesta seção como a mais forte encontrada até agora, fica registrada como a configuração de referência para a Fase 4 do roadmap (Previsão — seção 53), a ser retomada quando o projeto começar a construir o modelo de previsão de produção propriamente dito (distinto do rating).

3. **O fold 2014→2017 permanece como um caso não resolvido**, perdendo para o Elo puro em toda extensão testada até agora (recência, performance, e a combinação). Não deve ser tratado como motivo para descartar as extensões (a evidência agregada e fold a fold nos outros 5 folds é forte), mas também não deve ser ignorado — é candidato a investigação futura (ex.: eventos atípicos do período, mudança na composição das divisões, ou uma limitação genuína do modelo).

Artefatos: `data/results/combined_temporal_results.csv`, `data/results/combined_predictions.csv`.

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
