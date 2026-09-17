# Resolução de identidade

O nome textual do lutador não será usado como chave primária.

A chave canônica será o ID extraído da URL do UFCStats:

```text
http://ufcstats.com/fighter-details/98a58c26c5b1ed17
                                      └──────────────┘
                                           fighter_id
```

## Exceções já auditadas

Os seguintes nomes aparecem duplicados em `ufc_fighter_details.csv` e também nas estatísticas:

```text
Bruno Silva       -> 12ebd7d157e91701
Jean Silva        -> 52ef95b5860fb28c
Joey Gomez        -> 0778f94eb5d588a5
Michael McDonald  -> d0314416a7f26527
Mike Davis        -> fb3e61720be4690c
Victor Valenzuela -> 078695e385ec2f57
```

Existem também 13 nomes das estatísticas que não têm correspondência exata em `FIRST + LAST`, mas foram encontrados no `ufc_fighter_tott.csv` com URL.

Essas exceções devem ser representadas explicitamente no pipeline, e não corrigidas diretamente nos CSVs brutos.
