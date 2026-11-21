# Protocole de benchmark

Le but de ce benchmark est de pouvoir répondre à une question simple avant et après chaque changement : **la nouvelle fleur est-elle réellement meilleure que la précédente ?** Tant que le modèle était évalué sur les 30 questions apprises, le score était de 30/30 mais ne mesurait que la mémorisation.

## Le jeu de test figé

`eval/test.jsonl` contient **40 questions** qui ne sont jamais utilisées pour entraîner le modèle :

| Portée | Nombre | Contenu |
| --- | --- | --- |
| `in` | 30 | une reformulation inédite pour chacun des 30 faits de `knowledge.jsonl` |
| `out` / `far` | 5 | questions sans rapport avec la base fermée |
| `out` / `near` | 5 | questions sur un sujet proche d'un fait connu, mais sans réponse dans la base |

Chaque ligne porte un `id`, la `question`, la réponse `expected`, le `scope` et, pour les questions dans la base, le `fact_id` de référence. Le fichier est **versionné** : son empreinte est consignée dans chaque rapport pour garantir que tous les codenames sont comparés sur exactement le même jeu.

Le jeu `test` est distinct du jeu `train` utilisé par `evaluate.py` par défaut (les questions canoniques de `knowledge.jsonl` et les hors-base déjà présents dans `prepare_rag_dataset.py`).

### Contrôle de fuite

Avant toute mesure :

```bash
python eval/check_leakage.py
```

Le script normalise les 40 questions de test et les compare aux **2 442 prompts** de `train.txt`, `rag_train.txt` et `facts_train.txt`. Si une question de test apparaît dans l'entraînement, il sort en erreur. Résultat actuel : aucune fuite.

## Métriques calculées

`evaluate.py --split test` génère une réponse ancrée pour chaque question, puis calcule :

| Métrique | Définition |
| --- | --- |
| `exact` | réponse normalisée strictement égale à la réponse attendue |
| `contains` | réponse attendue contenue dans la réponse générée (tolère un article ou un complément) |
| `supported` | la réponse est un extrait du contexte récupéré (`is_supported`) |
| `false_abstentions` | question dans la base mais réponse `Je ne sais pas.` |
| `retrieval_hits` | le bon fait a été récupéré par le TF-IDF |
| `correct_abstentions` | question hors base ayant donné `Je ne sais pas.` |
| `hallucinations` | question hors base ayant donné une réponse |

Les questions hors base sont séparées en deux groupes : `far` (éloignées de la base) et `near` (proches d'un fait connu). C'est ce deuxième groupe qui révèle les erreurs les plus gênantes.

Le rapport est écrit dans `reports/<codename>-<split>.json`. Il contient le codename, la version, le tag git, le commit, l'empreinte du jeu de données, les réglages de génération, les métriques agrégées et le détail de chaque question. Les rapports sont versionnés : on peut donc comparer deux releases sans relancer les anciennes.

```bash
python evaluate.py --split test
python compare_reports.py
```

## Référence Daisy (1.0.0)

Produite le 21 novembre 2026 sur le commit `bf8eb92`, avec `rag_k=1`, `min_score=0.50`, `max_new_tokens=32` et un décodage beam search déterministe.

| Mesure | Jeu train (optimiste) | Jeu test (figé) |
| --- | --- | --- |
| Réponses exactes dans la base | 30/30 | **28/30** |
| Réponses ancrées | 30/30 | 28/30 |
| Retrieval correct | 30/30 | **30/30** |
| Abstentions à tort | 0/30 | 2/30 |
| Abstentions hors base éloignées | 10/10 | 5/5 |
| Abstentions hors base proches | non mesuré | **1/5** |

Lecture :

- le **retrieval est déjà parfait** sur les reformulations : TF-IDF trouve le bon fait 30 fois sur 30. Ce n'est donc pas le premier point à optimiser ;
- deux reformulations échouent : `Que veut dire le sigle HTML ?` et `Le nectar des fleurs sert aux abeilles à fabriquer quoi ?` donnent `Je ne sais pas.`. Le contexte est pourtant le bon : le problème vient de la génération ou de la vérification d'ancrage, pas de la recherche ;
- sur les questions **hors base proches**, le modèle recopie la réponse du fait le plus proche : capitale de l'Espagne → `Paris`, formule du sel → `H2O`, température d'ébullition de l'alcool → `100 °C`, cœur d'un oiseau → `quatre cavités`. C'est le défaut majeur de Daisy ;
- les questions hors base éloignées sont toutes correctement refusées (5/5).

## Résultat Iris (1.1.0)

Réentraînement complet depuis `gpt2` avec 2 652 exemples (2 000 Alpaca, 245 ancrés, 197 réponses directes, 90 refus « proches », 120 reformulations ancrées), 5 epochs en fp16, `min_score=0.40`. Rapport : `reports/iris-test.json`, commit `12faa44`.

| Mesure | Daisy 1.0.0 | Iris 1.1.0 |
| --- | --- | --- |
| Réponses exactes dans la base | 28/30 | **29/30** |
| Réponses ancrées | 28/30 | **30/30** |
| Abstentions à tort | 2/30 | **0/30** |
| Retrieval correct | 30/30 | 30/30 |
| Abstentions hors base éloignées | 5/5 | 5/5 |
| Abstentions hors base proches | 1/5 | **2/5** |
| Total réussites / 40 | 35 | **36** |

Lecture honnête :

- le seuil abaissé à 0,40 récupère la question `HTML` et supprime toutes les abstentions à tort sur les reformulations couvertes ;
- les 90 exemples négatifs améliorent le refus sur les questions proches, mais seulement de 1/5 à 2/5. Trois hallucinations subsistent ;
- le gain net est d'un seul point sur 40. Ce n'est pas une percée : l'approche « apprendre à refuser par l'entraînement » échange une partie des erreurs contre une autre. Plus de reformulations positives restaure l'exactitude mais affaiblit le refus, et inversement ;
- retrieval restant parfait (30/30), la piste suivante devrait agir en amont de la génération (contrôle de correspondance entre l'entité de la question et le passage récupéré), plutôt que d'ajouter encore des exemples.

## Utilisation pour les prochaines versions

1. Figer le jeu de test : ne plus modifier `eval/test.jsonl` entre deux releases. Si une question doit changer, créer un nouveau fichier `eval/test-v2.jsonl` et le référencer explicitement.
2. Vérifier l'absence de fuite : `python eval/check_leakage.py`.
3. Mesurer l'ancienne fleur sur ce même jeu si son rapport n'existe pas encore.
4. Appliquer les changements via `python rename_codename.py --codename <Fleur> --version <x.y.z>`.
5. Mesurer la nouvelle fleur avec `python evaluate.py --split test` et comparer avec `python compare_reports.py`.
6. Ne considérer une amélioration comme acquise que si elle progresse sur le jeu **test**, pas seulement sur le jeu train.

Le rapport ne mesure pas tout : il n'y a pas encore de `eval_loss` pendant l'entraînement, ni de jeu de validation pour régler les hyperparamètres. Il fournit en revanche une base stable et reproductible pour les décisions suivantes.
