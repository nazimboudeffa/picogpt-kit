# 12. Ancrage / RAG

Ce chapitre décrit le pipeline d'ancrage ajouté au projet : le modèle ne répond plus librement, il doit **copier un extrait d'un texte récupéré** dans une petite base de connaissances, ou s'abstenir. Aucun changement d'architecture n'est nécessaire : il s'agit d'un retriever TF-IDF, d'un prompt au format appris et d'un décodage contraint.

## Principe

1. `knowledge.jsonl` contient une **base fermée de 30 faits en français**. Chaque ligne a les champs `id`, `text`, `question`, `answer`. Ce fichier est versionné.
2. `rag.py` indexe `text + question` avec un **TF-IDF sur les caractères** (`analyzer="char_wb"`, ngrammes de 3 à 5). Indexer aussi la question améliore la correspondance quand la question reformule le fait.
3. Le meilleur passage est retenu uniquement si son score atteint `DEFAULT_MIN_SCORE = 0.50`. Sinon, **aucun contexte n'est fourni** et la fonction renvoie immédiatement `Je ne sais pas.`
4. Sinon, le prompt est construit au format appris par le modèle :

   ```
   Contexte: <texte récupéré> | Question: <question>
   ```

   Le saut de ligne final est ajouté automatiquement par `generate()` s'il manque : à l'entraînement, ce séparateur était masqué dans les labels.

5. Le décodage est contraint par `prefix_allowed_tokens_fn` : seuls les tokens qui **maintiennent la réponse comme extrait littéral du contexte** sont autorisés. Les mots de plus de 2 caractères déjà présents dans la question sont exclus, pour éviter que le modèle ne recopie « la capitale de la France » au lieu d'extraire « Paris ». Le décodage est déterministe (`do_sample=False`, `num_beams=5`).
6. Enfin, `is_supported()` normalise et vérifie que la réponse est bien un extrait du contexte. Si ce n'est pas le cas, le modèle s'abstient.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `knowledge.jsonl` | Base fermée de 30 faits en français, champs `id`, `text`, `question`, `answer` (**versionné**) |
| `prepare_rag_dataset.py` | Génère `rag_train.txt` depuis `knowledge.jsonl` ; à lancer **après** `prepare_dataset.py` |
| `rag_train.txt` | 245 exemples d'entraînement : 8 variantes ancrées par fait (240) + 5 exemples d'abstention sans contexte (**ignoré par Git**) |
| `rag.py` | TF-IDF, construction du prompt, copie contrainte, `is_supported()`, `ABSTENTION = "Je ne sais pas."` |
| `evaluate.py` | Mesure exactitude, ancrage, abstentions et retrieval |
| `test.py` | Flags `--rag`, `--rag-k`, `--min-score` |
| `serve.py` | Champs JSON `use_rag`, `rag_k`, `min_score` |
| `model_utils.py` | `generate()` avec `return_full_text=False`, `prefix_allowed_tokens_fn` et `num_beams` |

La dépendance **scikit-learn** (`sklearn`) est requise pour le TF-IDF :

```bash
python -m pip install scikit-learn
```

## Commandes

Préparation des données, puis entraînement sur les deux jeux :

```bash
python prepare_dataset.py       # train.txt : exemples Alpaca filtrés
python prepare_rag_dataset.py   # rag_train.txt : exemples ancrés + abstention
python train.py                 # DATA_PATHS = [train.txt, rag_train.txt]
```

`train.py` utilise `DATA_PATHS = [train.txt, rag_train.txt]` (l'ancien `DATA_PATH` a disparu) et `MAX_EXAMPLES = 2000`. Le jeu actuel fait **745 exemples**, chargés en 10 epochs (~1 860 steps, loss finale ≈ 4,54). Le tokenizer est pris dans `PicoGPT1-Daisy/` s'il y est, sinon depuis `gpt2` : cela évite une requête réseau TLS et garantit le même vocabulaire.

Génération et évaluation :

```bash
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python test.py --prompt "Quel est le prix du pain à Paris ?" --rag --min-score 0.5
python evaluate.py
```

Via le serveur local, ajoutez les champs au corps JSON :

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Quelle est la capitale de la France ?","use_rag":true,"rag_k":1,"min_score":0.5}'
```

Quand `use_rag` est vrai, la réponse est courte (extrait ancré ou `Je ne sais pas.`), pas le prompt complet.

Après un réentraînement intégrant `rag_train.txt`, pensez à régénérer le GGUF (`python export_gguf.py`) : le GGUF ne contient ni `knowledge.jsonl` ni le code TF-IDF, l'ancrage n'existe que dans les scripts Python.

## Format des données et du prompt

Une ligne de `knowledge.jsonl` :

```json
{"id": "capital_fr", "text": "La capitale de la France est Paris.", "question": "Quelle est la capitale de la France ?", "answer": "Paris"}
```

`prepare_rag_dataset.py` écrit des blocs de deux lignes, identiques à ceux de `train.txt` :

```
Contexte: La capitale de la France est Paris. | Question: Quelle est la capitale de la France ?
Paris

Question: Quelle est la capitale de la France ?
Je ne sais pas.
```

Les variantes ancrées présentent le fait avec 0 à 4 autres faits avant/après, afin que le modèle apprenne à extraire la bonne réponse plutôt qu'à mémoriser un texte unique. Les 5 exemples d'abstention n'ont **jamais** de contexte. Les questions hors base sont gérées à l'inférence par `is_supported()` et le seuil.

## Réglage du seuil

- `--min-score` (CLI), `min_score` (API) et `DEFAULT_MIN_SCORE = 0.50` dans `rag.py` définissent le même seuil.
- Si le score du meilleur passage est **inférieur à 0,50**, aucun contexte n'est inséré et l'abstention est renvoyée immédiatement. Le code commente des scores d'environ 0,74 et plus pour les questions couvertes, et inférieurs à 0,40 hors base.
- Baisser le seuil augmente la couverture mais le risque d'accepter une question hors base ; le monter rend l'abstention plus fréquente. Recalibrez avec `evaluate.py` après avoir modifié `knowledge.jsonl`.
- `--rag-k` (défaut `1`, champ `rag_k` côté API) contrôle le nombre de passages récupérés ; la construction du prompt n'utilise toutefois que le meilleur passage (`matches[0]`).

## Métriques actuelles

Commande : `python evaluate.py`

| Mesure | Résultat |
| --- | --- |
| Questions dans la base | 30 |
| Réponses exactes | 9/30 |
| Réponses ancrées | 30/30 |
| Abstentions à tort | 0/30 |
| Retrieval correct | 30/30 |
| Questions hors base | 10 |
| Abstentions correctes | 10/10 |

Interprétation : le modèle ne fabrique plus de fait absent du contexte ; l'extrait choisi n'est pas toujours le meilleur (9 exacts), mais il est toujours supporté par le texte récupéré.

## Limites

- Base **fermée de 30 faits** : aucune connaissance générale, seulement ce qui est dans `knowledge.jsonl`.
- Seuil 0,50 à recalibrer selon les reformulations et la taille de la base.
- 9 extraits exacts sur 30 : l'ancrage garantit la plausibilité, pas la pertinence optimale de l'extrait.
- Dépendance à **scikit-learn** ; pas d'ancrage dans le GGUF / llama.cpp.
- Aucun jeu de validation : `evaluate.py` fournit un diagnostic ponctuel, pas une validation croisée.

Pour le contexte général, voir [11-limites.md](11-limites.md) et le [README racine](../README.md).
