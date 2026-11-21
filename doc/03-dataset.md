# 3. Dataset

## Générer les deux jeux de données

La préparation se fait en trois étapes, **dans cet ordre** :

```bash
python prepare_dataset.py --max-examples 2000   # Alpaca filtré → train.txt
python prepare_rag_dataset.py                   # knowledge.jsonl → rag_train.txt
python prepare_facts_dataset.py                 # knowledge.jsonl → facts_train.txt
```

`prepare_dataset.py` télécharge le split `train` de `tatsu-lab/alpaca`, **filtre les exemples invalides** puis écrit jusqu'à **500 exemples valides** dans `train.txt` (ou 2 000 avec `--max-examples 2000`, recommandé). Sont écartés :

- les réponses (ou prompts) sur plusieurs lignes, que le parseur de `train.py` ne peut pas lire ;
- les doublons de couples prompt / réponse ;
- quelques faits erronés connus des premiers exemples Alpaca : l'eau qui bout à 115 °C, le « mole » décrit comme un petit mammifère, ou la réponse `<nooutput>`.

`train.txt` est ignoré par Git.

`prepare_rag_dataset.py` lit `knowledge.jsonl`, une **base fermée de 30 faits en français** (versionnée, champs `id`, `text`, `question`, `answer`) et écrit `rag_train.txt` (**ignoré par Git**) : 8 variantes ancrées par fait (240 exemples), plus 5 abstentions sur des questions **hors base**, soit **245 exemples**. Il doit être lancé après `prepare_dataset.py`, car il importe `rag.py` et suit le même format de blocs.

`prepare_facts_dataset.py` écrit `facts_train.txt` (**ignoré par Git**) : 197 couples prompt/réponse **sans contexte**, générés depuis `knowledge.jsonl`. Chaque fait est présenté avec sa question exacte, sa version en minuscules, sans `?`, préfixée par `Question: `, et quelques reformulations (par exemple `quelle est la capitale de France ?`). Le fichier contient aussi des calculs simples (`que font 2+2 ?` → `4`) et cinq abstentions hors base. C'est ce jeu qui permet au GGUF de répondre directement dans llama.app, sans coller de contexte.

Le détail du pipeline est dans [12-rag.md](12-rag.md).

## Format attendu

`train.py` lit les deux fichiers par blocs séparés par une ligne vide. Chaque bloc doit contenir **exactement deux lignes non vides** :

```
<prompt>
<réponse>

<prompt>
<réponse>
```

Exemple :

```
What is the capital of France?
Paris

Explain photosynthesis in one sentence.
Plants convert light into chemical energy.
```

Si l'instruction Alpaca contient un champ `input`, `prepare_dataset.py` l'intègre sur la même ligne :

```
Write a sentence about X Input: X
X is an example.
```

Cela garantit que le prompt reste sur une seule ligne et que la réponse est bien la deuxième ligne du bloc.

Les exemples de `rag_train.txt` utilisent le même format, avec un prompt au format d'ancrage :

```
Contexte: <texte récupéré> | Question: <question>
<réponse extraite du texte>

Question: <question sans contexte>
Je ne sais pas.
```

## Constantes

Dans `prepare_dataset.py` :

| Constante / option | Valeur | Rôle |
| --- | --- | --- |
| `MAX_EXAMPLES` / `--max-examples` | `500` | Nombre maximum d'exemples Alpaca valides écrits |
| `OUTPUT_FILE` / `--output-file` | `train.txt` | Fichier de sortie |

Dans `prepare_rag_dataset.py` :

| Constante | Valeur | Rôle |
| --- | --- | --- |
| `KB_PATH` | `knowledge.jsonl` | Base de faits lue (versionnée) |
| `OUTPUT_FILE` | `rag_train.txt` | Fichier de sortie (ignoré par Git) |

Dans `prepare_facts_dataset.py` :

| Constante | Valeur | Rôle |
| --- | --- | --- |
| `KB_PATH` | `knowledge.jsonl` | Base de faits lue (versionnée) |
| `OUTPUT_FILE` | `facts_train.txt` | Fichier de sortie (ignoré par Git) |

Dans `train.py` :

| Constante | Valeur | Rôle |
| --- | --- | --- |
| `DATA_PATHS` | `[train.txt, rag_train.txt, facts_train.txt]` | Fichiers lus à l'entraînement |
| `MAX_EXAMPLES` / `--max-examples` | `3000` | Nombre maximum d'exemples, tous fichiers confondus |

L'ancien `DATA_PATH` est remplacé par `DATA_PATHS` : un fichier absent est ignoré avec un avertissement. Le total utilisé pour le dernier entraînement est de 2 442 exemples (2 000 + 245 + 197), sous la limite de 3 000. Si vous agrandissez les jeux, gardez `MAX_EXAMPLES` de `train.py` supérieur au total. Le tokenizer est chargé depuis `PicoGPT1-Iris/` quand il y est, sinon depuis le modèle de départ (évite une requête réseau TLS).

## Dataset personnalisé

Vous pouvez remplacer `train.txt` et/ou `rag_train.txt` par vos propres couples prompt/réponse, à condition de respecter le format ci-dessus. Lancez ensuite `python train.py`.

Pour étendre l'ancrage, modifiez `knowledge.jsonl` puis relancez `python prepare_rag_dataset.py` avant `python train.py`. Pour un format multi-lignes, il faut d'abord adapter le parseur dans `train.py` (`load_datasets`).
