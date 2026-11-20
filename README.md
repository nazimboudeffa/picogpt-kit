# PicoGPT1-Daisy

**Un mini GPT-2 entraîné de zéro, du dataset jusqu'au chat dans le navigateur.**

`PicoGPT1-Daisy` est un projet pédagogique qui montre le cycle complet d'un petit modèle de langage : préparation d'un jeu de données d'instructions, entraînement d'un GPT-2 minuscule en Python, génération en ligne de commande ou via un serveur HTTP local, export au format GGUF, publication sur Hugging Face et test avec [llama.app](https://llama.app).

Tous les scripts Python fonctionnent **sans llama.cpp ni Ollama** : ils utilisent directement les poids Transformers. Le format GGUF n'est produit que pour une utilisation optionnelle avec llama.cpp / llama.app.

> ⚠️ **Modèle expérimental** : 745 exemples d'entraînement (500 Alpaca filtrés + 245 exemples d'ancrage) et 6,86 M paramètres. Sans ancrage, les réponses sont souvent approximatives, voire incohérentes. L'objectif est de comprendre et de faire fonctionner la chaîne complète, pas de fournir un assistant généraliste.

---

## Sommaire

1. [Présentation](#1-présentation)
2. [Structure du dépôt](#2-structure-du-dépôt)
3. [Prérequis](#3-prérequis)
4. [Installation](#4-installation)
5. [Démarrage rapide](#5-démarrage-rapide)
6. [Préparer le dataset](#6-préparer-le-dataset)
7. [Entraîner le modèle](#7-entraîner-le-modèle)
8. [Générer une réponse avec `test.py`](#8-générer-une-réponse-avec-testpy)
9. [Serveur HTTP local avec `serve.py`](#9-serveur-http-local-avec-servepy)
10. [Exporter en GGUF](#10-exporter-en-gguf)
11. [Publier sur Hugging Face](#11-publier-sur-hugging-face)
12. [Utiliser le modèle publié](#12-utiliser-le-modèle-publié)
13. [Architecture du modèle](#13-architecture-du-modèle)
14. [Dépannage](#14-dépannage)
15. [Limites et pistes d'amélioration](#15-limites-et-pistes-damélioration)
16. [Liens et références](#16-liens-et-références)

---

## 1. Présentation

### Ce que fait le projet

| Étape | Script | Résultat |
| --- | --- | --- |
| Données | `prepare_dataset.py` | `train.txt` (jusqu'à 500 couples prompt / réponse Alpaca validés) |
| Connaissances | `knowledge.jsonl` | Base fermée de 30 faits en français (versionnée) |
| Données ancrées | `prepare_rag_dataset.py` | `rag_train.txt` (245 exemples, ignoré par Git) |
| Entraînement | `train.py` | Poids Transformers dans `PicoGPT1-Daisy/` |
| Ancrage | `rag.py` | Réponse extraite du contexte ou `Je ne sais pas.` |
| Évaluation | `evaluate.py` | Exactitude, ancrage, abstentions, retrieval |
| Test CLI | `test.py` | Affichage d'une réponse générée (avec ou sans `--rag`) |
| API locale | `serve.py` | `POST /generate` sur `http://127.0.0.1:8000` |
| Export | `export_gguf.py` | `PicoGPT1-Daisy.gguf` (~26,7 Mo) |
| Publication | `publish_hf.py` | Dépôt Hugging Face public |

### État actuel

| Élément | Détail |
| --- | --- |
| Entraînement de référence | 745 exemples, 10 epochs, ~1 860 steps, loss finale ≈ 4,54 |
| Jeu de données | `train.txt` (500 exemples Alpaca filtrés) + `rag_train.txt` (245 exemples ancrés / abstention) |
| Base de connaissances | `knowledge.jsonl` : 30 faits en français, versionnée |
| Évaluation RAG | 30/30 réponses ancrées, 10/10 abstentions hors base |
| Poids locaux | `PicoGPT1-Daisy/model.safetensors` (**non versionné**) |
| Modèle publié | [nazimboudeffa/PicoGPT1-Daisy](https://huggingface.co/nazimboudeffa/PicoGPT1-Daisy) |
| Compatibilité llama.app | Confirmée via `llama cli -hf ...` |
| Versions utilisées | Python 3.13 / Windows 11, `transformers` 4.52.4, `huggingface_hub` 0.33.0, `torch` CPU |

Les notebooks `train.ipynb` et `test.ipynb` sont conservés comme référence historique ; les scripts Python en sont l'équivalent direct et sont la voie recommandée.

Le pipeline d'ancrage (RAG) est détaillé dans [doc/12-rag.md](doc/12-rag.md).

---

## 2. Structure du dépôt

```
.
├── PicoGPT1-Daisy/          # modèle entraîné (config, tokenizer, poids)
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors      # généré, ignoré par Git
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── checkpoint-1000/       # checkpoint Trainer, ignoré par Git
├── doc/                       # documentation détaillée, chapitre par chapitre
├── evaluate.py                # métriques du pipeline RAG
├── export_gguf.py             # Transformers → GGUF
├── knowledge.jsonl            # base fermée de 30 faits (versionnée)
├── model_utils.py             # chargement du modèle + génération (partagé)
├── prepare_dataset.py         # téléchargement Alpaca → train.txt
├── prepare_rag_dataset.py     # knowledge.jsonl → rag_train.txt
├── publish_hf.py              # publication du modèle sur Hugging Face
├── rag.py                     # TF-IDF, copie contrainte, abstention
├── serve.py                   # serveur HTTP local
├── test.py                    # génération en ligne de commande
├── train.py                   # entraînement
├── train.ipynb                # référence historique
├── test.ipynb                 # référence historique
├── train.txt                  # dataset généré, ignoré par Git
├── rag_train.txt              # dataset ancré généré, ignoré par Git
├── PicoGPT1-Daisy.gguf      # export GGUF, ignoré par Git
├── MODEL_CARD.md              # fiche modèle publiée sur le Hub
└── README.md                  # ce fichier
```

**Rôle des fichiers Python**

| Fichier | Responsabilité |
| --- | --- |
| `prepare_dataset.py` | Télécharge un sous-ensemble de `tatsu-lab/alpaca`, filtre les exemples invalides et écrit `train.txt` |
| `prepare_rag_dataset.py` | Génère `rag_train.txt` depuis `knowledge.jsonl` (à lancer après `prepare_dataset.py`) |
| `train.py` | Construit le GPT-2, entraîne sur `train.txt` et `rag_train.txt`, sauvegarde dans `PicoGPT1-Daisy/` |
| `rag.py` | TF-IDF sur `knowledge.jsonl`, copie contrainte, vérification d'ancrage et abstention |
| `evaluate.py` | Mesure exactitude, ancrage, abstentions et retrieval du pipeline RAG |
| `model_utils.py` | `load_model()` et `generate()`, utilisés par `test.py`, `serve.py` et `rag.py` |
| `test.py` | Interface CLI pour générer une réponse, avec ou sans ancrage (`--rag`) |
| `serve.py` | Serveur HTTP local basé sur la bibliothèque standard |
| `export_gguf.py` | Télécharge le convertisseur llama.cpp épinglé et produit le `.gguf` |
| `publish_hf.py` | Crée le dépôt Hub et upload poids, fiche modèle et GGUF |

Pour approfondir chaque partie, consultez [doc/README.md](doc/README.md), et [doc/12-rag.md](doc/12-rag.md) pour l'ancrage.

---

## 3. Prérequis

- **Python 3.10 ou supérieur** (testé avec Python 3.13)
- **Git**
- Windows, macOS ou Linux
- Environnement **CPU suffisant** : l'entraînement complet prend quelques minutes sur CPU
- Accès Internet pour télécharger le dataset, les dépendances et (si besoin) le modèle publié
- **scikit-learn** (`sklearn`), nécessaire au pipeline d'ancrage (`rag.py`, `evaluate.py`)

Optionnels :

- Un compte [Hugging Face](https://huggingface.co) avec un token d'écriture, uniquement pour publier
- Le CLI [llama.app](https://llama.app) / llama.cpp, uniquement pour tester le GGUF

---

## 4. Installation

### 4.1 Cloner le dépôt

```bash
git clone https://github.com/nazimboudeffa/picogpt1-daisy.git
cd picogpt1-daisy
```

### 4.2 Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers datasets huggingface_hub scikit-learn
```

`tokenizers`, `accelerate`, `safetensors`, `requests`, `urllib3` et `scipy` sont installés automatiquement comme dépendances. Aucun `requirements.txt` n'est nécessaire. `scikit-learn` fournit le module `sklearn` utilisé par le TF-IDF de `rag.py`.

### 4.4 Vérifier l'installation

```bash
python -m py_compile train.py test.py serve.py model_utils.py prepare_dataset.py prepare_rag_dataset.py rag.py evaluate.py export_gguf.py publish_hf.py
```

Aucune sortie signifie que la syntaxe est correcte. Vous pouvez aussi vérifier les versions :

```bash
python -c "import torch, transformers, datasets, huggingface_hub, sklearn; print(torch.__version__, transformers.__version__, datasets.__version__, huggingface_hub.__version__, sklearn.__version__)"
```

### 4.5 Connexion Hugging Face (facultatif)

Nécessaire **uniquement** pour `publish_hf.py` :

```bash
huggingface-cli login
```

Le token est saisi dans l'invite interactive. Ne le collez jamais dans un chat, un script ou un fichier versionné. La variable d'environnement `HF_TOKEN` est également acceptée.

### 4.6 Installer le CLI llama.app (facultatif)

```bash
# Windows
winget install llama.cpp

# macOS / Linux
curl -LsSf https://llama.app/install.sh | sh

# Vérification
llama --version
```

Sous Windows, l'application [llama.app](https://llama.app) peut aussi être installée directement.

### 4.7 Attention : les poids ne sont pas dans Git

Le fichier `.gitignore` exclut `*.safetensors`, `*.gguf`, `*.pt`, `train.txt`, `rag_train.txt` et les checkpoints. Après un clonage, **il n'y a donc pas de modèle utilisable**. Deux solutions :

1. **Entraîner localement** : `python prepare_dataset.py`, `python prepare_rag_dataset.py` puis `python train.py` (recommandé pour comprendre le projet). `knowledge.jsonl`, lui, est versionné.
2. **Télécharger le modèle publié** via llama.app (voir [section 12](#12-utiliser-le-modèle-publié)).

---

## 5. Démarrage rapide

Enchaînement complet, depuis le dépôt cloné :

```bash
python -m pip install torch transformers datasets huggingface_hub scikit-learn
python prepare_dataset.py       # crée train.txt (exemples Alpaca filtrés)
python prepare_rag_dataset.py   # crée rag_train.txt depuis knowledge.jsonl
python train.py                 # entraîne sur les deux fichiers, sauvegarde dans PicoGPT1-Daisy/
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python evaluate.py              # métriques d'ancrage et d'abstention
python serve.py                 # API locale sur le port 8000
```

Publication et export (nécessitent `huggingface-cli login`) :

```bash
python export_gguf.py
python publish_hf.py
```

> Le `\n` à la fin du prompt reproduit le séparateur utilisé pendant l'entraînement. `generate()` l'ajoute désormais automatiquement s'il manque, mais vous pouvez l'écrire explicitement : sous PowerShell, utilisez `` `n `` à la place : `--prompt "What is 2+2?`n"`.

---

## 6. Préparer le dataset

La préparation des données se fait maintenant en **deux étapes**, dans cet ordre :

```bash
python prepare_dataset.py       # 1. Alpaca filtré → train.txt
python prepare_rag_dataset.py   # 2. knowledge.jsonl → rag_train.txt
```

`prepare_dataset.py` télécharge le split `train` de [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) et écrit jusqu'à **500 exemples valides** dans `train.txt`. Les exemples dont la réponse s'étale sur plusieurs lignes, les doublons et quelques faits erronés connus (l'eau qui bout à 115 °C, le « mole » présenté comme un mammifère, `<nooutput>`) sont écartés : le parseur ne lit que deux lignes par bloc.

`prepare_rag_dataset.py` doit être lancé **après** `prepare_dataset.py`. Il lit `knowledge.jsonl` (base fermée de 30 faits en français, champs `id`, `text`, `question`, `answer`, **versionnée**) et écrit `rag_train.txt` (**ignoré par Git**) : 8 variantes ancrées par fait, soit 240 exemples, auxquels s'ajoutent 5 exemples d'abstention sans contexte — **245 exemples au total**.

Le détail du pipeline est dans [doc/12-rag.md](doc/12-rag.md).

### Format attendu

`train.py` lit les deux fichiers (`train.txt` et `rag_train.txt`) par blocs séparés par une ligne vide. Chaque bloc doit contenir **exactement deux lignes non vides** : le prompt puis la réponse.

```
<prompt>
<réponse>

<prompt>
<réponse>
```

Exemple concret :

```
What is the capital of France?
Paris

Explain photosynthesis in one sentence.
Plants convert light into chemical energy.
```

Si l'exemple Alpaca possède un champ `input`, il est ajouté sur la même ligne que l'instruction (`Write a sentence about X Input: X`) afin que le prompt reste sur une seule ligne.

Les exemples de `rag_train.txt` suivent le même format, mais leur prompt est au format d'ancrage :

```
Contexte: <texte récupéré> | Question: <question>
<réponse extraite du texte>

Question: <question sans contexte>
Je ne sais pas.
```

### Constantes

| Fichier | Constante | Défaut | Rôle |
| --- | --- | --- | --- |
| `prepare_dataset.py` | `MAX_EXAMPLES` | `500` | Nombre maximum d'exemples Alpaca valides écrits |
| `prepare_dataset.py` | `OUTPUT_FILE` | `train.txt` | Fichier de sortie |
| `prepare_rag_dataset.py` | `KB_PATH` | `knowledge.jsonl` | Base de faits lue (versionnée) |
| `prepare_rag_dataset.py` | `OUTPUT_FILE` | `rag_train.txt` | Fichier de sortie (ignoré par Git) |
| `train.py` | `DATA_PATHS` | `[train.txt, rag_train.txt]` | Fichiers lus à l'entraînement |
| `train.py` | `MAX_EXAMPLES` | `2000` | Nombre maximum d'exemples, tous fichiers confondus |

L'ancienne constante `DATA_PATH` est remplacée par `DATA_PATHS` : un fichier manquant est simplement ignoré avec un avertissement. Le jeu actuel fait 745 exemples (500 + 245), sous la limite de 2 000. Si vous changez la taille des jeux, gardez `MAX_EXAMPLES` de `train.py` supérieur au total.

### Dataset personnalisé

Remplacez `train.txt` et/ou `rag_train.txt` par vos propres couples prompt / réponse en respectant le format ci-dessus, puis relancez `python train.py`. Pour des réponses sur plusieurs lignes, il faut d'abord adapter le parseur `load_datasets()` dans `train.py`. Pour étendre l'ancrage, modifiez `knowledge.jsonl` puis relancez `python prepare_rag_dataset.py` avant l'entraînement.

---

## 7. Entraîner le modèle

```bash
python train.py
```

`train.txt` et `rag_train.txt` doivent exister (voir section précédente). Le second fichier peut manquer : `train.py` affiche alors un avertissement et continue avec les exemples Alpaca seuls.

### Constantes configurables

En tête de `train.py` :

| Constante | Défaut | Description |
| --- | --- | --- |
| `DATA_PATHS` | `[train.txt, rag_train.txt]` | Fichiers d'exemples (remplace l'ancien `DATA_PATH`) |
| `OUTPUT_DIR` | `PicoGPT1-Daisy` | Dossier de sauvegarde |
| `MAX_LEN` | `256` | Longueur maximale du contexte |
| `MAX_EXAMPLES` | `2000` | Nombre maximum d'exemples, tous fichiers confondus |
| `EPOCHS` | `10` | Nombre d'epochs |
| `BATCH_SIZE` | `4` | Taille du batch |
| `LEARNING_RATE` | `5e-4` | Taux d'apprentissage |

> `MAX_LEN` est également défini dans `model_utils.py` pour tronquer les prompts à la génération. Les deux valeurs doivent rester identiques.

### Ce que fait le script

1. Charge le tokenizer depuis `PicoGPT1-Daisy/` s'il y est (même vocabulaire, pas de requête réseau TLS), sinon depuis `gpt2`. Il utilise `<|endoftext|>` comme token EOS et PAD.
2. Définit un `chat_template` correspondant au format appris `prompt\nreponse<|endoftext|>`.
3. Lit successivement `train.txt` et `rag_train.txt` et tokenise chaque exemple (tronqué à 256 tokens).
4. Masque les tokens du prompt dans les labels (`-100`) : la loss ne porte que sur la réponse.
5. Construit un `GPT2LMHeadModel` (2 couches, 128 dimensions, 2 têtes).
6. Entraîne avec le `Trainer` de Transformers (`logging_steps=50`, `save_steps=1000`).
7. Sauvegarde le modèle et le tokenizer dans `PicoGPT1-Daisy/`.

### Durée et sorties

Sur CPU, comptez quelques minutes pour la configuration par défaut. La référence actuelle porte sur **745 exemples, 10 epochs, ~1 860 steps, avec une loss finale ≈ 4,54** — valeurs élevées mais attendues avec un modèle aussi petit.

Fichiers générés dans `PicoGPT1-Daisy/` : `config.json`, `generation_config.json`, `model.safetensors`, `merges.txt`, `vocab.json`, `tokenizer.json`, `special_tokens_map.json`, `tokenizer_config.json`.

Un dossier de checkpoint (`checkpoint-1000/` avec ~1 860 steps) peut apparaître : il est ignoré par Git et peut être supprimé sans risque, les poids finaux sont dans `model.safetensors`. Après un réentraînement, pensez à régénérer le GGUF (section 10).

---

## 8. Générer une réponse avec `test.py`

```bash
python test.py
```

Sans argument, le prompt est `What is the capital of France?\n` et le script affiche le texte complet (prompt + réponse).

### Options

```bash
python test.py --prompt "Explique la psychanalyse.\n" --do-sample
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--model-dir` | `PicoGPT1-Daisy` | Dossier du modèle entraîné |
| `--prompt` | `What is the capital of France?\n` | Texte à compléter |
| `--max-new-tokens` | `64` | Nombre maximum de tokens générés |
| `--do-sample` | activé | Échantillonnage (défaut, évite les boucles) |
| `--greedy` | désactivé | Décodage glouton (déconseillé) |
| `--temperature` | `0.8` | Température (si échantillonnage) |
| `--top-p` | `0.9` | Nucleus sampling (si échantillonnage) |
| `--top-k` | `40` | Limite aux 40 tokens les plus probables |
| `--repetition-penalty` | `1.15` | Pénalité de répétition |
| `--no-repeat-ngram-size` | `3` | Interdit la répétition de trigrammes |
| `--rag` | désactivé | Ancre la réponse dans `knowledge.jsonl` |
| `--rag-k` | `1` | Nombre de passages récupérés à considérer |
| `--min-score` | `0.50` | Score TF-IDF minimal pour fournir un contexte |

L'échantillonnage est activé par défaut : le décodage glouton fait boucler ce petit modèle. La génération s'arrête à l'EOS ou au premier saut de ligne produit après la réponse (format appris : `prompt\nréponse<|endoftext|>`). Terminez votre prompt par `\n` pour rester proche du format d'entraînement.

Exemples :

```bash
# bash
python test.py --prompt "What is the capital of France?\n"

# PowerShell
python test.py --prompt "What is 2+2?`n" --max-new-tokens 80 --top-k 20

# Ancrage RAG : réponse courte, extraite du contexte
python test.py --prompt "Quelle est la capitale de la France ?" --rag
```

### Génération ancrée (`--rag`)

Avec `--rag`, `test.py` passe par `rag.py` : un passage de `knowledge.jsonl` est récupéré par TF-IDF, inséré dans le prompt au format `Contexte: <texte> | Question: <question>\n`, puis la réponse est contrainte à rester un extrait littéral du contexte. Si le score de récupération est inférieur à `--min-score` (0,50 par défaut), aucun contexte n'est fourni et le script renvoie immédiatement `Je ne sais pas.`

```bash
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python test.py --prompt "Quel est le prix du pain à Paris ?" --rag --min-score 0.5
python test.py --prompt "À quelle température l'eau bout-elle ?" --rag --rag-k 3 --min-score 0.4
```

Contrairement au mode sans ancrage, seul le texte de la réponse est affiché (pas le prompt complet). Le prompt se termine toujours par `\n` : `generate()` l'ajoute automatiquement s'il manque, car ce séparateur était masqué pendant l'entraînement. Voir [doc/12-rag.md](doc/12-rag.md).

### Depuis votre propre code Python

Toute la logique est dans `model_utils.py` :

```python
from model_utils import load_model, generate

tokenizer, model = load_model("PicoGPT1-Daisy")
print(generate(tokenizer, model, "What is 2+2?\n", do_sample=True))
```

`load_model()` charge tokenizer + modèle, passe en mode évaluation et neutralise `generation_config.max_length` pour éviter un conflit avec `max_new_tokens`. `generate()` tronque le prompt à 256 tokens (en ajoutant le `\n` final si besoin), applique les réglages anti-boucles (échantillonnage, `top_k=40`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`), s'arrête à l'EOS ou au premier saut de ligne généré, puis renvoie le texte décodé. Les paramètres `return_full_text`, `prefix_allowed_tokens_fn` et `num_beams` sont décrits à la section 13.

Si les poids sont absents, le message suivant s'affiche :

```
Poids introuvables dans 'PicoGPT1-Daisy'. Lancez d'abord `python train.py`.
```

---

## 9. Serveur HTTP local avec `serve.py`

Le modèle peut être exposé par une petite API locale, sans framework web ni Ollama : seule la bibliothèque standard de Python est utilisée. Le modèle reste chargé en mémoire pendant toute la durée du serveur.

### Démarrage

```bash
python serve.py
```

Sortie attendue :

```
Serveur prêt : http://127.0.0.1:8000/generate
```

### Route `POST /generate`

Champs du corps JSON :

| Champ | Type | Défaut | Description |
| --- | --- | --- | --- |
| `prompt` | string | obligatoire | Texte à compléter |
| `max_new_tokens` | int | `64` | Nombre maximum de tokens générés |
| `do_sample` | bool | `true` | Échantillonnage (déconseillé de mettre `false`) |
| `temperature` | float | `0.8` | Température |
| `top_p` | float | `0.9` | Nucleus sampling |
| `top_k` | int | `40` | Limite de tokens candidats |
| `repetition_penalty` | float | `1.15` | Pénalité de répétition |
| `no_repeat_ngram_size` | int | `3` | Anti-répétition de trigrammes |
| `use_rag` | bool | `false` | Active l'ancrage dans `knowledge.jsonl` |
| `rag_k` | int | `1` | Nombre de passages récupérés à considérer |
| `min_score` | float | `0.50` | Score TF-IDF minimal pour fournir un contexte |

Réponses :

```json
{ "generated_text": "prompt + réponse" }
```

Quand `use_rag` vaut `true`, `generated_text` contient **uniquement la réponse courte** (extrait ancré ou `Je ne sais pas.`), pas le prompt complet.

```json
{ "error": "message d'erreur" }
```

Toute autre route renvoie `404`.

### Exemples d'appel

**curl (bash / macOS / Linux)**

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is psychoanalysis?\n","do_sample":true}'

# Avec ancrage RAG
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Quelle est la capitale de la France ?","use_rag":true,"rag_k":1,"min_score":0.5}'
```

**PowerShell**

```powershell
$body = @{
    prompt = "What is psychoanalysis?`n"
    max_new_tokens = 64
    do_sample = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -Body $body -ContentType "application/json"
```

**Python**

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/generate",
    json={"prompt": "What is 2+2?\n", "do_sample": True},
)
print(r.json()["generated_text"])
```

### Options CLI

```bash
python serve.py --host 127.0.0.1 --port 9000 --model-dir PicoGPT1-Daisy
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Adresse d'écoute |
| `--port` | `8000` | Port d'écoute |
| `--model-dir` | `PicoGPT1-Daisy` | Dossier du modèle |

Arrêtez le serveur avec `Ctrl+C`.

---

## 10. Exporter en GGUF

Les scripts Python utilisent `model.safetensors`. llama.cpp / llama.app attendent un fichier **GGUF** : `export_gguf.py` fait la conversion.

```bash
python export_gguf.py
```

Prérequis : `PicoGPT1-Daisy/model.safetensors` doit exister (`python train.py`).

### Fonctionnement

1. Télécharge `convert_hf_to_gguf.py` depuis une version épinglée de llama.cpp (`b4000`).
2. Installe le paquet Python `gguf` dans un dossier temporaire s'il manque.
3. Lance la conversion avec `--outtype f16`.
4. Supprime les fichiers temporaires.

Résultat : `PicoGPT1-Daisy.gguf` (~26,7 Mo, 29 tenseurs), ignoré par Git. Le `chat_template` présent dans `tokenizer_config.json` est conservé dans le GGUF.

### Vérification locale

```bash
llama cli -m PicoGPT1-Daisy.gguf -st -p "What is 2+2?" -n 20
```

Après chaque réentraînement, relancez l'export puis la publication pour garder poids Transformers et GGUF synchronisés :

```bash
python export_gguf.py
python publish_hf.py
```

### Après l'ajout du pipeline RAG

Les exemples `rag_train.txt` font désormais partie de l'entraînement : après `python prepare_rag_dataset.py` et `python train.py`, **pensez à régénérer le GGUF** avec `python export_gguf.py` avant de le tester dans llama.app, puis à le republier si besoin. Le GGUF ne contient ni `knowledge.jsonl` ni le code TF-IDF : l'ancrage (`--rag`, `use_rag`) fonctionne uniquement côté scripts Python, avec `scikit-learn` installé.

---

## 11. Publier sur Hugging Face

Le modèle est publié publiquement ici : **[nazimboudeffa/PicoGPT1-Daisy](https://huggingface.co/nazimboudeffa/PicoGPT1-Daisy)**.

### Prérequis

- `huggingface-cli login` effectué (ou `HF_TOKEN` défini)
- Poids Transformers présents : `python train.py`
- GGUF présent : `python export_gguf.py`
- `MODEL_CARD.md` présent à la racine

### Commande

```bash
python publish_hf.py
```

Le script :

1. Crée le dépôt s'il n'existe pas (`private=False`, `exist_ok=True`).
2. Upload le contenu de `PicoGPT1-Daisy/` en ignorant `checkpoint-*`.
3. Upload `MODEL_CARD.md` comme `README.md` du dépôt Hub.
4. Upload `PicoGPT1-Daisy.gguf`.

Une session HTTP avec retries est configurée pour absorber les erreurs TLS intermittentes de certains réseaux. En cas d'échec, relancez simplement la commande.

---

## 12. Utiliser le modèle publié

### Avec le CLI llama.app

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Daisy -hff PicoGPT1-Daisy.gguf
```

`-hff` force le fichier GGUF à utiliser. Comme le dépôt ne contient qu'un seul GGUF, la forme courte fonctionne aussi :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Daisy
```

### Interface web de chat

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Daisy -hff PicoGPT1-Daisy.gguf
# puis ouvrir http://localhost:9931
```

### Avec Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nazimboudeffa/PicoGPT1-Daisy"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Quelle est la capitale de la France ?\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=48, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Avec le serveur local

Si vous avez entraîné vos propres poids, `python serve.py` expose directement `PicoGPT1-Daisy/`. Pour un autre dossier :

```bash
python serve.py --model-dir chemin/vers/mon/modele --port 9000
```

---

## 13. Architecture du modèle

Configuration définie dans `train.py` et sauvegardée dans `PicoGPT1-Daisy/config.json` :

| Paramètre | Valeur |
| --- | --- |
| `model_type` | `gpt2` |
| `architectures` | `GPT2LMHeadModel` |
| `n_layer` | `2` |
| `n_embd` | `128` |
| `n_head` | `2` |
| `n_positions` / `n_ctx` | `256` |
| `vocab_size` | `50 257` |
| `activation_function` | `gelu_new` |
| `torch_dtype` | `float32` |

Soit environ **6,86 millions de paramètres**.

### Tokenizer et format appris

- Tokenizer GPT-2 (`GPT2TokenizerFast`), vocabulaire de 50 257 tokens.
- `<|endoftext|>` (ID `50256`) sert de token EOS, BOS, UNK et PAD.
- Chaque exemple est tokenisé comme `{prompt}\n{reponse}<|endoftext|>`.
- Les tokens du prompt, y compris le `\n` final, ont un label `-100` : la loss ne porte que sur la réponse.

### Chat template

Sauvegardé dans `tokenizer_config.json` et repris dans le GGUF :

```jinja
{% for message in messages %}{{ message['content'] }}{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n' }}{% endif %}
```

Les messages sont séparés par des sauts de ligne, et un saut de ligne est ajouté lorsque llama.cpp demande le début de la réponse.

### Génération

`model_utils.generate()` utilise `max_new_tokens` (64 par défaut), `pad_token_id = eos_token_id = 50256`, l'échantillonnage par défaut (`temperature=0.8`, `top_p=0.9`, `top_k=40`, `repetition_penalty=1.15`), `no_repeat_ngram_size=3` et un arrêt à l'EOS ou au premier saut de ligne généré. Le `\n` final du prompt est ajouté automatiquement s'il manque, car ce séparateur était masqué à l'entraînement.

### Génération contrainte pour le RAG

La signature de `generate()` comprend aussi trois paramètres utilisés par `rag.py` :

| Paramètre | Défaut | Rôle |
| --- | --- | --- |
| `return_full_text` | `True` | Avec `False`, ne renvoie que la réponse générée (utile pour l'ancrage). |
| `prefix_allowed_tokens_fn` | `None` | Fonction `(batch_id, input_ids)` qui limite les tokens autorisés à chaque étape. `rag.py` n'autorise que les tokens maintenant la réponse comme extrait littéral du contexte, et exclut les mots de plus de 2 caractères déjà présents dans la question. |
| `num_beams` | `1` | Beam search ; `rag.py` passe à `5` avec `do_sample=False` pour un décodage déterministe. |

`rag.py` vérifie ensuite `is_supported(answer, context)` : si la réponse n'est pas un extrait du contexte, il renvoie `Je ne sais pas.`. Voir [doc/12-rag.md](doc/12-rag.md).

---

## 14. Dépannage

| Symptôme | Solution |
| --- | --- |
| `Poids introuvables dans 'PicoGPT1-Daisy'` | Les poids ne sont pas versionnés : lancez `python train.py` (puis `python export_gguf.py` pour le GGUF). |
| `Token is required ... no token found` | Connectez-vous avec `huggingface-cli login` ou définissez `HF_TOKEN`. |
| Erreur SSL pendant la publication | Les retries sont déjà configurés ; relancez `python publish_hf.py`. |
| Le modèle ne génère rien avec llama.cpp | Régénérez le GGUF après le dernier entraînement : `python export_gguf.py`, puis testez avec `llama cli ...`. |
| Réponses incohérentes | C'est attendu avec un jeu réduit et 6,86 M paramètres. L'échantillonnage et l'arrêt au saut de ligne limitent les boucles, pas les erreurs factuelles : utilisez `--rag` pour ajouter un contexte fiable. |
| `ModuleNotFoundError: No module named 'sklearn'` | Installez la dépendance : `python -m pip install scikit-learn`. Elle est requise par `rag.py`, `test.py --rag`, `serve.py` avec `use_rag` et `evaluate.py`. |
| `rag_train.txt introuvable, ignoré.` | Lancez `python prepare_rag_dataset.py` après `python prepare_dataset.py`, puis réentraînez avec `python train.py`. |
| Le mode `--rag` répond toujours `Je ne sais pas.` | Le score TF-IDF est sous le seuil : vérifiez `knowledge.jsonl`, reformulez la question ou baissez `--min-score` (ex. `0.4`). Les questions hors base doivent, elles, s'abstenir. |
| Le contexte récupéré n'est pas le bon | Augmentez `--rag-k`, reformulez la question, ou enrichissez `knowledge.jsonl` puis relancez `python prepare_rag_dataset.py`. |
| Port 8000 déjà utilisé | `python serve.py --port 9000`. |
| `LF will be replaced by CRLF...` sous Windows | Avertissement Git sans conséquence. |
| Dossier de checkpoint (`checkpoint-1000/`) | Créé par `Trainer`, ignoré par Git : supprimable sans risque. |

---

## 15. Limites et pistes d'amélioration

### Limites actuelles

- Jeu de données réduit : 500 exemples Alpaca filtrés + 245 exemples d'ancrage.
- Architecture très petite (2 couches, 128 dimensions) et contexte de 256 tokens.
- Aucun jeu de validation : le loss affiché est uniquement le loss d'entraînement.
- Échantillonnage par défaut avec arrêt au saut de ligne ; le mode `--greedy` boucle encore.
- GGUF exporté en F16, sans quantification.

### Limites du pipeline RAG

- Base **fermée de 30 faits** : aucune question hors de `knowledge.jsonl` ne peut recevoir de réponse factuelle (abstention attendue).
- Sur les 30 questions de la base, seulement **9 réponses exactes** : l'extrait copié est toujours supporté par le contexte (30/30), mais ce n'est pas toujours le meilleur.
- Seuil `DEFAULT_MIN_SCORE = 0.50` à recalibrer selon les reformulations et la taille de la base.
- Dépendance à **scikit-learn** (`sklearn`) pour le TF-IDF ; l'ancrage n'existe pas dans le GGUF / llama.cpp.
- Aucune validation croisée du retrieval ni du seuil : `evaluate.py` fournit un diagnostic ponctuel, pas un jeu de validation.

Résultats détaillés dans [doc/12-rag.md](doc/12-rag.md).

### Améliorations possibles

1. Augmenter `MAX_EXAMPLES` dans `prepare_dataset.py` **et** `train.py` (limite actuelle : 2 000).
2. Passer à 4 couches / 256 dimensions, ou partir d'un GPT-2 pré-entraîné.
3. Réserver 10 % des exemples pour un jeu de validation et suivre `eval_loss`.
4. Ajuster `temperature`, `top_p`, `top_k`, `repetition_penalty` et `no_repeat_ngram_size` selon l'usage.
5. Exporter le GGUF en `Q4_K_M` pour réduire sa taille et accélérer l'inférence.
6. Étendre `knowledge.jsonl`, recalibrer `--min-score` et comparer plusieurs valeurs de `--rag-k` avec `evaluate.py`.
7. Pousser les commits locaux sur GitHub si le dépôt doit être partagé.

---

## 16. Liens et références

- Dépôt GitHub : <https://github.com/nazimboudeffa/picogpt1-daisy>
- Modèle Hugging Face : <https://huggingface.co/nazimboudeffa/PicoGPT1-Daisy>
- Dataset : <https://huggingface.co/datasets/tatsu-lab/alpaca>
- Documentation détaillée : [doc/README.md](doc/README.md)
- Pipeline RAG / ancrage : [doc/12-rag.md](doc/12-rag.md)
- Fiche modèle : [MODEL_CARD.md](MODEL_CARD.md)
- llama.app : <https://llama.app>
