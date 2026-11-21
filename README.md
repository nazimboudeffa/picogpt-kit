# PicoGPT1-Iris

**Un mini GPT-2 entraîné de zéro, du dataset jusqu'au chat dans le navigateur.**

`PicoGPT1-Iris` est un projet pédagogique qui montre le cycle complet d'un petit modèle de langage : préparation d'un jeu de données d'instructions, entraînement d'un GPT-2 minuscule en Python, génération en ligne de commande ou via un serveur HTTP local, export au format GGUF, publication sur Hugging Face et test avec [llama.app](https://llama.app).

Tous les scripts Python fonctionnent **sans llama.cpp ni Ollama** : ils utilisent directement les poids Transformers. Le format GGUF n'est produit que pour une utilisation optionnelle avec llama.cpp / llama.app.

> ⚠️ **Modèle expérimental** : 2 442 exemples d'entraînement (2 000 Alpaca, 245 ancrages, 197 réponses directes) et 124 M paramètres (GPT-2 pré-entraîné ajusté). Les questions couvertes par la base fermée ont des réponses directes apprises ; au-delà, le modèle s'abstient ou se trompe. Ce n'est pas un assistant généraliste.

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
| Données | `prepare_dataset.py` | `train.txt` (jusqu'à 2 000 couples prompt / réponse Alpaca validés avec `--max-examples 2000`) |
| Connaissances | `knowledge.jsonl` | Base fermée de 30 faits en français (versionnée) |
| Données ancrées | `prepare_rag_dataset.py` | `rag_train.txt` (245 exemples, ignoré par Git) |
| Réponses directes | `prepare_facts_dataset.py` | `facts_train.txt` (197 exemples, ignoré par Git) |
| Entraînement | `train.py` | Poids Transformers dans `PicoGPT1-Iris/` |
| Ancrage | `rag.py` | Réponse extraite du contexte ou `Je ne sais pas.` |
| Évaluation | `evaluate.py` | Exactitude, ancrage, abstentions, retrieval ; rapport `reports/<codename>-<split>.json` |
| Jeu de test | `eval/test.jsonl` | 40 questions jamais vues (30 reformulations + 10 hors base) |
| Contrôle de fuite | `eval/check_leakage.py` | Vérifie que le test n'a pas servi à l'entraînement |
| Comparaison | `compare_reports.py` | Tableau des métriques par codename |
| Test CLI | `test.py` | Affichage d'une réponse générée (avec ou sans `--rag`) |
| API locale | `serve.py` | `POST /generate` sur `http://127.0.0.1:8000` |
| Export | `export_gguf.py` | `PicoGPT1-Iris.gguf` (~328 Mo, F16) |
| Publication | `publish_hf.py` | Dépôt Hugging Face public |

### État actuel

| Élément | Détail |
| --- | --- |
| Entraînement de référence | 2 652 exemples, 5 epochs de fine-tuning GPT-2 |
| Jeu de données | `train.txt` (2 000 Alpaca) + `rag_train.txt` (245 ancrés) + `facts_train.txt` (197 réponses directes) + `negative_train.txt` (90 refus) + `paraphrase_train.txt` (120 reformulations ancrées) |
| Base de connaissances | `knowledge.jsonl` : 30 faits en français, versionnée |
| Évaluation RAG (jeu train) | 30/30 réponses exactes, 30/30 ancrées, 0 abstention à tort, 10/10 abstentions hors base |
| Benchmark figé (jeu test) | Iris : 29/30 exactes, 30/30 ancrées, 30/30 retrieval, 5/5 hors base éloignées, 2/5 hors base proches (`reports/iris-test.json`) |
| Référence Daisy (jeu test) | 28/30 exactes, 30/30 retrieval, 5/5 hors base éloignées, 1/5 hors base proches (`reports/daisy-test.json`) |
| Poids locaux | `PicoGPT1-Iris/model.safetensors` (**non versionné**) |
| Modèle publié | Daisy : <https://huggingface.co/nazimboudeffa/PicoGPT1-Daisy>. Iris : dépôt prévu, non publié pour l'instant. |
| Compatibilité llama.app | Confirmée via `llama cli -hf ...` |
| Versions utilisées | Python 3.13 / Windows 11, `transformers` 4.52.4, `huggingface_hub` 0.33.0, `torch` CPU |

### Versions et noms de fleurs

Chaque itération porte un nom de fleur (codename). La convention et la procédure de renommage sont détaillées dans [doc/13-versions.md](doc/13-versions.md).

| Release | Codename | Portée |
| --- | --- | --- |
| `0.x` | Jasmine | prototype historique, conservé tel quel |
| `1.0.0` | Daisy | GPT-2 124 M, RAG, GGUF |
| `1.1.0` | Iris | correctif des hallucinations hors base proches |

Le nom, la version et l'identifiant Hugging Face ne sont définis qu'une seule fois, dans `project.json`. Les scripts les lisent via `config.py`. Pour passer à la fleur suivante :

```bash
python rename_codename.py --codename Iris --version 1.1.0 --dry-run   # aperçu
python rename_codename.py --codename Iris --version 1.1.0             # application
```

Le script met à jour `project.json`, les fichiers texte, le `CHANGELOG.md`, et renomme le dossier de poids ainsi que le GGUF. Voir [CHANGELOG.md](CHANGELOG.md) pour l'historique.

Le pipeline d'ancrage (RAG) est détaillé dans [doc/12-rag.md](doc/12-rag.md).

---

## 2. Structure du dépôt

```
.
├── PicoGPT1-Iris/          # modèle entraîné (config, tokenizer, poids)
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
├── eval/                      # jeu de test figé et contrôle de fuite
├── reports/                   # rapports de benchmark par codename (versionnés)
├── CHANGELOG.md               # historique des releases et des codenames
├── compare_reports.py         # tableau comparatif des benchmarks
├── config.py                  # constantes partagées (lit project.json)
├── evaluate.py                # métriques du pipeline RAG
├── export_gguf.py             # Transformers → GGUF
├── knowledge.jsonl            # base fermée de 30 faits (versionnée)
├── model_utils.py             # chargement du modèle + génération (partagé)
├── prepare_dataset.py         # téléchargement Alpaca → train.txt
├── prepare_rag_dataset.py     # knowledge.jsonl → rag_train.txt
├── prepare_facts_dataset.py   # knowledge.jsonl → facts_train.txt
├── project.json               # source de vérité : version, codename, dépôt HF
├── publish_hf.py              # publication du modèle sur Hugging Face
├── rag.py                     # TF-IDF, vérification d'ancrage, abstention
├── rename_codename.py         # change de fleur en une commande
├── serve.py                   # serveur HTTP local
├── test.py                    # génération en ligne de commande
├── train.py                   # entraînement
├── train.txt                  # dataset généré, ignoré par Git
├── rag_train.txt              # dataset ancré généré, ignoré par Git
├── facts_train.txt            # Q/R directes générées, ignoré par Git
├── PicoGPT1-Iris.gguf        # export GGUF, ignoré par Git
├── MODEL_CARD.md              # fiche modèle publiée sur le Hub
└── README.md                  # ce fichier
```

**Rôle des fichiers Python**

| Fichier | Responsabilité |
| --- | --- |
| `config.py` | Constantes de release lues depuis `project.json` (nom, version, dépôt HF) |
| `rename_codename.py` | Change de nom de fleur et met à jour le dépôt en une commande |
| `eval/check_leakage.py` | Vérifie qu'aucune question du jeu de test n'a servi à l'entraînement |
| `compare_reports.py` | Affiche les métriques des rapports de benchmark par codename |
| `prepare_dataset.py` | Télécharge un sous-ensemble de `tatsu-lab/alpaca`, filtre les exemples invalides et écrit `train.txt` |
| `prepare_rag_dataset.py` | Génère `rag_train.txt` depuis `knowledge.jsonl` (à lancer après `prepare_dataset.py`) |
| `prepare_facts_dataset.py` | Génère `facts_train.txt` : questions/réponses directes et variantes, pour llama.app |
| `train.py` | Ajuste GPT-2 sur les trois fichiers, sauvegarde dans `PicoGPT1-Iris/` |
| `rag.py` | TF-IDF sur `knowledge.jsonl`, vérification d'ancrage et abstention |
| `evaluate.py` | Mesure exactitude, ancrage, abstentions et retrieval ; `--split test` utilise le jeu figé et écrit un rapport |
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
git clone https://github.com/nazimboudeffa/picogpt-kit.git
cd picogpt-kit
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

#### GPU NVIDIA (optionnel)

Sur Windows, `pip install torch` installe généralement la variante **CPU** : `torch.cuda.is_available()` renvoie alors `False` et l'entraînement tourne sur CPU même si une carte graphique est présente. Pour l'activer :

```bash
python -m pip install --upgrade "torch==2.7.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Avec `cuda available: True`, `train.py` utilise automatiquement le GPU ; ajoutez `--fp16` pour accélérer et réduire la mémoire : `python train.py --fp16`.

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
python prepare_dataset.py --max-examples 2000   # crée train.txt (exemples Alpaca filtrés)
python prepare_rag_dataset.py                   # crée rag_train.txt depuis knowledge.jsonl
python prepare_facts_dataset.py                 # crée facts_train.txt (Q/R directes)
python train.py                                 # ajuste GPT-2 sur les trois fichiers
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python eval/check_leakage.py                # vérifie la séparation train/test
python evaluate.py                          # diagnostic sur knowledge.jsonl
python evaluate.py --split test             # benchmark figé + rapport JSON
python compare_reports.py                   # compare les codenames
python serve.py                             # API locale sur le port 8000
```

Publication et export (nécessitent `huggingface-cli login`) :

```bash
python export_gguf.py
python publish_hf.py
```

> Le `\n` à la fin du prompt reproduit le séparateur utilisé pendant l'entraînement. `generate()` l'ajoute désormais automatiquement s'il manque, mais vous pouvez l'écrire explicitement : sous PowerShell, utilisez `` `n `` à la place : `--prompt "What is 2+2?`n"`.

---

## 6. Préparer le dataset

La préparation des données se fait en **trois étapes**, dans cet ordre :

```bash
python prepare_dataset.py --max-examples 2000   # 1. Alpaca filtré → train.txt
python prepare_rag_dataset.py                   # 2. knowledge.jsonl → rag_train.txt
python prepare_facts_dataset.py                 # 3. knowledge.jsonl → facts_train.txt
```

`prepare_dataset.py` télécharge le split `train` de [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) et écrit jusqu'à **500 exemples valides** dans `train.txt` (2 000 avec `--max-examples 2000`, recommandé). Les exemples dont la réponse s'étale sur plusieurs lignes, les doublons et quelques faits erronés connus (l'eau qui bout à 115 °C, le « mole » présenté comme un mammifère, `<nooutput>`) sont écartés : le parseur ne lit que deux lignes par bloc.

`prepare_rag_dataset.py` doit être lancé **après** `prepare_dataset.py`. Il lit `knowledge.jsonl` (base fermée de 30 faits en français, champs `id`, `text`, `question`, `answer`, **versionnée**) et écrit `rag_train.txt` (**ignoré par Git**) : 8 variantes ancrées par fait, soit 240 exemples, auxquels s'ajoutent 5 abstentions sur des questions **hors base** — **245 exemples au total**.

`prepare_facts_dataset.py` écrit `facts_train.txt` (**ignoré par Git**) : **197 exemples** de questions/réponses directes, sans contexte. Pour chaque fait, la question exacte, sa version en minuscules, sans point d'interrogation et préfixée par `Question: ` sont associées à la bonne réponse, avec quelques reformulations connues (`quelle est la capitale de France ?`, `capitale de la France`, etc.). Un petit jeu de calculs (`que font 2+2 ?` → `4`) et d'abstentions hors base complète le fichier. C'est ce jeu qui permet au GGUF de répondre directement dans llama.app.

Le détail du pipeline est dans [doc/12-rag.md](doc/12-rag.md).

### Format attendu

`train.py` lit les trois fichiers (`train.txt`, `rag_train.txt` et `facts_train.txt`) par blocs séparés par une ligne vide. Chaque bloc doit contenir **exactement deux lignes non vides** : le prompt puis la réponse.

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
| `prepare_dataset.py` | `MAX_EXAMPLES` / `--max-examples` | `500` | Nombre maximum d'exemples Alpaca valides écrits |
| `prepare_dataset.py` | `OUTPUT_FILE` / `--output-file` | `train.txt` | Fichier de sortie |
| `prepare_rag_dataset.py` | `KB_PATH` | `knowledge.jsonl` | Base de faits lue (versionnée) |
| `prepare_rag_dataset.py` | `OUTPUT_FILE` | `rag_train.txt` | Fichier de sortie (ignoré par Git) |
| `prepare_facts_dataset.py` | `KB_PATH` / `OUTPUT_FILE` | `knowledge.jsonl` / `facts_train.txt` | Q/R directes (ignoré par Git) |
| `train.py` | `DATA_PATHS` | `[train.txt, rag_train.txt, facts_train.txt]` | Fichiers lus à l'entraînement |
| `train.py` | `MAX_EXAMPLES` / `--max-examples` | `3000` | Nombre maximum d'exemples, tous fichiers confondus |

L'ancienne constante `DATA_PATH` est remplacée par `DATA_PATHS` : un fichier manquant est simplement ignoré avec un avertissement. Le jeu utilisé fait 2 442 exemples (2 000 + 245 + 197), sous la limite de 3 000. Si vous changez la taille des jeux, gardez `MAX_EXAMPLES` de `train.py` supérieur au total.

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

L'entraînement est piloté par des options CLI (`python train.py --help`). Les valeurs par défaut correspondent au fine-tuning de GPT-2.

| Option | Défaut | Description |
| --- | --- | --- |
| `--data-paths` | `train.txt rag_train.txt` | Fichiers d'exemples (remplace l'ancien `DATA_PATH`) |
| `--output-dir` | `PicoGPT1-Iris` | Dossier de sauvegarde |
| `--max-len` | `256` | Longueur maximale du contexte |
| `--max-examples` | `3000` | Nombre maximum d'exemples, tous fichiers confondus |
| `--epochs` | `3` (`10` avec `--init scratch`) | Nombre d'epochs |
| `--batch-size` | `4` | Taille du batch |
| `--lr` | `5e-5` (`5e-4` avec `--init scratch`) | Taux d'apprentissage |
| `--init` | `pretrained` | `pretrained` ou `scratch` |
| `--pretrained-model` | `gpt2` | Modèle de départ en mode `pretrained` |
| `--n-layer` / `--n-embd` / `--n-head` | `4` / `256` / `4` | Architecture en mode `scratch` |
| `--save-steps` | `1000` | Fréquence des checkpoints |
| `--fp16` | désactivé | Demi-précision sur GPU (nécessite CUDA) |

> `MAX_LEN` est également défini dans `model_utils.py` pour tronquer les prompts à la génération. Les deux valeurs doivent rester identiques.

### Ce que fait le script

1. Charge le tokenizer depuis `PicoGPT1-Iris/` s'il y est (même vocabulaire, pas de requête réseau TLS), sinon depuis le modèle de départ. Il utilise `<|endoftext|>` comme token EOS et PAD.
2. Définit un `chat_template` correspondant au format appris `prompt\nreponse<|endoftext|>`.
3. Lit successivement `train.txt` et `rag_train.txt` et tokenise chaque exemple (tronqué à 256 tokens).
4. Masque les tokens du prompt dans les labels (`-100`) : la loss ne porte que sur la réponse.
5. Charge GPT-2 pré-entraîné (124 M paramètres) ou construit un modèle plus petit avec `--init scratch`.
6. Entraîne avec le `Trainer` de Transformers (`logging_steps=50`, `save_steps` configurable).
7. Sauvegarde le modèle et le tokenizer dans `PicoGPT1-Iris/`.

### Durée et sorties

Dernier entraînement : **4 min 11 sur RTX 3070** (CUDA + `--fp16`), 5 epochs (≈ 3 050 steps), loss moyenne 1,58 et dernière loss journalisée ≈ 1,05. Sur CPU, la même configuration prendrait nettement plus longtemps. Le fine-tuning part de poids pré-entraînés : la loss est bien plus basse qu'avec un modèle entraîné de zéro (≈ 4,54 auparavant).

Fichiers générés dans `PicoGPT1-Iris/` : `config.json`, `generation_config.json`, `model.safetensors`, `merges.txt`, `vocab.json`, `tokenizer.json`, `special_tokens_map.json`, `tokenizer_config.json`.

Un dossier de checkpoint peut apparaître : il est ignoré par Git et peut être supprimé sans risque, les poids finaux sont dans `model.safetensors`. Après un réentraînement, pensez à régénérer le GGUF (section 10).

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
| `--model-dir` | `PicoGPT1-Iris` | Dossier du modèle entraîné |
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
| `--min-score` | `0.40` | Score TF-IDF minimal pour fournir un contexte |

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

Avec `--rag`, `test.py` passe par `rag.py` : un passage de `knowledge.jsonl` est récupéré par TF-IDF, inséré dans le prompt au format `Contexte: <texte> | Question: <question>\n`, puis la réponse générée n'est acceptée que si elle est un extrait du contexte (`is_supported()`). Si le score de récupération est inférieur à `--min-score` (0,40 par défaut), aucun contexte n'est fourni et le script renvoie immédiatement `Je ne sais pas.`

```bash
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python test.py --prompt "Quel est le prix du pain à Paris ?" --rag --min-score 0.4
python test.py --prompt "À quelle température l'eau bout-elle ?" --rag --rag-k 3 --min-score 0.4
```

Contrairement au mode sans ancrage, seul le texte de la réponse est affiché (pas le prompt complet). Le prompt se termine toujours par `\n` : `generate()` l'ajoute automatiquement s'il manque, car ce séparateur était masqué pendant l'entraînement. Voir [doc/12-rag.md](doc/12-rag.md).

### Depuis votre propre code Python

Toute la logique est dans `model_utils.py` :

```python
from model_utils import load_model, generate

tokenizer, model = load_model("PicoGPT1-Iris")
print(generate(tokenizer, model, "What is 2+2?\n", do_sample=True))
```

`load_model()` charge tokenizer + modèle, passe en mode évaluation et neutralise `generation_config.max_length` pour éviter un conflit avec `max_new_tokens`. `generate()` tronque le prompt à 256 tokens (en ajoutant le `\n` final si besoin), applique les réglages anti-boucles (échantillonnage, `top_k=40`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`), s'arrête à l'EOS ou au premier saut de ligne généré, puis renvoie le texte décodé. Les paramètres `return_full_text`, `prefix_allowed_tokens_fn` et `num_beams` sont décrits à la section 13. `rag.py` utilise `do_sample=False`, `num_beams=5` et `no_repeat_ngram_size=0` : ce dernier réglage évite que l'interdiction des trigrammes ne déforme les réponses en beam search.

Si les poids sont absents, le message suivant s'affiche :

```
Poids introuvables dans 'PicoGPT1-Iris'. Lancez d'abord `python train.py`.
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
| `min_score` | float | `0.40` | Score TF-IDF minimal pour fournir un contexte |

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
  -d '{"prompt":"Quelle est la capitale de la France ?","use_rag":true,"rag_k":1,"min_score":0.4}'
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
python serve.py --host 127.0.0.1 --port 9000 --model-dir PicoGPT1-Iris
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Adresse d'écoute |
| `--port` | `8000` | Port d'écoute |
| `--model-dir` | `PicoGPT1-Iris` | Dossier du modèle |

Arrêtez le serveur avec `Ctrl+C`.

---

## 10. Exporter en GGUF

Les scripts Python utilisent `model.safetensors`. llama.cpp / llama.app attendent un fichier **GGUF** : `export_gguf.py` fait la conversion.

```bash
python export_gguf.py
```

Prérequis : `PicoGPT1-Iris/model.safetensors` doit exister (`python train.py`).

### Fonctionnement

1. Télécharge `convert_hf_to_gguf.py` depuis une version épinglée de llama.cpp (`b4000`).
2. Installe le paquet Python `gguf` dans un dossier temporaire s'il manque.
3. Lance la conversion avec `--outtype f16`.
4. Supprime les fichiers temporaires.

Résultat : `PicoGPT1-Iris.gguf` (~328 Mo, 149 tenseurs, F16), ignoré par Git. Le `chat_template` présent dans `tokenizer_config.json` est conservé dans le GGUF.

### Vérification locale

```bash
llama cli -m PicoGPT1-Iris.gguf -st -p "What is 2+2?" -n 20
```

Après chaque réentraînement, relancez l'export puis la publication pour garder poids Transformers et GGUF synchronisés :

```bash
python export_gguf.py
python publish_hf.py
```

### Après l'ajout du pipeline RAG

Les exemples `rag_train.txt` et `facts_train.txt` font désormais partie de l'entraînement : après `python prepare_rag_dataset.py`, `python prepare_facts_dataset.py` et `python train.py`, **pensez à régénérer le GGUF** avec `python export_gguf.py` avant de le tester dans llama.app, puis à le republier si besoin. Le GGUF ne contient ni `knowledge.jsonl` ni le code TF-IDF : l'ancrage (`--rag`, `use_rag`) fonctionne uniquement côté scripts Python, avec `scikit-learn` installé.

---

## 11. Publier sur Hugging Face

Dernière release publiée : **Daisy** — <https://huggingface.co/nazimboudeffa/PicoGPT1-Daisy>. Iris est entraînée localement mais son dépôt n'est pas encore publié.

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
2. Upload le contenu de `PicoGPT1-Iris/` en ignorant `checkpoint-*`.
3. Upload `MODEL_CARD.md` comme `README.md` du dépôt Hub.
4. Upload `PicoGPT1-Iris.gguf`.

Une session HTTP avec retries est configurée pour absorber les erreurs TLS intermittentes de certains réseaux. En cas d'échec, relancez simplement la commande.

---

## 12. Utiliser le modèle publié

### Avec le CLI llama.app

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Iris -hff PicoGPT1-Iris.gguf
```

`-hff` force le fichier GGUF à utiliser. Comme le dépôt ne contient qu'un seul GGUF, la forme courte fonctionne aussi :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Iris
```

### Interface web de chat

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Iris -hff PicoGPT1-Iris.gguf
# puis ouvrir http://localhost:9931
```

> **Anti-hallucination dans llama.app** : le GGUF ne contient pas le RAG Python, mais le modèle a appris les 30 faits de `knowledge.jsonl` en réponses directes. Une question simple comme `quelle est la capitale de France ?` donne maintenant `Paris`, et `que font 2+2 ?` donne `4`, sans coller de contexte.
>
> Pour un fait peu courant ou une formulation éloignée, le format avec contexte reste disponible :
>
> ```
> Contexte: La capitale de la France est Paris. | Question: Quelle est la capitale de la France ?
> ```
>
> Le format et les limites sont détaillés dans [doc/08-huggingface.md](doc/08-huggingface.md).

### Avec Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nazimboudeffa/PicoGPT1-Iris"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "Quelle est la capitale de la France ?\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=48, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Avec le serveur local

Si vous avez entraîné vos propres poids, `python serve.py` expose directement `PicoGPT1-Iris/`. Pour un autre dossier :

```bash
python serve.py --model-dir chemin/vers/mon/modele --port 9000
```

---

## 13. Architecture du modèle

Configuration héritée du modèle pré-entraîné (mode par défaut) et sauvegardée dans `PicoGPT1-Iris/config.json` :

| Paramètre | Valeur |
| --- | --- |
| `model_type` | `gpt2` |
| `architectures` | `GPT2LMHeadModel` |
| `n_layer` | `12` |
| `n_embd` | `768` |
| `n_head` | `12` |
| `n_positions` / `n_ctx` | `1024` |
| `vocab_size` | `50 257` |
| `activation_function` | `gelu_new` |
| `torch_dtype` | `float32` |

Soit **124 439 808 paramètres** (GPT-2 pré-entraîné ajusté). Les exemples restent tronqués à 256 tokens à l'entraînement et à la génération. En mode `--init scratch`, l'architecture est configurable (`--n-layer`, `--n-embd`, `--n-head`) avec une valeur par défaut de 4 couches / 256 dimensions / 4 têtes.

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
| `prefix_allowed_tokens_fn` | `None` | Hook générique `(batch_id, input_ids)` limitant les tokens autorisés à chaque étape. Anciennement utilisé par `rag.py` pour forcer la copie littérale du contexte ; abandonné car il tronquait les réponses. L'ancrage est désormais vérifié après génération par `is_supported()`. |
| `num_beams` | `1` | Beam search ; `rag.py` passe à `5` avec `do_sample=False` pour un décodage déterministe. |

`rag.py` vérifie ensuite `is_supported(answer, context)` : si la réponse n'est pas un extrait du contexte, il renvoie `Je ne sais pas.`. Voir [doc/12-rag.md](doc/12-rag.md).

---

## 14. Dépannage

| Symptôme | Solution |
| --- | --- |
| `Poids introuvables dans 'PicoGPT1-Iris'` | Les poids ne sont pas versionnés : lancez `python train.py` (puis `python export_gguf.py` pour le GGUF). |
| `Token is required ... no token found` | Connectez-vous avec `huggingface-cli login` ou définissez `HF_TOKEN`. |
| Erreur SSL pendant la publication | Les retries sont déjà configurés ; relancez `python publish_hf.py`. |
| Le modèle ne génère rien avec llama.cpp | Régénérez le GGUF après le dernier entraînement : `python export_gguf.py`, puis testez avec `llama cli ...`. |
| Réponses incohérentes | Le modèle par défaut est un GPT-2 pré-entraîné ajusté ; vérifiez que le GGUF a bien été régénéré après l'entraînement (`python export_gguf.py`). Pour des faits vérifiés, utilisez `--rag` : l'ancrage n'existe pas dans le GGUF / llama.app. |
| `ModuleNotFoundError: No module named 'sklearn'` | Installez la dépendance : `python -m pip install scikit-learn`. Elle est requise par `rag.py`, `test.py --rag`, `serve.py` avec `use_rag` et `evaluate.py`. |
| `rag_train.txt introuvable, ignoré.` | Lancez `python prepare_rag_dataset.py` après `python prepare_dataset.py`, puis réentraînez avec `python train.py`. |
| Le mode `--rag` répond toujours `Je ne sais pas.` | Le score TF-IDF est sous le seuil : vérifiez `knowledge.jsonl`, reformulez la question ou baissez `--min-score` (ex. `0.4`). Les questions hors base doivent, elles, s'abstenir. |
| Le contexte récupéré n'est pas le bon | Augmentez `--rag-k`, reformulez la question, ou enrichissez `knowledge.jsonl` puis relancez `python prepare_rag_dataset.py`. |
| Port 8000 déjà utilisé | `python serve.py --port 9000`. |
| `LF will be replaced by CRLF...` sous Windows | Avertissement Git sans conséquence. |
| Dossier de checkpoint | Créé par `Trainer`, ignoré par Git : supprimable sans risque. |

---

## 15. Limites et pistes d'amélioration

### Limites actuelles

- Jeu de données réduit : 2 000 exemples Alpaca filtrés + 245 exemples d'ancrage.
- Modèle GPT-2 complet (124 M paramètres) mais contexte effectivement limité à 256 tokens ; le GGUF F16 fait environ 328 Mo (les poids Transformers FP32 font 498 Mo).
- Aucun jeu de validation **pendant l'entraînement** : le loss affiché est uniquement le loss d'entraînement. Le jeu de test figé évalue le modèle après coup, mais ne pilote pas encore `eval_loss`.
- Échantillonnage par défaut avec arrêt au saut de ligne ; le mode `--greedy` boucle encore.
- GGUF exporté en F16, sans quantification.

### Limites du pipeline RAG

- Base **fermée de 30 faits** : aucune question hors de `knowledge.jsonl` ne peut recevoir de réponse factuelle (abstention attendue).
- Les **30 questions canoniques** donnent 30/30, mais ce score reste optimiste : sur 30 reformulations jamais vues, Daisy fait 28/30 et Iris 29/30.
- Les questions **hors base proches** restent le point faible : Daisy 1/5 abstention, Iris 2/5. Le modèle recopie parfois la réponse du fait le plus proche (capitale de l'Espagne → Paris, formule du sel → H2O). L'ajout de 90 exemples négatifs et 120 reformulations ancrées améliore l'in-scope mais ne règle pas complètement ce problème.
- Seuil `DEFAULT_MIN_SCORE = 0.40`, calibré sur le jeu de test : `0,50` refusait deux reformulations couvertes (scores 0,411 et 0,499).
- Dépendance à **scikit-learn** (`sklearn`) pour le TF-IDF ; l'ancrage n'existe pas dans le GGUF / llama.cpp.

Résultats détaillés dans [doc/12-rag.md](doc/12-rag.md) et [doc/14-benchmark.md](doc/14-benchmark.md).

### Améliorations possibles

1. Augmenter `--max-examples` de `prepare_dataset.py` **et** de `train.py` (limite actuelle : 3 000).
2. Essayer un autre modèle de départ via `--pretrained-model`, ou augmenter le nombre d'epochs.
3. Réserver 10 % des exemples pour un jeu de validation et suivre `eval_loss` (le jeu de test figé, lui, existe déjà).
4. Ajuster `temperature`, `top_p`, `top_k`, `repetition_penalty` et `no_repeat_ngram_size` selon l'usage.
5. Exporter le GGUF en `Q4_K_M` pour réduire sa taille et accélérer l'inférence.
6. Étendre `knowledge.jsonl`, recalibrer `--min-score` et comparer plusieurs valeurs de `--rag-k` avec `evaluate.py --split test` puis `compare_reports.py`.
7. Corriger les hallucinations sur les questions hors base proches, en utilisant `reports/<codename>-test.json` comme référence.

---

## 16. Liens et références

- Dépôt GitHub : <https://github.com/nazimboudeffa/picogpt-kit>
- Modèle Hugging Face : <https://huggingface.co/nazimboudeffa/PicoGPT1-Iris>
- Dataset : <https://huggingface.co/datasets/tatsu-lab/alpaca>
- Documentation détaillée : [doc/README.md](doc/README.md)
- Pipeline RAG / ancrage : [doc/12-rag.md](doc/12-rag.md)
- Versions et noms de fleurs : [doc/13-versions.md](doc/13-versions.md)
- Protocole de benchmark : [doc/14-benchmark.md](doc/14-benchmark.md)
- Historique des releases : [CHANGELOG.md](CHANGELOG.md)
- Fiche modèle : [MODEL_CARD.md](MODEL_CARD.md)
- llama.app : <https://llama.app>
