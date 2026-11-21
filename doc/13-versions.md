# Versions et noms de fleurs

Le projet porte un **codename végétal** qui change à chaque itération : `Jasmine` (prototype), `Daisy` (1.0.0) puis `Iris` (version actuelle). Cela permet de désigner rapidement une release, de garder les anciennes versions comparables et d'éviter de mélanger les poids d'une itération avec ceux de la suivante.

## Convention

| Release | Codename | Portée |
| --- | --- | --- |
| `0.x` | Jasmine | prototype historique, conservé tel quel |
| `1.0.0` | Daisy | GPT-2 pré-entraîné, RAG, GGUF, publication HF |
| `1.1.0` | Iris | correctif des hallucinations hors base proches, mêmes poids de base |
| `1.2.0` | Poppy *(proposé)* | fiabilité de l'ancrage, validation, benchmark intégré, GGUF quantifié |
| `1.3.0` | Lavender *(proposé)* | dialogue multi-tours, historique, contexte étendu |
| `2.0.0` | Orchid *(proposé, officiellement à choisir)* | changement de base et d'architecture (ex. LoRA), raisonnement |
| `2.1.0` | Sunflower *(proposé)* | raisonnement ancré multi-sauts |

Le **numéro de version** décrit l'évolution technique (semver simplifié) ; le **codename** est le nom lisible de la release. Le nom complet est `PicoGPT<majeure>-<Fleur>`. Le tag git associé est `v<version>-<fleur en minuscules>`, par exemple `v1.1.0-iris`.

Règle recommandée : un nouveau codename à chaque publication, un nouveau dépôt Hugging Face par codename (comme `PicoGPT1-Jasmine` et `PicoGPT1-Iris`), et un tag git sur le commit de release. Ainsi, une régression peut toujours être identifiée et reproductible.

## Feuille de route proposée (non encore approuvée)

Les lignes `1.2.0` à `2.1.0` du tableau ci-dessus sont une **proposition** de trajectoire pour passer d'un modèle qui restitue des faits ancrés à un modèle capable de discuter puis de raisonner. Elles ne sont pas encore dans `project.json` : seule la release courante (`1.1.0` / Iris) est active. Les codenames proposés suivent la convention végétale existante ; celui de `2.0.0` restait « à choisir » dans la version précédente de ce document.

| Version | Codename proposé | Famille / tag | Ce que la version apporte |
| --- | --- | --- | --- |
| `1.2.0` | Poppy | `PicoGPT1-Poppy`, `v1.2.0-poppy` | Fiabilité de l'ancrage : contrôle entité question ↔ passage récupéré **en amont** de la génération, jeu de validation et `eval_loss`, seuil recalibré, 3 000 exemples, GGUF quantifié |
| `1.3.0` | Lavender | `PicoGPT1-Lavender`, `v1.3.0-lavender` | Dialogue multi-tours : historique `messages[]` dans `serve.py`, données de conversation, `chat_template` multi-tours, contexte 512/1024 |
| `2.0.0` | Orchid | `PicoGPT2-Orchid`, `v2.0.0-orchid` | Changement de base et d'architecture : checkpoint instruct plus capable, LoRA/QLoRA, données de raisonnement en quelques étapes |
| `2.1.0` | Sunflower | `PicoGPT2-Sunflower`, `v2.1.0-sunflower` | Raisonnement ancré : RAG multi-sauts, vérification de chaque étape, abstention si une étape n'est pas supportée |

Le passage à la famille `PicoGPT2` en `2.0.0` découle de la convention `PicoGPT<majeure>-<Fleur>` : utiliser `python rename_codename.py --codename Orchid --version 2.0.0 --family PicoGPT2`. Jusqu'ici, `--family PicoGPT1` reste la valeur par défaut.

> Deux réserves : LoRA est une méthode de réglage efficace en paramètres, pas une source de capacité de raisonnement — le gain viendra surtout du modèle de base et des données. Et avec 124 M paramètres, 256 tokens et 2 652 exemples, Iris peut viser un dialogue court et fiable (`1.3.0`), pas un raisonnement général (`2.0.0`).

### Critères de réussite

| Version | Critères (à valider sur le jeu **test**, jamais sur le jeu train) |
| --- | --- |
| `1.2.0` Poppy | `40/40` sur `eval/test.jsonl` (contre 36/40 pour Iris), dont **5/5** hors base proches ; 30/30 exactes, 30/30 ancrées et 30/30 retrieval conservés ; 0 fuite (`eval/check_leakage.py`) ; `eval_loss` et rapport `reports/poppy-test.json` présents ; GGUF Q4_K_M ≤ 100 Mo |
| `1.3.0` Lavender | Nouveau jeu figé `eval/test-v2.jsonl` : ≥ 20 dialogues de 2 à 3 tours, ≥ 90 % de réponses correctes et ancrées ; maintien de 40/40 sur `test.jsonl` ; contexte ≥ 512 tokens ; reprise pronominale et rappel d'un fait après un tour positifs |
| `2.0.0` Orchid | Jeu figé `eval/reasoning.jsonl` (arithmétique simple + inférences à 2 sauts) : ≥ 80 % de réponses correctes avec étapes vérifiables ; ≥ 40/40 sur le benchmark historique et ≥ 90 % sur le dialogue ; GGUF quantifié publié |
| `2.1.0` Sunflower | ≥ 90 % sur `reasoning.jsonl` ; 0 hallucination sur un jeu élargi de questions hors base proches ; chaque étape de la réponse est supportée par le contexte récupéré |

Ces seuils sont des **cibles de travail**, à figer avant l'entraînement de chaque release. Si un jeu doit évoluer, créer un nouveau fichier versionné (`test-v2.jsonl`, `reasoning-v2.jsonl`) plutôt que de modifier l'ancien : c'est la règle énoncée dans [14-benchmark.md](14-benchmark.md).

### Procédure à chaque jalon

Reprendre le protocole de [14-benchmark.md](14-benchmark.md) §93, puis :

```bash
python eval/check_leakage.py
python evaluate.py --split test        # --split test-v2 à partir de Lavender
python compare_reports.py
python rename_codename.py --codename Poppy --version 1.2.0 --dry-run
python rename_codename.py --codename Poppy --version 1.2.0
# ... entraînement, mesure, export, publication ...
git tag -a v1.2.0-poppy -m "Release PicoGPT1-Poppy"
git push origin v1.2.0-poppy
```

## Source de vérité

Le fichier [`project.json`](../project.json) contient toutes les données de release :

```json
{
  "family": "PicoGPT1",
  "version": "1.1.0",
  "codename": "Iris",
  "hf_owner": "nazimboudeffa"
}
```

Le module [`config.py`](../config.py) lit ce fichier et expose les constantes utilisées par les scripts :

| Constante | Valeur pour la release actuelle | Utilisée par |
| --- | --- | --- |
| `MODEL_NAME` | `PicoGPT1-Iris` | `train.py`, `test.py`, `publish_hf.py` |
| `MODEL_DIR` | `PicoGPT1-Iris` | `model_utils.py`, `train.py`, `export_gguf.py`, `publish_hf.py` |
| `GGUF_FILE` | `PicoGPT1-Iris.gguf` | `export_gguf.py`, `publish_hf.py` |
| `HF_REPO_ID` | `nazimboudeffa/PicoGPT1-Iris` | `publish_hf.py` |
| `RELEASE_TAG` | `v1.1.0-iris` | suggestion du script de renommage |

Vérification rapide :

```bash
python config.py
# PicoGPT1-Iris — v1.1.0 (v1.1.0-iris) — nazimboudeffa/PicoGPT1-Iris
```

Aucun script Python ne code plus le nom de la fleur en dur : changer `project.json` suffit à rediriger entraînement, génération, export et publication.

## Changer de fleur

Le script [`rename_codename.py`](../rename_codename.py) effectue le changement complet :

```bash
# 1. Aperçu : aucun fichier n'est écrit
python rename_codename.py --codename Iris --version 1.1.0 --dry-run

# 2. Application
python rename_codename.py --codename Iris --version 1.1.0
```

Il :

1. met à jour `project.json` ;
2. remplace l'ancien nom dans les fichiers texte (README, MODEL_CARD, `doc/`) ;
3. insère une entrée datée dans [`CHANGELOG.md`](../CHANGELOG.md) ;
4. renomme le dossier de poids `PicoGPT1-Iris/` et le fichier `PicoGPT1-Iris.gguf` s'ils existent ;
5. affiche les commandes de vérification, de publication et le tag git suggéré.

Options utiles :

| Option | Rôle |
| --- | --- |
| `--codename Iris` | nouveau nom de fleur (obligatoire) |
| `--version 1.1.0` | nouvelle version (défaut : la version courante) |
| `--family PicoGPT2` | change la famille du modèle |
| `--hf-owner mon-compte` | change le propriétaire du dépôt Hugging Face |
| `--no-data` | conserve l'ancien dossier de poids et l'ancien GGUF |
| `--dry-run` | affiche les changements sans rien modifier |

Le remplacement texte ignore volontairement `tokenizer.json` et `vocab.json` : le token `ĠDaisy` y est présent, mais il appartient au vocabulaire de GPT-2 et n'a aucun lien avec le nom de la release. Il ne faut pas le modifier.

## Checklist après renommage

```bash
python config.py
python -m py_compile config.py rename_codename.py train.py test.py serve.py evaluate.py export_gguf.py publish_hf.py
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python export_gguf.py
python publish_hf.py
git tag -a v1.1.0-iris -m "Release PicoGPT1-Iris"
git push origin v1.1.0-iris
```

À vérifier manuellement :

- le tableau comparatif du `MODEL_CARD.md` (la fleur considérée comme « ancienne » peut changer) ;
- le nouveau dépôt Hugging Face, créé par `publish_hf.py` à partir de `HF_REPO_ID` ;
- la fiche modèle uploadée comme `README.md` du dépôt Hub ;
- le GGUF : il doit être régénéré depuis les poids de la release courante, puis republié.

Si les poids n'ont pas été renommés (`--no-data`), `config.py` pointe vers un dossier vide : relancez `python train.py` ou déplacez manuellement le dossier de poids.
