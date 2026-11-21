# Versions et noms de fleurs

Le projet porte un **codename végétal** qui change à chaque itération : `Jasmine` (prototype) puis `Daisy` (version actuelle). Cela permet de désigner rapidement une release, de garder les anciennes versions comparables et d'éviter de mélanger les poids d'une itération avec ceux de la suivante.

## Convention

| Release | Codename | Portée |
| --- | --- | --- |
| `0.x` | Jasmine | prototype historique, conservé tel quel |
| `1.0.0` | Daisy | GPT-2 pré-entraîné, RAG, GGUF, publication HF |
| `1.1.0` | Iris | correctif des hallucinations hors base proches, mêmes poids de base |
| `2.0.0` | (à choisir) | changement d'architecture (ex. LoRA) |

Le **numéro de version** décrit l'évolution technique (semver simplifié) ; le **codename** est le nom lisible de la release. Le nom complet est `PicoGPT<majeure>-<Fleur>`. Le tag git associé est `v<version>-<fleur en minuscules>`, par exemple `v1.0.0-daisy`.

Règle recommandée : un nouveau codename à chaque publication, un nouveau dépôt Hugging Face par codename (comme `PicoGPT1-Jasmine` et `PicoGPT1-Daisy`), et un tag git sur le commit de release. Ainsi, une régression peut toujours être identifiée et reproductible.

## Source de vérité

Le fichier [`project.json`](../project.json) contient toutes les données de release :

```json
{
  "family": "PicoGPT1",
  "version": "1.0.0",
  "codename": "Daisy",
  "hf_owner": "nazimboudeffa"
}
```

Le module [`config.py`](../config.py) lit ce fichier et expose les constantes utilisées par les scripts :

| Constante | Valeur pour la release actuelle | Utilisée par |
| --- | --- | --- |
| `MODEL_NAME` | `PicoGPT1-Daisy` | `train.py`, `test.py`, `publish_hf.py` |
| `MODEL_DIR` | `PicoGPT1-Daisy` | `model_utils.py`, `train.py`, `export_gguf.py`, `publish_hf.py` |
| `GGUF_FILE` | `PicoGPT1-Daisy.gguf` | `export_gguf.py`, `publish_hf.py` |
| `HF_REPO_ID` | `nazimboudeffa/PicoGPT1-Daisy` | `publish_hf.py` |
| `RELEASE_TAG` | `v1.0.0-daisy` | suggestion du script de renommage |

Vérification rapide :

```bash
python config.py
# PicoGPT1-Daisy — v1.0.0 (v1.0.0-daisy) — nazimboudeffa/PicoGPT1-Daisy
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
4. renomme le dossier de poids `PicoGPT1-Daisy/` et le fichier `PicoGPT1-Daisy.gguf` s'ils existent ;
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
