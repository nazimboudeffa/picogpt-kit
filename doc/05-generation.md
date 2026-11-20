# 5. Génération avec `test.py`

## Commande simple

```bash
python test.py
```

Par défaut, le prompt est :

```
What is the capital of France?
```

Le script affiche le texte complet : prompt + réponse générée.

## Options

```bash
python test.py --prompt "Explique la psychanalyse.\n" --do-sample
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--model-dir` | `PicoGPT1-Daisy` | Dossier du modèle |
| `--prompt` | `What is the capital of France?\n` | Texte à compléter |
| `--max-new-tokens` | `64` | Nombre maximum de tokens générés |
| `--do-sample` | activé | Échantillonnage (défaut, évite les boucles) |
| `--greedy` | désactivé | Décodage glouton (déconseillé) |
| `--temperature` | `0.8` | Température si échantillonnage |
| `--top-p` | `0.9` | Nucleus sampling si échantillonnage |
| `--top-k` | `40` | Limite aux 40 tokens les plus probables |
| `--repetition-penalty` | `1.15` | Pénalité de répétition |
| `--no-repeat-ngram-size` | `3` | Interdit la répétition de trigrammes |

L'échantillonnage est activé par défaut car le décodage glouton fait boucler le modèle. La génération s'arrête à l'EOS ou au premier saut de ligne produit dans la réponse.

## Fonctions partagées

`model_utils.py` contient deux fonctions utilisées par `test.py` et `serve.py` :

- `load_model(model_dir)` : charge tokenizer + modèle, passe en mode évaluation et neutralise `generation_config.max_length`.
- `generate(tokenizer, model, prompt, ...)` : tokenise le prompt (tronqué à 256 tokens), applique les réglages anti-boucles, s'arrête à l'EOS ou au premier saut de ligne généré, puis renvoie le texte décodé.

Erreur attendue si les poids sont absents :

```
Poids introuvables dans 'PicoGPT1-Daisy'. Lancez d'abord `python train.py`.
```

## Conseil sur le prompt

Le modèle a été entraîné avec un saut de ligne entre le prompt et la réponse. Ajoutez donc `\n` à la fin de votre prompt pour rester proche du format d'entraînement.

Exemple PowerShell :

```powershell
python test.py --prompt "What is 2+2?`n" --do-sample
```
