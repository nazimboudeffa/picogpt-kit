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
| `--model-dir` | `PicoGPT1-Iris` | Dossier du modèle |
| `--prompt` | `What is the capital of France?\n` | Texte à compléter |
| `--max-new-tokens` | `64` | Nombre maximum de tokens générés |
| `--do-sample` | activé | Échantillonnage (défaut, évite les boucles) |
| `--greedy` | désactivé | Décodage glouton (déconseillé) |
| `--temperature` | `0.8` | Température si échantillonnage |
| `--top-p` | `0.9` | Nucleus sampling si échantillonnage |
| `--top-k` | `40` | Limite aux 40 tokens les plus probables |
| `--repetition-penalty` | `1.15` | Pénalité de répétition |
| `--no-repeat-ngram-size` | `3` | Interdit la répétition de trigrammes |
| `--rag` | désactivé | Ancre la réponse dans `knowledge.jsonl` |
| `--rag-k` | `1` | Nombre de passages récupérés à considérer |
| `--min-score` | `0.50` | Score TF-IDF minimal pour fournir un contexte |

L'échantillonnage est activé par défaut car le décodage glouton fait boucler le modèle. La génération s'arrête à l'EOS ou au premier saut de ligne produit dans la réponse.

## Génération ancrée avec `--rag`

```bash
python test.py --prompt "Quelle est la capitale de la France ?" --rag
```

Sans `--rag`, le script affiche le texte complet (prompt + réponse). Avec `--rag`, il passe par `rag.py` et n'affiche que la **réponse courte** : un extrait du contexte récupéré, ou `Je ne sais pas.`

Le déroulement est le suivant :

1. `knowledge.jsonl` est indexé par TF-IDF (`char_wb`, ngrammes de 3 à 5 caractères) sur `text + question`.
2. Si le meilleur score est inférieur à `--min-score` (0,50 par défaut), aucun contexte n'est ajouté : le script renvoie immédiatement `Je ne sais pas.`
3. Sinon, le prompt prend la forme `Contexte: <texte> | Question: <question>\n`. La génération est déterministe (`num_beams=5`, `do_sample=False`, `no_repeat_ngram_size=0`) ; la réponse est ensuite acceptée seulement si `is_supported()` la trouve dans le contexte, sinon `Je ne sais pas.` est renvoyé.
4. `is_supported()` vérifie enfin que la réponse est bien un extrait du contexte ; sinon, abstention.

`generate()` ajoute automatiquement le `\n` final au prompt s'il manque : ce séparateur était masqué dans les labels pendant l'entraînement.

```bash
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python test.py --prompt "Quel est le prix du pain à Paris ?" --rag --min-score 0.5
python test.py --prompt "À quelle température l'eau bout-elle ?" --rag --rag-k 3 --min-score 0.4
```

Le réglage du seuil et les métriques sont détaillés dans [12-rag.md](12-rag.md). `scikit-learn` doit être installé (`python -m pip install scikit-learn`).

## Fonctions partagées

`model_utils.py` contient deux fonctions utilisées par `test.py` et `serve.py` :

- `load_model(model_dir)` : charge tokenizer + modèle, passe en mode évaluation et neutralise `generation_config.max_length`.
- `generate(tokenizer, model, prompt, ...)` : ajoute le `\n` final si besoin, tokenise le prompt (tronqué à 256 tokens), applique les réglages anti-boucles, s'arrête à l'EOS ou au premier saut de ligne généré, puis renvoie le texte décodé. `rag.py` utilise `return_full_text=False`, `num_beams=5` et désactive `no_repeat_ngram_size` pour la beam search.

Erreur attendue si les poids sont absents :

```
Poids introuvables dans 'PicoGPT1-Iris'. Lancez d'abord `python train.py`.
```

## Conseil sur le prompt

Le modèle a été entraîné avec un saut de ligne entre le prompt et la réponse. Ajoutez donc `\n` à la fin de votre prompt pour rester proche du format d'entraînement.

Exemple PowerShell :

```powershell
python test.py --prompt "What is 2+2?`n" --do-sample
```
