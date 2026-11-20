# 6. Serveur local avec `serve.py`

`serve.py` expose le modèle sur HTTP avec la bibliothèque standard de Python. Aucun framework web ni Ollama n'est nécessaire.

## Démarrage

```bash
python serve.py
```

Sortie attendue :

```
Serveur prêt : http://127.0.0.1:8000/generate
```

Le modèle reste chargé en mémoire pendant toute la durée du serveur.

## Route

`POST /generate`

Corps JSON :

| Champ | Type | Défaut | Description |
| --- | --- | --- | --- |
| `prompt` | string | obligatoire | Texte à compléter |
| `max_new_tokens` | int | `64` | Nombre maximum de tokens |
| `do_sample` | bool | `true` | Échantillonnage (déconseillé de mettre `false`) |
| `temperature` | float | `0.8` | Température |
| `top_p` | float | `0.9` | Nucleus sampling |
| `top_k` | int | `40` | Limite de tokens candidats |
| `repetition_penalty` | float | `1.15` | Pénalité de répétition |
| `no_repeat_ngram_size` | int | `3` | Anti-répétition de trigrammes |
| `use_rag` | bool | `false` | Active l'ancrage dans `knowledge.jsonl` |
| `rag_k` | int | `1` | Nombre de passages récupérés à considérer |
| `min_score` | float | `0.50` | Score TF-IDF minimal pour fournir un contexte |

Réponse succès (`200`) :

```json
{ "generated_text": "prompt + réponse" }
```

Avec `use_rag: true`, `generated_text` ne contient que la réponse courte (extrait ancré ou `Je ne sais pas.`), et non le prompt complet. Le seuil et le nombre de passages sont détaillés dans [12-rag.md](12-rag.md).

Erreur (`400`) :

```json
{ "error": "message d'erreur" }
```

Toute autre route renvoie `404`.

## Exemples

PowerShell :

```powershell
$body = @{
    prompt = "What is psychoanalysis?`n"
    max_new_tokens = 64
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/generate -Method Post -Body $body -ContentType "application/json"
```

curl :

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is psychoanalysis?\n","do_sample":true}'
```

Corps JSON avec ancrage RAG :

```json
{
  "prompt": "Quelle est la capitale de la France ?",
  "use_rag": true,
  "rag_k": 1,
  "min_score": 0.5
}
```

Appel curl correspondant :

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Quelle est la capitale de la France ?","use_rag":true,"rag_k":1,"min_score":0.5}'
```

`scikit-learn` doit être installé pour utiliser `use_rag`.

## Options CLI

```bash
python serve.py --host 127.0.0.1 --port 9000 --model-dir PicoGPT1-Daisy
```

| Option | Défaut | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Adresse d'écoute |
| `--port` | `8000` | Port d'écoute |
| `--model-dir` | `PicoGPT1-Daisy` | Dossier du modèle |

Arrêter le serveur avec `Ctrl+C`.
