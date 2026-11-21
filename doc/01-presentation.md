# 1. Présentation

## Objectif

`PicoGPT1-Iris` est un mini modèle GPT-2 entraîné à partir de zéro sur un petit jeu d'instructions/réponses. Le projet privilégie des **scripts Python directs** : aucun notebook n'est nécessaire pour entraîner ou tester le modèle.

Les scripts locaux n'utilisent **ni llama.cpp, ni Ollama**. Une variante GGUF est toutefois exportée pour permettre de tester avec [llama.app](https://llama.app).

## État actuel

| Élément | État |
| --- | --- |
| Entraînement | Fine-tuning GPT-2 sur 2 442 exemples, 5 epochs |
| Poids locaux | `PicoGPT1-Iris/model.safetensors` (non versionné) |
| Publication | https://huggingface.co/nazimboudeffa/PicoGPT1-Iris |
| Compatibilité llama.app | Confirmée via `llama cli -hf ...` |
| Branche Git locale | `main` |

> Le modèle est expérimental : malgré le passage à GPT-2 pré-entraîné, les réponses générées restent approximatives. C'est un projet d'apprentissage, pas un assistant généraliste.

## Arborescence

```
.
├── PicoGPT1-Iris/        # modèle entraîné (configs + tokenizer + poids)
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors    # généré, ignoré par Git
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── doc/                     # cette documentation
├── export_gguf.py           # conversion Transformers -> GGUF
├── model_utils.py           # chargement et génération partagés
├── prepare_dataset.py       # téléchargement Alpaca -> train.txt
├── publish_hf.py            # publication du modèle sur Hugging Face
├── serve.py                 # serveur HTTP local
├── test.py                  # génération en ligne de commande
├── train.py                 # entraînement
└── README.md                # vue d'ensemble
```

## Versions utilisées lors du développement

- Python 3.13 sur Windows 11
- `torch` (CPU)
- `transformers` 4.52.4
- `datasets`
- `huggingface_hub` 0.33.0
- CLI llama.cpp / llama.app build b12375

