---
library_name: transformers
version: 1.0.0
codename: daisy
tags:
  - gpt2
  - gguf
  - llama.cpp
  - llama.app
  - alpaca
  - instruction
  - daisy
  - rag
  - abstention
---

# PicoGPT1-Daisy

> Release `1.0.0` — codename Daisy. Les versions et les noms de fleurs sont décrits dans le dépôt source (`CHANGELOG.md`).

GPT-2 pré-entraîné ajusté sur 2 000 instructions/réponses Alpaca filtrées, 245 exemples d'ancrage et 197 questions/réponses directes issus d'une base fermée de 30 faits (2 442 exemples au total).

- Architecture : GPT-2 complet, 12 couches, 768 dimensions, 12 têtes d'attention (124 M paramètres)
- Contexte : 1 024 positions dans le modèle, exemples tronqués à 256 tokens
- Tokenizer : GPT-2 (50 257 tokens)
- Format appris : `prompt\nreponse<|endoftext|>`
- Fichiers : poids Transformers (`model.safetensors`) et variante `PicoGPT1-Daisy.gguf`

> Modèle expérimental : les réponses en génération libre peuvent être approximatives. Pour des faits vérifiés, utilisez l'ancrage décrit ci-dessous (scripts Python uniquement).

## Différences avec `PicoGPT1-Jasmine`

L'ancien dépôt [`nazimboudeffa/PicoGPT1-Jasmine`](https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine) est conservé tel quel comme référence historique. Celui-ci est une **nouvelle version**, avec un modèle plus grand issu d'un fine-tuning de GPT-2.

| Élément | `PicoGPT1-Jasmine` (ancien) | `PicoGPT1-Daisy` (ce dépôt) |
| --- | --- | --- |
| Architecture | GPT-2 : 2 couches, 128 dims, 2 têtes, contexte 256 | **GPT-2 complet** : 12 couches, 768 dims, 12 têtes (124 M paramètres) |
| Jeu d'entraînement | 500 exemples Alpaca bruts | 2 000 Alpaca **filtrés** + 245 ancrages + 197 Q/R directes = **2 442 exemples** |
| Données | Quelques erreurs factuelles et doublons | Erreurs connues retirées (eau à 115 °C, fausse définition du « mole », `<nooutput>`), doublons supprimés, format uniformisé |
| Décodage | Glouton par défaut, boucles fréquentes | Échantillonnage par défaut (`top_k=40`, `no_repeat_ngram_size=3`, `repetition_penalty=1.15`), arrêt à l'EOS ou au premier `\n`, séparateur `\n` garanti |
| Ancrage / abstention | Aucun | `knowledge.jsonl` (30 faits), TF-IDF, vérification d'ancrage, `Je ne sais pas.` hors base |
| Entraînement | 10 epochs depuis zéro, ~1 250 steps, loss finale ≈ 4,95 | 5 epochs de fine-tuning (2 442 exemples) |
| GGUF | Ancien export | Régénéré depuis les nouveaux poids |

En résumé : **modèle plus grand issu de GPT-2 pré-entraîné, données nettoyées, décodage corrigé et ancrage anti-hallucination** côté Python. Le passage à GPT-2 pré-entraîné améliore la cohérence des réponses en génération libre, y compris dans le GGUF. Le code source et la documentation détaillée sont dans <https://github.com/nazimboudeffa/picogpt-kit>.

## Utiliser avec llama.app / llama.cpp

Le CLI fourni par [llama.app](https://llama.app) télécharge le GGUF depuis ce dépôt :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Daisy -hff PicoGPT1-Daisy.gguf
```

Pour ouvrir l'interface web de chat :

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Daisy -hff PicoGPT1-Daisy.gguf
# puis http://localhost:9931
```

## Utiliser avec Transformers

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

## Ancrage (RAG)

Le dépôt fournit un pipeline d'ancrage côté Python : `rag.py` récupère un fait dans `knowledge.jsonl` avec un TF-IDF (`char_wb` 3-5) sur `text + question`, construit le prompt `Contexte: <texte> | Question: <question>\n`, génère en beam search déterministe (`num_beams=5`) puis vérifie `is_supported()`. Si le score est inférieur à 0,40 ou si la réponse n'est pas extraite du contexte, le modèle répond `Je ne sais pas.`

- `python test.py --prompt "Quelle est la capitale de la France ?" --rag`
- `python evaluate.py` (nécessite `scikit-learn`)
- Le GGUF ne contient ni la base de faits ni le TF-IDF : l'ancrage fonctionne uniquement via les scripts Python.

Métriques actuelles (`python evaluate.py`) :

| Mesure | Résultat |
| --- | --- |
| Questions dans la base | 30 |
| Réponses exactes | 30/30 |
| Réponses ancrées | 30/30 |
| Abstentions à tort | 0/30 |
| Retrieval correct | 30/30 |
| Questions hors base | 10 |
| Abstentions correctes | 10/10 |

Interprétation : avec les 197 réponses directes ajoutées à l'entraînement, le modèle donne la bonne réponse aux 30 questions de la base, et chaque réponse reste un extrait du contexte récupéré. Les abstentions hors base fonctionnent toujours. Ce résultat ne couvre que les formulations proches de la base fermée. Détails dans `doc/12-rag.md`.
