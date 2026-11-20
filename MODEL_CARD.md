---
library_name: transformers
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

Petit modèle GPT-2 entraîné sur un sous-ensemble de 500 instructions/réponses Alpaca filtrées, complété de 245 exemples d'ancrage issus d'une base fermée de 30 faits.

- Architecture : 2 couches, 128 dimensions, 2 têtes d'attention, contexte 256 tokens
- Tokenizer : GPT-2 (50 257 tokens)
- Format appris : `prompt\nreponse<|endoftext|>`
- Fichiers : poids Transformers (`model.safetensors`) et variante `PicoGPT1-Daisy.gguf`

> Modèle expérimental de très petite taille : les réponses peuvent être incohérentes.

## Différences avec `PicoGPT1-Jasmine`

L'ancien dépôt [`nazimboudeffa/PicoGPT1-Jasmine`](https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine) est conservé tel quel comme référence historique. Celui-ci est une **nouvelle version**, sans changement d'architecture.

| Élément | `PicoGPT1-Jasmine` (ancien) | `PicoGPT1-Daisy` (ce dépôt) |
| --- | --- | --- |
| Architecture | GPT-2 : 2 couches, 128 dims, 2 têtes, contexte 256 | **Identique** (6,86 M paramètres) |
| Jeu d'entraînement | 500 exemples Alpaca bruts | 500 exemples Alpaca **filtrés** + 245 exemples d'ancrage = **745 exemples** |
| Données | Quelques erreurs factuelles et doublons | Erreurs connues retirées (eau à 115 °C, fausse définition du « mole », `<nooutput>`), doublons supprimés, format uniformisé |
| Décodage | Glouton par défaut, boucles fréquentes | Échantillonnage par défaut (`top_k=40`, `no_repeat_ngram_size=3`, `repetition_penalty=1.15`), arrêt à l'EOS ou au premier `\n`, séparateur `\n` garanti |
| Ancrage / abstention | Aucun | `knowledge.jsonl` (30 faits), TF-IDF, copie contrainte du contexte, `Je ne sais pas.` hors base |
| Entraînement | 10 epochs, ~1 250 steps, loss finale ≈ 4,95 | 10 epochs, ~1 860 steps, loss finale ≈ 4,54 |
| GGUF | Ancien export | Régénéré depuis les nouveaux poids |

En résumé : **même architecture, mais données nettoyées, décodage corrigé et ancrage anti-hallucination**. Le passage d'une génération libre à une réponse extraite d'un contexte vérifié est le changement le plus important. Le code source et la documentation détaillée sont dans <https://github.com/nazimboudeffa/picogpt1-daisy>.

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

Le dépôt fournit un pipeline d'ancrage côté Python : `rag.py` récupère un fait dans `knowledge.jsonl` avec un TF-IDF (`char_wb` 3-5) sur `text + question`, construit le prompt `Contexte: <texte> | Question: <question>\n`, contraint le décodage aux extraits littéraux du contexte (beam search déterministe, `num_beams=5`), puis vérifie `is_supported()`. Si le score est inférieur à 0,50 ou si la réponse n'est pas supportée, le modèle répond `Je ne sais pas.`

- `python test.py --prompt "Quelle est la capitale de la France ?" --rag`
- `python evaluate.py` (nécessite `scikit-learn`)
- Le GGUF ne contient ni la base de faits ni le TF-IDF : l'ancrage fonctionne uniquement via les scripts Python.

Métriques actuelles (`python evaluate.py`) :

| Mesure | Résultat |
| --- | --- |
| Questions dans la base | 30 |
| Réponses exactes | 9/30 |
| Réponses ancrées | 30/30 |
| Abstentions à tort | 0/30 |
| Retrieval correct | 30/30 |
| Questions hors base | 10 |
| Abstentions correctes | 10/10 |

Interprétation : le modèle ne fabrique plus de fait absent du contexte ; l'extrait choisi n'est pas toujours le meilleur (9 exacts), mais il est toujours supporté par le texte récupéré. Détails dans `doc/12-rag.md`.
