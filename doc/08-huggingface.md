# 8. Publication Hugging Face et llama.app

## Dépôt publié

https://huggingface.co/nazimboudeffa/PicoGPT1-Daisy

Le dépôt est public et contient :

- les poids Transformers et le tokenizer ;
- `PicoGPT1-Daisy.gguf` ;
- le `README.md` (fiche modèle) généré depuis `MODEL_CARD.md`.

Hugging Face reconnaît automatiquement les tags `gguf`, `llama.cpp`, `llama.app` et affiche les commandes correspondantes dans « Use this model ».

## Publication

Prérequis : être connecté via `huggingface-cli login` (ou avoir `HF_TOKEN`), et avoir généré le GGUF.

```bash
python export_gguf.py
python publish_hf.py
```

`publish_hf.py` :

1. Crée le dépôt s'il n'existe pas (`private=False`, `exist_ok=True`).
2. Upload le contenu de `PicoGPT1-Daisy/` en ignorant `checkpoint-*`.
3. Upload `MODEL_CARD.md` comme `README.md`.
4. Upload `PicoGPT1-Daisy.gguf`.

Le script configure une session HTTP avec retries pour absorber les erreurs TLS intermittentes observées sur certains réseaux.

## Tester avec le CLI llama.app

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Daisy -hff PicoGPT1-Daisy.gguf
```

`-hff` force le fichier GGUF à utiliser. Comme le dépôt ne contient qu'un seul GGUF, la forme courte fonctionne aussi :

```bash
llama cli -hf nazimboudeffa/PicoGPT1-Daisy
```

## Interface web de chat

```bash
llama serve -hf nazimboudeffa/PicoGPT1-Daisy -hff PicoGPT1-Daisy.gguf
# http://localhost:9931
```

Le port par défaut du serveur llama.app est `9931`.

## Éviter les hallucinations dans llama.app

Le GGUF ne contient ni `knowledge.jsonl` ni le code TF-IDF : sans contexte, le modèle peut répondre n'importe quoi de manière fluide. En revanche, il a été entraîné à **extraire la réponse d'un contexte** fourni dans le prompt. Dans llama.app, collez le fait utile dans le format appris :

```
Contexte: La capitale de la France est Paris. | Question: Quelle est la capitale de la France ?
```

Le modèle répond `Paris`. Plusieurs faits peuvent être placés dans le `Contexte:`, séparés par des espaces : le modèle choisit toujours l'extrait qui répond à la question. Les faits sont dans `knowledge.jsonl` (champ `text`). Le pipeline Python (`python test.py --rag`) fait automatiquement cette sélection par TF-IDF ; dans llama.app, elle doit être faite manuellement.

## Tester avec Transformers

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
