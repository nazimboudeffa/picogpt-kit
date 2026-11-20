# 9. Architecture du modèle

## Configuration

Définie dans `train.py` et sauvegardée dans `PicoGPT1-Daisy/config.json` :

| Paramètre | Valeur |
| --- | --- |
| `model_type` | `gpt2` |
| `architectures` | `GPT2LMHeadModel` |
| `n_layer` | `2` |
| `n_embd` | `128` |
| `n_head` | `2` |
| `n_positions` / `n_ctx` | `256` |
| `vocab_size` | `50 257` |
| `activation_function` | `gelu_new` |
| `torch_dtype` | `float32` |

Nombre de paramètres : environ **6,86 M**.

## Tokenizer

- Tokenizer GPT-2 (`GPT2TokenizerFast`).
- `<|endoftext|>` sert à la fois de token EOS, BOS, UNK et PAD.
- L'ID du token spécial est `50256`.
- `model_max_length` est fixé à `256`.

## Format d'entraînement

Chaque exemple est tokenisé comme :

```
{prompt}\n{reponse}<|endoftext|>
```

Les tokens appartenant au prompt (y compris le `\n` final) ont un label `-100` : ils sont ignorés dans le calcul de la loss. Le modèle apprend donc uniquement à produire la réponse.

## Chat template

Sauvegardé dans `tokenizer_config.json` et repris dans le GGUF :

```jinja
{% for message in messages %}{{ message['content'] }}{% if not loop.last %}{{ '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '\n' }}{% endif %}
```

Ce template correspond au format d'entraînement : les messages sont simplement séparés par des sauts de ligne, et un saut de ligne est ajouté quand llama.cpp demande le début de la réponse.

## Génération

`model_utils.generate` utilise :

- `max_new_tokens` (64 par défaut) ;
- `pad_token_id = eos_token_id = 50256` ;
- échantillonnage par défaut (`temperature=0.8`, `top_p=0.9`, `top_k=40`, `repetition_penalty=1.15`) ;
- `no_repeat_ngram_size=3` et arrêt à l'EOS ou au premier saut de ligne généré.

Le prompt est tronqué à 256 tokens.
