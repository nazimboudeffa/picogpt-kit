# 4. Entraînement

## Commande recommandée

```bash
python prepare_dataset.py --max-examples 2000   # train.txt : 2 000 exemples Alpaca filtrés
python prepare_rag_dataset.py                   # rag_train.txt : 245 exemples ancrés
python prepare_facts_dataset.py                 # facts_train.txt : 197 Q/R directes
python train.py                                 # fine-tuning de GPT-2 (défaut)
```

Le script nécessite `train.txt` et, idéalement, `rag_train.txt` (voir [dataset](03-dataset.md)).

`train.py` part par défaut du modèle pré-entraîné `gpt2` (124 M paramètres, 12 couches, 768 dimensions). C'est le réglage qui donne des réponses cohérentes dans le GGUF / llama.app : un modèle entraîné de zéro sur quelques milliers d'exemples ne peut pas apprendre la langue correctement, même en agrandissant l'architecture.

## Retour à l'entraînement depuis zéro

```bash
python train.py --init scratch --n-layer 4 --n-embd 256 --n-head 4
```

Les options `--n-layer`, `--n-embd` et `--n-head` ne sont utilisées qu'avec `--init scratch` ; elles sont ignorées en mode `pretrained`, où l'architecture est fixée par les poids de départ.

## Options CLI

| Option | Défaut | Description |
| --- | --- | --- |
| `--data-paths` | `train.txt rag_train.txt facts_train.txt` | Fichiers d'exemples (un fichier absent est ignoré) |
| `--output-dir` | `PicoGPT1-Iris` | Dossier de sauvegarde |
| `--max-len` | `256` | Longueur maximale du contexte |
| `--max-examples` | `3000` | Nombre maximum d'exemples, tous fichiers confondus |
| `--epochs` | `3` en fine-tuning, `10` depuis zéro | Nombre d'epochs |
| `--batch-size` | `4` | Taille du batch |
| `--lr` | `5e-5` en fine-tuning, `5e-4` depuis zéro | Taux d'apprentissage |
| `--init` | `pretrained` | `pretrained` (recommandé) ou `scratch` |
| `--pretrained-model` | `gpt2` | Identifiant Hugging Face du modèle de départ |
| `--n-layer` / `--n-embd` / `--n-head` | `4` / `256` / `4` | Architecture en mode `scratch` |
| `--save-steps` | `1000` | Fréquence des checkpoints intermédiaires |
| `--fp16` | désactivé | Demi-précision ; à utiliser uniquement avec CUDA (`torch.cuda.is_available()`) |

`MAX_LEN` est aussi utilisé dans `model_utils.py` pour tronquer les prompts à la génération : les deux valeurs doivent rester identiques. Le modèle pré-entraîné garde ses 1 024 positions : seules les 256 premières sont utilisées, puisque les exemples sont tronqués à `max_len`.

## Ce que fait le script

1. Charge le tokenizer depuis `--output-dir` s'il y est, sinon depuis le modèle de départ : même vocabulaire, pas de requête réseau inutile.
2. Définit un `chat_template` correspondant au format appris : `prompt\nreponse<|endoftext|>`.
3. Lit les fichiers de données et tokenise chaque exemple (tronqué à `max_len`).
4. Masque le prompt dans les labels (`-100`) : la loss ne porte que sur la réponse.
5. Charge GPT-2 pré-entraîné ou construit un `GPT2LMHeadModel` aléatoire selon `--init`.
6. Entraîne avec `Trainer` (`logging_steps=50`, `save_steps` configurable).
7. Sauvegarde le modèle et le tokenizer dans `--output-dir`.

## Sorties

Fichiers générés dans `PicoGPT1-Iris/` :

- `config.json`
- `generation_config.json`
- `model.safetensors` (ignoré par Git, ~500 Mo avec GPT-2 complet)
- `merges.txt`, `vocab.json`, `tokenizer.json`
- `special_tokens_map.json`, `tokenizer_config.json`

Un dossier de checkpoint peut apparaître ; il est ignoré par Git et peut être supprimé sans risque.

Après l'entraînement, régénérez le GGUF pour llama.app :

```bash
python export_gguf.py
```

## Durée observée

Dernier fine-tuning (2 442 exemples, 5 epochs, ≈ 3 050 steps) : **250,8 s sur RTX 3070** en `--fp16` (12,2 steps/s, ~49 exemples/s), contre nettement plus sur CPU. Loss moyenne : 1,58 ; dernière loss journalisée : 1,05.
