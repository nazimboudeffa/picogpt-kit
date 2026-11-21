# 2. Installation

## Prérequis

- Python 3.10 ou plus récent (testé avec Python 3.13)
- Git
- Environnement Windows, macOS ou Linux

## Cloner le dépôt

```bash
git clone https://github.com/nazimboudeffa/picogpt1-daisy.git
cd picogpt1-daisy
```

## Dépendances Python

```bash
python -m pip install torch transformers datasets huggingface_hub scikit-learn
```

`tokenizers`, `accelerate` et `safetensors` sont installés automatiquement avec `transformers`. `scikit-learn` est requis par le pipeline d'ancrage (`rag.py`).

### Entraînement sur GPU (optionnel)

Par défaut, `pip install torch` installe une version **CPU** sur Windows. Pour utiliser une carte NVIDIA, installez la roue CUDA correspondant à votre version de PyTorch :

```bash
python -m pip install --upgrade "torch==2.7.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128
```

Vérification :

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Si `cuda available` vaut `True`, `train.py` et `generate()` utilisent automatiquement le GPU. Lancez ensuite `python train.py --fp16` pour réduire la mémoire et accélérer l'entraînement. Sans CUDA, la même commande fonctionne sur CPU (mais `--fp16` doit être omis).

Vérification rapide :

```bash
python -m py_compile train.py test.py serve.py model_utils.py prepare_dataset.py prepare_rag_dataset.py rag.py evaluate.py export_gguf.py publish_hf.py
```

Aucune sortie signifie que la syntaxe est correcte.

## Connexion Hugging Face

La connexion n'est nécessaire **que pour publier** le modèle :

```bash
huggingface-cli login
```

Le token est saisi dans le formulaire interactif. Il ne doit jamais être collé dans un chat ou commité dans le dépôt.

## llama.app / CLI llama.cpp

Pour tester le modèle publié avec llama.app :

- Windows : installer l'application llama.app, ou `winget install llama.cpp`
- macOS/Linux : `curl -LsSf https://llama.app/install.sh | sh`

Vérifier l'installation :

```bash
llama --version
```

## Poids du modèle

Les poids (`model.safetensors`) et le GGUF sont ignorés par Git. Deux options :

1. Entraîner le modèle localement : `python train.py`
2. Télécharger le modèle publié : `llama cli -hf nazimboudeffa/PicoGPT1-Daisy -hff PicoGPT1-Daisy.gguf`
