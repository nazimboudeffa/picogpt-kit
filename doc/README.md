# Documentation — PicoGPT1-Iris

Cette documentation décrit le dépôt local, l'entraînement, la génération, le serveur HTTP, l'export GGUF, la publication sur Hugging Face et le pipeline d'ancrage (RAG).

## Sommaire

1. [Présentation](01-presentation.md)
2. [Installation](02-installation.md)
3. [Dataset](03-dataset.md)
4. [Entraînement](04-entrainement.md)
5. [Génération avec `test.py`](05-generation.md)
6. [Serveur local avec `serve.py`](06-serveur-local.md)
7. [Export GGUF](07-gguf.md)
8. [Publication Hugging Face et llama.app](08-huggingface.md)
9. [Architecture du modèle](09-architecture.md)
10. [Dépannage](10-depannage.md)
11. [Limites et pistes d'amélioration](11-limites.md)
12. [Ancrage / RAG](12-rag.md)
13. [Versions et noms de fleurs](13-versions.md)
14. [Protocole de benchmark](14-benchmark.md)

## Démarrage rapide

```bash
python -m pip install torch transformers datasets huggingface_hub scikit-learn
python prepare_dataset.py --max-examples 2000
python prepare_rag_dataset.py
python prepare_facts_dataset.py
python train.py
python test.py --prompt "Quelle est la capitale de la France ?" --rag
python evaluate.py
python eval/check_leakage.py
python evaluate.py --split test
python compare_reports.py
```

Serveur local :

```bash
python serve.py
```

Export et publication (nécessite `huggingface-cli login`) :

```bash
python export_gguf.py
python publish_hf.py
```

## Liens

- Dépôt GitHub : https://github.com/nazimboudeffa/picogpt-kit
- Modèle Hugging Face : https://huggingface.co/nazimboudeffa/PicoGPT1-Iris
- Historique des releases : https://github.com/nazimboudeffa/picogpt-kit/blob/main/CHANGELOG.md
- llama.app : https://llama.app
