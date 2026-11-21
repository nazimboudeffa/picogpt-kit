# Changelog

Toutes les évolutions notables du projet sont consignées ici. Le codename (nom de fleur) change à chaque itération : la convention est décrite dans [doc/13-versions.md](doc/13-versions.md).

Format d'une release : `PicoGPT<majeure>-<Fleur>`, avec un tag git `v<version>-<fleur en minuscules>`.

## [1.0.0] — Daisy — 2026-11-21

Première version documentée du kit.

- Ajout : fine-tuning de GPT-2 pré-entraîné (124 M paramètres) sur 2 442 exemples.
- Ajout : pipeline RAG ancré dans `knowledge.jsonl` (30 faits), avec abstention hors base.
- Ajout : export GGUF et publication sur Hugging Face.
- Ajout : `project.json`, `config.py`, `rename_codename.py` et cette convention de versions.

## [0.x] — Jasmine — 2026-11-20

Prototype historique, conservé tel quel comme référence : petit GPT-2 entraîné de zéro sur 500 exemples Alpaca, décodage glouton, sans ancrage. Voir <https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine>.
