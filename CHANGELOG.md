# Changelog

Toutes les évolutions notables du projet sont consignées ici. Le codename (nom de fleur) change à chaque itération : la convention est décrite dans [doc/13-versions.md](doc/13-versions.md).

Format d'une release : `PicoGPT<majeure>-<Fleur>`, avec un tag git `v<version>-<fleur en minuscules>`.

## [Non publié]

- Ajout : jeu de test figé `eval/test.jsonl` (40 questions jamais vues) et contrôle de fuite `eval/check_leakage.py`.
- Ajout : `evaluate.py --split test` écrit un rapport par codename dans `reports/`, avec empreinte du jeu, commit et détails des réponses.
- Ajout : `compare_reports.py` et [doc/14-benchmark.md](doc/14-benchmark.md).
- Mesure de référence sur Daisy : 28/30 réponses exactes sur le jeu test (contre 30/30 sur le jeu train), 30/30 retrieval, 5/5 abstentions hors base éloignées, 1/5 hors base proches. Voir `reports/daisy-test.json`.
- Codename retenu pour la release suivante : `Iris` (`1.1.0`), avec pour objectif de corriger les hallucinations sur les questions hors base proches.

## [1.0.0] — Daisy — 2026-11-21

Première version documentée du kit.

- Ajout : fine-tuning de GPT-2 pré-entraîné (124 M paramètres) sur 2 442 exemples.
- Ajout : pipeline RAG ancré dans `knowledge.jsonl` (30 faits), avec abstention hors base.
- Ajout : export GGUF et publication sur Hugging Face.
- Ajout : `project.json`, `config.py`, `rename_codename.py` et cette convention de versions.

## [0.x] — Jasmine — 2026-11-20

Prototype historique, conservé tel quel comme référence : petit GPT-2 entraîné de zéro sur 500 exemples Alpaca, décodage glouton, sans ancrage. Voir <https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine>.
