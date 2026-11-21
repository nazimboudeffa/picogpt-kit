# Changelog

Toutes les évolutions notables du projet sont consignées ici. Le codename (nom de fleur) change à chaque itération : la convention est décrite dans [doc/13-versions.md](doc/13-versions.md).

Format d'une release : `PicoGPT<majeure>-<Fleur>`, avec un tag git `v<version>-<fleur en minuscules>`.

## [Non publié]

- Ajout : jeu de test figé `eval/test.jsonl` (40 questions jamais vues) et contrôle de fuite `eval/check_leakage.py`.
- Ajout : `evaluate.py --split test` écrit un rapport par codename dans `reports/`, avec empreinte du jeu, commit et détails des réponses.
- Ajout : `compare_reports.py` et [doc/14-benchmark.md](doc/14-benchmark.md).
- Mesure de référence sur Daisy : 28/30 réponses exactes sur le jeu test (contre 30/30 sur le jeu train), 30/30 retrieval, 5/5 abstentions hors base éloignées, 1/5 hors base proches. Voir `reports/daisy-test.json`.

## [1.1.0] — Iris — 2026-11-21

Changement de codename : `PicoGPT1-Daisy` → `PicoGPT1-Iris`.

- Ajout : 90 exemples négatifs générés par `prepare_negative_dataset.py` (questions même domaine mais entité absente de la base) pour apprendre à refuser le contexte récupéré à tort.
- Ajout : 120 reformulations ancrées générées par `prepare_paraphrase_dataset.py`, pour éviter que le modèle n'apprenne « formulation inédite → abstention ».
- Réglage : `DEFAULT_MIN_SCORE` passe de 0,50 à 0,40 ; deux reformulations couvertes étaient refusées juste sous l'ancien seuil.
- Résultat mesuré sur le jeu figé : 29/30 exactes, 30/30 ancrées, 30/30 retrieval, 5/5 hors base éloignées, 2/5 hors base proches (`reports/iris-test.json`). Gain net de 1 point sur Daisy (36/40 contre 35/40), mais le problème d'hallucination proche n'est que partiellement corrigé.

## [1.0.0] — Daisy — 2026-11-21

Première version documentée du kit.

- Ajout : fine-tuning de GPT-2 pré-entraîné (124 M paramètres) sur 2 442 exemples.
- Ajout : pipeline RAG ancré dans `knowledge.jsonl` (30 faits), avec abstention hors base.
- Ajout : export GGUF et publication sur Hugging Face.
- Ajout : `project.json`, `config.py`, `rename_codename.py` et cette convention de versions.

## [0.x] — Jasmine — 2026-11-20

Prototype historique, conservé tel quel comme référence : petit GPT-2 entraîné de zéro sur 500 exemples Alpaca, décodage glouton, sans ancrage. Voir <https://huggingface.co/nazimboudeffa/PicoGPT1-Jasmine>.
