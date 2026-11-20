# 11. Limites et pistes d'amélioration

## Limites actuelles

- Jeu de données réduit : 500 exemples Alpaca filtrés + 245 exemples d'ancrage.
- Architecture très petite : 2 couches, 128 dimensions.
- Contexte limité à 256 tokens.
- Aucun jeu de validation : le loss affiché est uniquement le loss d'entraînement.
- Échantillonnage par défaut avec arrêt au saut de ligne ; le mode `--greedy` boucle encore.
- GGUF exporté en F16, sans quantification.

## Limites du pipeline RAG

- **Base fermée de 30 faits** dans `knowledge.jsonl` : au-delà, le modèle s'abstient (`Je ne sais pas.`) au lieu d'inventer une réponse.
- **Seuil `DEFAULT_MIN_SCORE = 0.50` à recalibrer** : trop haut, il refuse des questions couvertes ; trop bas, il accepte des questions hors base.
- **9 réponses exactes sur 30** : l'extrait choisi est toujours supporté par le contexte (30/30), mais ce n'est pas toujours le meilleur extrait.
- **Dépendance à scikit-learn** (`sklearn`) pour le TF-IDF, côté Python uniquement ; l'ancrage n'est pas présent dans le GGUF / llama.cpp.
- **Pas de validation** : `evaluate.py` donne un diagnostic ponctuel, sans jeu de validation ni validation croisée du retrieval ou du seuil.

Voir [12-rag.md](12-rag.md) pour les commandes et les métriques détaillées.

## Améliorations possibles

1. **Plus de données** : augmenter `MAX_EXAMPLES` dans `prepare_dataset.py` et `train.py` (limite commune portée à 2 000).
2. **Modèle plus grand** : passer à 4 couches / 256 dimensions, ou partir d'un GPT-2 pré-entraîné.
3. **Évaluation** : réserver 10 % des exemples pour un jeu de validation et suivre `eval_loss`.
4. **Génération** : régler `temperature`, `top_p`, `top_k`, `repetition_penalty` et `no_repeat_ngram_size` selon l'usage.
5. **GGUF quantifié** : exporter en Q4_K_M pour réduire la taille et accélérer l'inférence.
6. **Ancrage** : étendre `knowledge.jsonl`, recalibrer `--min-score` et comparer les valeurs de `--rag-k` avec `evaluate.py`.
7. **Historique git** : pousser les commits locaux sur GitHub si le dépôt doit être partagé.

## Rappel

Ce projet sert avant tout à comprendre le cycle complet : donnée → entraînement → scripts Python → GGUF → publication Hugging Face → test dans llama.app.
