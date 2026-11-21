# 11. Limites et pistes d'amélioration

## Limites actuelles

- Jeu de données réduit : 2 000 exemples Alpaca filtrés + 245 ancrages + 197 Q/R directes.
- GPT-2 pré-entraîné ajusté (124 M paramètres), mais contexte effectivement limité à 256 tokens.
- Le GGUF F16 pèse environ 328 Mo (poids FP32 : 498 Mo).
- Aucun jeu de validation : le loss affiché est uniquement le loss d'entraînement.
- Échantillonnage par défaut avec arrêt au saut de ligne ; le mode `--greedy` boucle encore.
- GGUF exporté en F16, sans quantification.

## Limites du pipeline RAG

- **Base fermée de 30 faits** dans `knowledge.jsonl` : au-delà, le modèle s'abstient (`Je ne sais pas.`) au lieu d'inventer une réponse.
- **Seuil `DEFAULT_MIN_SCORE = 0.40` à recalibrer** : abaissé de 0,50 pour Iris afin de récupérer les reformulations couvertes ; trop bas, il accepte des questions hors base.
- **30 réponses exactes sur 30** sur les formulations de la base : l'ancrage est vérifié, mais une question très reformulée peut retomber en abstention.
- **Dépendance à scikit-learn** (`sklearn`) pour le TF-IDF, côté Python uniquement ; l'ancrage n'est pas présent dans le GGUF / llama.cpp.
- **Pas de validation** : `evaluate.py` donne un diagnostic ponctuel, sans jeu de validation ni validation croisée du retrieval ou du seuil.

Voir [12-rag.md](12-rag.md) pour les commandes et les métriques détaillées.

## Améliorations possibles

1. **Plus de données** : augmenter `--max-examples` dans `prepare_dataset.py` et `train.py` (limite commune portée à 3 000).
2. **Autre modèle de départ** : essayer `--pretrained-model` avec un checkpoint plus capable ; `--init scratch` reste disponible pour l'expérimentation.
3. **Évaluation** : réserver 10 % des exemples pour un jeu de validation et suivre `eval_loss`.
4. **Génération** : régler `temperature`, `top_p`, `top_k`, `repetition_penalty` et `no_repeat_ngram_size` selon l'usage.
5. **GGUF quantifié** : exporter en Q4_K_M pour réduire la taille et accélérer l'inférence.
6. **Ancrage** : étendre `knowledge.jsonl`, recalibrer `--min-score` et comparer les valeurs de `--rag-k` avec `evaluate.py`.
7. **Historique git** : pousser les commits locaux sur GitHub si le dépôt doit être partagé.

## Feuille de route : fiabilité → dialogue → raisonnement

La trajectoire complète, les codenames proposés et les critères de réussite chiffrés sont détaillés dans [13-versions.md](13-versions.md) § « Feuille de route proposée ». En résumé, chaque jalon ferme une partie des limites ci-dessus :

| Jalon | Limites traitées | Ce qui change concrètement |
| --- | --- | --- |
| `1.2.0` Poppy *(proposé)* | Hallucinations hors base proches (2/5), seuil à recalibrer, absence de validation et d'`eval_loss`, GGUF F16 328 Mo, 2 652 exemples | Contrôle de correspondance **entité de la question ↔ passage récupéré** avant génération (le retrieval est déjà 30/30 : agir en amont plutôt qu'ajouter des exemples), 10 % de validation, 3 000 exemples, Q4_K_M |
| `1.3.0` Lavender *(proposé)* | Contexte 256 tokens, `serve.py` à prompt unique, `chat_template` monotone | Données de conversation de 2 à 3 tours, `messages[]` avec session côté API, contexte 512/1024, jeu de test de dialogue figé |
| `2.0.0` Orchid *(proposé)* | GPT‑2 124 M, génération libre approximative, aucun raisonnement | Nouveau modèle de base instruct + LoRA/QLoRA, données de raisonnement en étapes, jeu `eval/reasoning.jsonl` |
| `2.1.0` Sunflower *(proposé)* | Ancrage d'un seul fait, aucune vérification des étapes | RAG multi-sauts, vérification étape par étape, abstention si une étape n'est pas supportée |

Deux points à ne pas confondre : le chat web fourni par llama.app fonctionne déjà avec Iris, mais il ne gère ni historique multi-tours ni raisonnement ; et LoRA, cité comme « changement d'architecture » pour `2.0.0`, ne crée pas de capacité de raisonnement à lui seul — ce sont le modèle de base et les données qui comptent.

Le premier jalon reste le plus court et le plus mesurable : viser 5/5 sur les questions hors base proches et 40/40 au total sur le jeu figé, sans changer les poids de base.

## Rappel

Ce projet sert avant tout à comprendre le cycle complet : donnée → entraînement → scripts Python → GGUF → publication Hugging Face → test dans llama.app.
