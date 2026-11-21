"""Génère `negative_train.txt` : abstentions sur des questions proches de la base.

Ces questions portent sur le même thème qu'un fait connu, mais introduisent
une entité absente de `knowledge.jsonl` (autre pays, autre substance, autre
langage...). Le pipeline RAG récupère tout de même le fait le plus proche : il
faut apprendre au modèle à refuser de recopier sa réponse.

Les exemples sont construits avec le vrai prompt d'inférence (`rag.build_prompt`)
afin que l'apprentissage corresponde exactement à ce que le modèle verra.

Les questions du jeu de test figé (`eval/test.jsonl`) ne sont PAS utilisées ici.

À lancer après `prepare_rag_dataset.py` et `prepare_facts_dataset.py`, avant `train.py`.
"""

import json
from pathlib import Path

from rag import ABSTENTION, DEFAULT_MIN_SCORE, get_kb

KB_PATH = Path("knowledge.jsonl")
OUTPUT_FILE = Path("negative_train.txt")

# Trois questions hors base par fait : même domaine, entité différente.
NEAR_MISS = {
    "capital_fr": [
        "Quelle est la capitale du Portugal ?",
        "Quelle est la capitale de l'Allemagne ?",
        "Quelle est la capitale du Canada ?",
    ],
    "boiling_water": [
        "À quelle température bout l'huile ?",
        "À quelle température fond la glace ?",
        "À quelle température bout le lait ?",
    ],
    "water_formula": [
        "Quelle est la formule chimique du sucre ?",
        "Quelle est la formule chimique de l'ammoniac ?",
        "Quelle est la formule chimique du dioxyde de carbone ?",
    ],
    "photosynthesis": [
        "Que permet la photosynthèse chez les algues ?",
        "À quoi sert la respiration cellulaire ?",
        "Quel processus permet aux animaux de convertir la lumière ?",
    ],
    "moon": [
        "Quel est le satellite naturel de Mars ?",
        "Quel est le satellite naturel de Jupiter ?",
        "Quelle planète tourne autour de la Lune ?",
    ],
    "mercury": [
        "Quelle planète est la plus proche de la Terre ?",
        "Quelle planète est la plus éloignée du Soleil ?",
        "Quelle planète est la plus proche de Vénus ?",
    ],
    "jupiter": [
        "Quelle est la plus petite planète du système solaire ?",
        "Quelle est la plus grande lune du système solaire ?",
        "Quelle est la plus grande planète de la Galaxie ?",
    ],
    "heart": [
        "Combien de cavités possède le cœur d'un poisson ?",
        "Combien de cavités a le cœur d'un reptile ?",
        "Combien de ventricules possède le cœur humain ?",
    ],
    "brain": [
        "Quel organe dirige le système digestif ?",
        "Quel est l'organe principal du système nerveux chez les insectes ?",
        "Quelle partie du cerveau contrôle la respiration ?",
    ],
    "gravity": [
        "Quelle est l'accélération de la gravité sur la Lune ?",
        "Quelle est la force de gravité sur Mars ?",
        "À combien est estimée la gravité terrestre en altitude ?",
    ],
    "light_speed": [
        "À quelle vitesse le son se propage-t-il dans l'air ?",
        "À quelle vitesse la lumière se propage-t-elle dans l'eau ?",
        "Quelle est la vitesse du son dans le vide ?",
    ],
    "french_revolution": [
        "En quelle année a commencé la Révolution américaine ?",
        "En quelle année s'est terminée la Révolution française ?",
        "En quelle année a commencé la Révolution russe ?",
    ],
    "ww2_end": [
        "En quelle année a commencé la Seconde Guerre mondiale ?",
        "En quelle année s'est terminée la Première Guerre mondiale ?",
        "En quelle année s'est terminée la Guerre froide ?",
    ],
    "printing": [
        "Qui a inventé l'écriture ?",
        "Qui a inventé l'imprimerie au XXe siècle ?",
        "Qui a créé la première machine à vapeur ?",
    ],
    "web": [
        "Qui a créé le réseau Internet ?",
        "Qui a inventé le navigateur web ?",
        "Qui a créé le World Wide Web en 1995 ?",
    ],
    "python": [
        "Qui a créé le langage Ruby ?",
        "Qui a inventé le langage C ?",
        "Quel programmeur a créé le langage JavaScript ?",
    ],
    "html": [
        "Que signifie l'abréviation CSS ?",
        "Que veut dire le sigle XML ?",
        "Que signifie HTTP ?",
    ],
    "capital_be": [
        "Quelle est la capitale des Pays-Bas ?",
        "Quelle est la capitale du Luxembourg ?",
        "Quelle est la capitale de la Suisse ?",
    ],
    "capital_it": [
        "Quelle est la capitale de la Grèce ?",
        "Quelle est la capitale de l'Autriche ?",
        "Quelle est la capitale de la Hongrie ?",
    ],
    "capital_jp": [
        "Quelle est la capitale de la Corée du Sud ?",
        "Quelle est la capitale de la Chine ?",
        "Quelle est la capitale de l'Inde ?",
    ],
    "pacific": [
        "Quel est le plus grand lac de la Terre ?",
        "Quel est le plus petit océan du monde ?",
        "Quel est le plus grand désert de la Terre ?",
    ],
    "everest": [
        "Quelle est la deuxième plus haute montagne du monde ?",
        "Quelle est la plus haute montagne d'Europe ?",
        "Quel est le plus haut sommet d'Afrique ?",
    ],
    "colors": [
        "Quelles sont les couleurs primaires en lumière ?",
        "Quelles sont les couleurs secondaires en peinture ?",
        "Quelles sont les couleurs du drapeau belge ?",
    ],
    "honey": [
        "À partir de quoi les abeilles produisent-elles de la cire ?",
        "Que produisent les fourmis à partir du nectar ?",
        "À partir de quoi les abeilles produisent-elles du pollen ?",
    ],
    "sleep": [
        "Combien d'heures dort un enfant en moyenne ?",
        "Combien d'heures dort un adulte chaque semaine ?",
        "Combien d'heures dort une chauve-souris ?",
    ],
    "vaccine": [
        "Qu'est-ce qu'un antibiotique stimule ?",
        "Quel système un vaccin détruit-il ?",
        "Quel système un médicament stimule-t-il ?",
    ],
    "cpu": [
        "À l'aide de quoi un ordinateur stocke-t-il l'information ?",
        "Grâce à quel composant une voiture traite-t-elle l'information ?",
        "À l'aide de quoi un téléphone traite-t-il les images ?",
    ],
    "flag_fr": [
        "Quelles sont les couleurs du drapeau italien ?",
        "Quelles sont les couleurs du drapeau allemand ?",
        "Quelles sont les couleurs du drapeau suisse ?",
    ],
    "euro": [
        "Quelle monnaie la Belgique utilise-t-elle ?",
        "Quelle monnaie le Japon utilise-t-il ?",
        "Quelle devise la France utilisait-elle avant 1999 ?",
    ],
    "sun": [
        "Qu'est-ce qu'une planète ?",
        "Qu'est-ce qu'une comète ?",
        "Quelle catégorie d'astre est Vénus ?",
    ],
}


def main():
    kb = get_kb()
    items = [
        json.loads(line)
        for line in KB_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    known_ids = {item["id"] for item in items}
    unknown = set(NEAR_MISS) - known_ids
    if unknown:
        raise SystemExit(f"IDs inconnus dans NEAR_MISS : {sorted(unknown)}")

    examples = []
    seen = set()
    for fact_id, questions in NEAR_MISS.items():
        for question in questions:
            if question in seen:
                raise SystemExit(f"Question dupliquée : {question}")
            seen.add(question)

            prompt, _context = kb.build_prompt(question, k=1, min_score=DEFAULT_MIN_SCORE)
            examples.append((prompt.rstrip("\n"), ABSTENTION))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, (prompt, response) in enumerate(examples):
            f.write(f"{prompt}\n{response}")
            if i < len(examples) - 1:
                f.write("\n\n")

    print(f"{len(examples)} exemples écrits dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
