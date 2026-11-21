"""Génère `facts_train.txt` : questions/réponses directes, sans contexte.

Objectif : le modèle doit répondre directement aux questions de la base fermée,
même avec une formulation différente de celle de `knowledge.jsonl`, au lieu de
se contenter d'extraire un texte fourni dans le prompt ou de s'abstenir.

À lancer après `prepare_rag_dataset.py`, avant `train.py`.
"""

import json
from pathlib import Path

KB_PATH = Path("knowledge.jsonl")
OUTPUT_FILE = Path("facts_train.txt")

# Formulations supplémentaires pour les faits les plus souvent demandés.
ALIASES = {
    "capital_fr": [
        "quelle est la capitale de France ?",
        "la capitale de la France ?",
        "capitale de la France",
        "France : capitale",
        "Quelle ville est la capitale de la France ?",
    ],
    "boiling_water": [
        "à quelle température l'eau bout-elle ?",
        "l'eau bout à quelle température au niveau de la mer ?",
        "température d'ébullition de l'eau au niveau de la mer",
    ],
    "water_formula": [
        "quelle est la formule de l'eau ?",
        "formule chimique de l'eau",
        "H2O, c'est l'eau ?",
    ],
    "moon": [
        "quel est le satellite de la Terre ?",
        "le satellite naturel de la Terre",
        "la Lune tourne autour de quoi ?",
    ],
    "heart": [
        "combien de cavités a le cœur ?",
        "le cœur humain a combien de cavités ?",
    ],
    "python": [
        "qui a inventé Python ?",
        "créateur du langage Python",
    ],
    "web": [
        "qui a inventé le World Wide Web ?",
        "créateur du Web",
    ],
    "colors": [
        "quelles sont les couleurs primaires ?",
        "couleurs primaires en peinture",
    ],
    "euro": [
        "quelle monnaie utilise la France ?",
        "la France utilise quelle monnaie ?",
    ],
    "sun": [
        "qu'est-ce que le Soleil ?",
        "le Soleil est une étoile ?",
    ],
}

# Petit jeu de calculs élémentaires : la base fermée ne les contient pas, mais
# l'utilisateur les pose naturellement en chat.
ARITHMETIC = [
    ("que font 2+2 ?", "4"),
    ("combien font 2 + 2 ?", "4"),
    ("2+2 ?", "4"),
    ("what is 2+2?", "4"),
    ("quelle est la somme de 2 et 2 ?", "4"),
    ("que font 3+3 ?", "6"),
    ("combien font 3 + 3 ?", "6"),
    ("what is 3+3?", "6"),
    ("que font 10+5 ?", "15"),
    ("que font 7-3 ?", "4"),
    ("combien font 7 - 3 ?", "4"),
    ("que font 6*2 ?", "12"),
    ("que font 6x2 ?", "12"),
    ("combien font 6 multiplié par 2 ?", "12"),
    ("que font 8/2 ?", "4"),
    ("combien font 8 divisé par 2 ?", "4"),
]

# Abstention sur des questions hors base, sans préfixe : dans llama.app,
# l'utilisateur tape directement sa question.
OUT_OF_SCOPE = [
    ("quel est le prix du pain à paris ?", "Je ne sais pas."),
    ("quel temps fera-t-il demain à lyon ?", "Je ne sais pas."),
    ("qui a gagné la coupe du monde de 2030 ?", "Je ne sais pas."),
    ("quel est le numéro de téléphone du musée ?", "Je ne sais pas."),
    ("quelle est la recette secrète de ce restaurant ?", "Je ne sais pas."),
]


def question_variants(question):
    """Quelques reformulations mécaniques, sans changer le sens."""
    q = question.strip()
    variants = [
        q,
        q.lower(),
        q.rstrip("?"),
        q.rstrip("?").lower(),
        f"Question: {q}",
    ]
    return [v for v in variants if v]


def main():
    items = [
        json.loads(line)
        for line in KB_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    examples = []
    seen = set()

    def add(question, answer):
        if question in seen:
            return
        seen.add(question)
        examples.append((question, answer))

    for item in items:
        for question in question_variants(item["question"]) + ALIASES.get(item["id"], []):
            add(question, item["answer"])

    for question, answer in ARITHMETIC + OUT_OF_SCOPE:
        add(question, answer)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, (prompt, response) in enumerate(examples):
            f.write(f"{prompt}\n{response}")
            if i < len(examples) - 1:
                f.write("\n\n")

    print(f"{len(examples)} exemples écrits dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
