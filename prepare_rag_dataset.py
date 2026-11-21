"""Génère `rag_train.txt` : exemples ancrés dans une base fermée + abstention.

Format identique à `train.txt` (deux lignes par bloc) pour que `train.py`
puisse lire les deux fichiers sans changer de parseur.

Lancer après `prepare_dataset.py`.
"""

import json
from pathlib import Path

from rag import ABSTENTION, CONTEXT_PREFIX, QUESTION_PREFIX, SEPARATOR

KB_PATH = Path("knowledge.jsonl")
OUTPUT_FILE = Path("rag_train.txt")

# Questions totalement absentes de la base : le modèle doit refuser de répondre.
OUT_OF_SCOPE_QUESTIONS = [
    "Quel est le prix du pain à Paris ?",
    "Qui a gagné la Coupe du monde de 2030 ?",
    "Quel temps fera-t-il demain à Lyon ?",
    "Combien coûte un billet de train pour Berlin ?",
    "Quel est le nom du chat de la voisine ?",
    "Quelle est la recette secrète de ce restaurant ?",
    "Qui sera le prochain président en 2050 ?",
    "Quel est le numéro de téléphone du musée ?",
    "Où se trouve la clé de la porte arrière ?",
    "Que s'est-il passé dans cette réunion privée ?",
]


def main():
    items = [
        json.loads(line)
        for line in KB_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = []

    for i, item in enumerate(items):
        question = item["question"]
        before = lambda n: items[(i - n) % len(items)]["text"]
        after = lambda n: items[(i + n) % len(items)]["text"]

        # 8 variantes ancrées par fait. Le modèle doit voir la bonne réponse
        # bien plus souvent que le refus : sinon « Je ne sais pas » domine.
        contexts = [
            item["text"],
            f"{before(1)} {item['text']}",
            f"{item['text']} {after(1)}",
            f"{before(2)} {item['text']}",
            f"{item['text']} {after(2)}",
            f"{before(3)} {item['text']}",
            f"{item['text']} {after(3)}",
            f"{before(4)} {item['text']} {after(4)}",
        ]
        for context in contexts:
            examples.append(
                (
                    f"{CONTEXT_PREFIX}{context}{SEPARATOR}{QUESTION_PREFIX}{question}",
                    item["answer"],
                )
            )

    # Peu d'exemples de refus : uniquement pour des questions hors base.
    # Les anciennes versions utilisaient items[:5] (des questions couvertes),
    # ce qui apprenait au modèle à refuser des questions auxquelles il sait
    # répondre. Les questions hors base sont aussi gérées par is_supported()
    # dans rag.py à l'inférence.
    for question in OUT_OF_SCOPE_QUESTIONS[:5]:
        examples.append((f"{QUESTION_PREFIX}{question}", ABSTENTION))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, (prompt, response) in enumerate(examples):
            f.write(f"{prompt}\n{response}")
            if i < len(examples) - 1:
                f.write("\n\n")

    print(f"{len(examples)} exemples écrits dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
