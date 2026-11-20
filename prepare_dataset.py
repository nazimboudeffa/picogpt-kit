"""Télécharge un sous-ensemble de tatsu-lab/alpaca et écrit `train.txt`.

Format attendu par train.py : deux lignes non vides par exemple, séparées
par une ligne vide. Les exemples dont la réponse s'étale sur plusieurs
lignes sont écartés (le parseur ne lit que les deux premières lignes).

Quelques réponses factuellement fausses du jeu Alpaca sont aussi filtrées.
"""

import re

from datasets import load_dataset

MAX_EXAMPLES = 500
OUTPUT_FILE = "train.txt"

# Réponses erronées connues dans les premiers exemples Alpaca.
BAD_PATTERNS = [
    r"boil at a temperature of 115",
    r"boils? at 115",
    r"mole is actually a small mammal that lives above ground",
    r"<nooutput>",  # remplacé par les exemples d'abstention de prepare_rag_dataset.py
]


def _is_valid(prompt, output):
    if not prompt or not output:
        return False
    if "\n" in prompt or "\n" in output:
        return False
    if len(output) > 1000:
        return False
    lowered = output.lower()
    return not any(re.search(pattern, lowered) for pattern in BAD_PATTERNS)


def main():
    print("Téléchargement de tatsu-lab/alpaca (train split)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    examples = []
    seen = set()

    for row in ds:
        instruction = row["instruction"].strip()
        inp = row.get("input", "").strip()
        output = row["output"].strip()

        if inp:
            prompt = f"{instruction} Input: {inp}"
        else:
            prompt = instruction

        if not _is_valid(prompt, output):
            continue

        key = (prompt.lower(), output.lower())
        if key in seen:
            continue
        seen.add(key)
        examples.append((prompt, output))

        if len(examples) >= MAX_EXAMPLES:
            break

    print(f"Écriture de {len(examples)} exemples valides dans {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, (prompt, response) in enumerate(examples):
            f.write(f"{prompt}\n{response}")
            if i < len(examples) - 1:
                f.write("\n\n")

    print("Terminé.")


if __name__ == "__main__":
    main()
