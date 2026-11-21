"""Vérifie que le jeu de test figé n'a pas fuité dans les données d'entraînement.

Le contrôle porte sur les prompts des trois fichiers générés par les scripts
``prepare_*``. Une question de test jugée trop proche d'un prompt appris est
signalée : le score obtenu ne mesurerait alors pas la généralisation.

Usage :
    python eval/check_leakage.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import normalize

TEST_PATH = ROOT / "eval/test.jsonl"
TRAIN_PATHS = [
    ROOT / "train.txt",
    ROOT / "rag_train.txt",
    ROOT / "facts_train.txt",
    ROOT / "negative_train.txt",
    ROOT / "paraphrase_train.txt",
]


def load_test():
    """Charge les questions du jeu de test."""
    return [
        json.loads(line)
        for line in TEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_training_prompts():
    """Renvoie (prompts normalisés, fichiers absents)."""
    prompts = []
    missing = []
    for path in TRAIN_PATHS:
        if not path.is_file():
            missing.append(str(path))
            continue
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if lines:
                prompts.append((str(path), normalize(lines[0])))
    return prompts, missing


def sha256(path):
    """Empreinte du jeu de test, à consigner dans le rapport."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    questions = load_test()
    prompts, missing = load_training_prompts()

    collisions = []
    for item in questions:
        question = normalize(item["question"])
        for source, prompt in prompts:
            if question == prompt or question in prompt:
                collisions.append((item["id"], source, prompt))
                break

    print(f"Jeu de test : {TEST_PATH} ({len(questions)} questions)")
    print(f"sha256      : {sha256(TEST_PATH)}")
    print(f"Prompts d'entraînement analysés : {len(prompts)}")
    if missing:
        print("Fichiers absents : " + ", ".join(missing))

    if collisions:
        print(f"\nFUITE DETECTEE : {len(collisions)} question(s) du test presente(s) dans l'entrainement.")
        for item_id, source, prompt in collisions:
            print(f"  {item_id} <- {source} : {prompt[:100]}")
        raise SystemExit(1)

    print("\nAucune fuite detectee : le jeu de test est bien separe de l'entrainement.")


if __name__ == "__main__":
    main()
