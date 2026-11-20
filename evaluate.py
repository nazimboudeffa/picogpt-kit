"""Évalue le comportement ancré : réponses exactes, ancrage et abstention.

Usage :
    python evaluate.py
    python evaluate.py --model-dir PicoGPT1-Daisy
"""

import argparse
import json
from pathlib import Path

from model_utils import load_model
from prepare_rag_dataset import OUT_OF_SCOPE_QUESTIONS
from rag import (
    ABSTENTION,
    answer_with_rag,
    get_kb,
    is_supported,
    normalize,
)

KB_PATH = Path("knowledge.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG")
    parser.add_argument("--model-dir", default="PicoGPT1-Daisy")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in KB_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer, model = load_model(args.model_dir)
    kb = get_kb()

    exact = 0
    supported = 0
    abstained_in = 0
    for item in items:
        answer = answer_with_rag(
            tokenizer,
            model,
            item["question"],
            max_new_tokens=args.max_new_tokens,
        )
        if answer == ABSTENTION:
            abstained_in += 1
        elif is_supported(answer, item["text"]):
            supported += 1
        if normalize(answer) == normalize(item["answer"]):
            exact += 1

    abstained_out = 0
    for question in OUT_OF_SCOPE_QUESTIONS:
        answer = answer_with_rag(
            tokenizer,
            model,
            question,
            max_new_tokens=args.max_new_tokens,
        )
        if answer == ABSTENTION:
            abstained_out += 1

    retrieval_hits = sum(
        kb.retrieve(item["question"])[0][0]["id"] == item["id"] for item in items
    )

    total_in = len(items)
    print(f"Questions dans la base      : {total_in}")
    print(f"  réponses exactes          : {exact}/{total_in}")
    print(f"  réponses ancrées (support): {supported}/{total_in}")
    print(f"  abstentions à tort        : {abstained_in}/{total_in}")
    print(f"  retrieval correct         : {retrieval_hits}/{total_in}")
    print("Questions hors base         :", len(OUT_OF_SCOPE_QUESTIONS))
    print(f"  abstentions correctes     : {abstained_out}/{len(OUT_OF_SCOPE_QUESTIONS)}")


if __name__ == "__main__":
    main()
