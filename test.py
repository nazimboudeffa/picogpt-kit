"""Teste le modèle entraîné en générant une réponse à partir d'un prompt.

Usage :
    python test.py
    python test.py --prompt "Quelle est la capitale de la France ?"
    python test.py --greedy  # déconseillé : le modèle boucle
"""

import argparse

from config import MODEL_NAME
from model_utils import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MODEL_DIR,
    generate,
    load_model,
)
from rag import DEFAULT_MIN_SCORE, answer_with_rag


def main():
    parser = argparse.ArgumentParser(description=f"Génère une réponse avec {MODEL_NAME}")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="dossier du modèle entraîné")
    parser.add_argument("--prompt", default="What is the capital of France?\n", help="prompt à compléter")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="nombre maximum de tokens générés",
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=True,
        help="échantillonnage (activé par défaut)",
    )
    parser.add_argument(
        "--greedy",
        dest="do_sample",
        action="store_false",
        help="décodage glouton (déconseillé : risque de boucles)",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY
    )
    parser.add_argument(
        "--no-repeat-ngram-size", type=int, default=DEFAULT_NO_REPEAT_NGRAM_SIZE
    )
    parser.add_argument(
        "--rag", action="store_true", help="ancre la réponse dans knowledge.jsonl"
    )
    parser.add_argument("--rag-k", type=int, default=1, help="nombre de passages à considérer")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args()

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    tokenizer, model = load_model(args.model_dir)
    if args.rag:
        print(
            answer_with_rag(
                tokenizer,
                model,
                args.prompt,
                k=args.rag_k,
                min_score=args.min_score,
                **gen_kwargs,
            )
        )
    else:
        print(generate(tokenizer, model, args.prompt, **gen_kwargs))


if __name__ == "__main__":
    main()
