"""Évalue le pipeline RAG : exactitude, ancrage et abstention.

Deux jeux sont disponibles :
- ``train`` (défaut) : les questions de knowledge.jsonl et les hors-base connus ;
- ``test`` : eval/test.jsonl, jeu figé jamais utilisé à l'entraînement.

Un rapport JSON est écrit dans ``reports/<codename>-<split>.json``.

Usage :
    python evaluate.py
    python evaluate.py --split test
    python eval/check_leakage.py
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from config import CODENAME, MODEL_DIR, RELEASE_TAG, VERSION
from model_utils import load_model
from prepare_rag_dataset import OUT_OF_SCOPE_QUESTIONS
from rag import (
    ABSTENTION,
    DEFAULT_MIN_SCORE,
    answer_with_rag,
    get_kb,
    is_supported,
    normalize,
)

KB_PATH = Path("knowledge.jsonl")
TEST_PATH = Path("eval/test.jsonl")
REPORT_DIR = Path("reports")


def load_jsonl(path):
    """Lit un jeu de questions au format JSONL."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path):
    """Empreinte du jeu de données, consignée dans le rapport."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    """Commit court sur lequel le rapport a été produit, si Git est disponible."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_items(split):
    """Construit la liste des questions à évaluer et renvoie le jeu utilisé."""
    if split == "test":
        return load_jsonl(TEST_PATH), TEST_PATH

    items = []
    for fact in load_jsonl(KB_PATH):
        items.append(
            {
                "id": fact["id"],
                "fact_id": fact["id"],
                "question": fact["question"],
                "expected": fact["answer"],
                "scope": "in",
            }
        )
    for index, question in enumerate(OUT_OF_SCOPE_QUESTIONS):
        items.append(
            {
                "id": f"out{index + 1:02d}",
                "question": question,
                "expected": ABSTENTION,
                "scope": "out",
                "kind": "far",
            }
        )
    return items, KB_PATH


def evaluate_items(tokenizer, model, items, args):
    """Génère une réponse par question et calcule les indicateurs associés."""
    kb = get_kb()
    results = []

    for item in items:
        answer = answer_with_rag(
            tokenizer,
            model,
            item["question"],
            k=args.rag_k,
            min_score=args.min_score,
            max_new_tokens=args.max_new_tokens,
        )
        matches = kb.retrieve(item["question"], args.rag_k)
        context = None
        if item["scope"] == "in":
            _, context = kb.build_prompt(
                item["question"], k=args.rag_k, min_score=args.min_score
            )

        results.append(
            {
                **item,
                "answer": answer,
                "abstained": answer == ABSTENTION,
                "exact": normalize(answer) == normalize(item["expected"]),
                "contains": normalize(item["expected"]) in normalize(answer),
                "supported": is_supported(answer, context) if item["scope"] == "in" else None,
                "retrieved_id": matches[0][0]["id"] if matches else None,
                "retrieved_score": round(matches[0][1], 4) if matches else None,
            }
        )

    return results


def summarize(results):
    """Agrège les résultats par portée (dans la base, hors base, hors base proche)."""
    in_scope = [r for r in results if r["scope"] == "in"]
    out_scope = [r for r in results if r["scope"] == "out"]
    near = [r for r in out_scope if r.get("kind") == "near"]
    far = [r for r in out_scope if r.get("kind") != "near"]

    def count(rows, key):
        return sum(bool(row[key]) for row in rows)

    def out_stats(rows):
        return {
            "count": len(rows),
            "correct_abstentions": count(rows, "abstained"),
            "hallucinations": len(rows) - count(rows, "abstained"),
        }

    return {
        "in_scope": {
            "count": len(in_scope),
            "exact": count(in_scope, "exact"),
            "contains": count(in_scope, "contains"),
            "supported": count(in_scope, "supported"),
            "false_abstentions": count(in_scope, "abstained"),
            "retrieval_hits": sum(
                r["retrieved_id"] == r.get("fact_id") for r in in_scope
            ),
        },
        "out_scope": out_stats(out_scope),
        "out_scope_far": out_stats(far),
        "out_scope_near": out_stats(near),
    }


def print_results(metrics):
    """Affiche un résumé lisible des métriques."""
    inside = metrics["in_scope"]
    total = inside["count"]
    print(f"Questions dans la base      : {total}")
    print(f"  réponses exactes          : {inside['exact']}/{total}")
    print(f"  réponses contenues        : {inside['contains']}/{total}")
    print(f"  réponses ancrées (support): {inside['supported']}/{total}")
    print(f"  abstentions à tort        : {inside['false_abstentions']}/{total}")
    print(f"  retrieval correct         : {inside['retrieval_hits']}/{total}")

    for label, key in (
        ("Questions hors base", "out_scope"),
        ("  dont éloignées", "out_scope_far"),
        ("  dont proches de la base", "out_scope_near"),
    ):
        group = metrics[key]
        if not group["count"]:
            continue
        print(f"{label} : {group['count']}")
        print(f"  abstentions correctes     : {group['correct_abstentions']}/{group['count']}")
        print(f"  hallucinations            : {group['hallucinations']}/{group['count']}")


def main():
    parser = argparse.ArgumentParser(description="Évalue le pipeline RAG et écrit un rapport JSON")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--rag-k", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--report", type=Path, default=None,
                        help=f"défaut : reports/{CODENAME.lower()}-<split>.json")
    parser.add_argument("--no-report", action="store_true", help="n'écrit pas de rapport")
    args = parser.parse_args()

    items, dataset_path = build_items(args.split)
    tokenizer, model = load_model(args.model_dir)
    results = evaluate_items(tokenizer, model, items, args)
    metrics = summarize(results)

    report = {
        "release": {"codename": CODENAME, "version": VERSION, "tag": RELEASE_TAG},
        "model_dir": str(args.model_dir),
        "git_commit": git_commit(),
        "split": args.split,
        "dataset": {"path": str(dataset_path), "sha256": sha256(dataset_path)},
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "rag_k": args.rag_k,
            "min_score": args.min_score,
        },
        "metrics": metrics,
        "details": results,
    }

    print(f"Release : {CODENAME} v{VERSION} ({RELEASE_TAG})")
    print(f"Jeu     : {dataset_path} [{args.split}]")
    print(f"Commit  : {report['git_commit'] or 'inconnu'}")
    print_results(metrics)

    if not args.no_report:
        report_path = args.report or REPORT_DIR / f"{CODENAME.lower()}-{args.split}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nRapport : {report_path}")


if __name__ == "__main__":
    main()
