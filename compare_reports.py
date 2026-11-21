"""Compare les benchmarks enregistrés dans ``reports/``.

Chaque rapport est produit par ``evaluate.py`` et porte le codename de la
release. Ce script affiche un tableau pour suivre les progrès d'une fleur à
l'autre sur le même jeu de test.

Usage :
    python compare_reports.py
"""

import json
from pathlib import Path

REPORT_DIR = Path("reports")
COLUMNS = [
    ("codename", "Codename"),
    ("version", "Version"),
    ("split", "Split"),
    ("commit", "Commit"),
    ("exact", "Exact"),
    ("supported", "Ancre"),
    ("retrieval", "Retrieval"),
    ("far", "Hors loin"),
    ("near", "Hors proche"),
]


def load_rows():
    """Lit les rapports JSON et en extrait les métriques principales."""
    rows = []
    for path in sorted(REPORT_DIR.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        release = report.get("release", {})
        metrics = report.get("metrics", {})
        inside = metrics.get("in_scope", {})
        far = metrics.get("out_scope_far", {})
        near = metrics.get("out_scope_near", {})
        total = inside.get("count", 0)

        rows.append(
            {
                "codename": release.get("codename", "?"),
                "version": release.get("version", "?"),
                "split": report.get("split", "?"),
                "commit": report.get("git_commit") or "-",
                "exact": f"{inside.get('exact', 0)}/{total}",
                "supported": f"{inside.get('supported', 0)}/{total}",
                "retrieval": f"{inside.get('retrieval_hits', 0)}/{total}",
                "far": f"{far.get('correct_abstentions', 0)}/{far.get('count', 0)}",
                "near": f"{near.get('correct_abstentions', 0)}/{near.get('count', 0)}",
            }
        )
    return rows


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit(
            f"Aucun rapport dans {REPORT_DIR}. Lancez d'abord `python evaluate.py --split test`."
        )

    widths = {
        key: max([len(label)] + [len(str(row[key])) for row in rows])
        for key, label in COLUMNS
    }
    header = "  ".join(label.ljust(widths[key]) for key, label in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key, _ in COLUMNS))


if __name__ == "__main__":
    main()
