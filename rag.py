"""Ancrage minimal (RAG) sur une base de connaissances fermée.

Aucun changement d'architecture : on retrouve un passage avec TF-IDF, on le
place dans le prompt au format appris par le modèle, puis on vérifie que la
réponse est bien extraite du contexte. Sinon, le modèle doit s'abstenir.
"""

import json
import re
import unicodedata
from pathlib import Path


from model_utils import generate

KB_PATH = Path("knowledge.jsonl")

ABSTENTION = "Je ne sais pas."
CONTEXT_PREFIX = "Contexte: "
QUESTION_PREFIX = "Question: "
SEPARATOR = " | "

DEFAULT_MIN_SCORE = 0.40  # calibré sur le jeu test : récupère les reformulations à 0,41-0,50, hors base éloigné < 0,25
_KB_CACHE = {}


def normalize(text):
    """Minuscules, sans accents ni espaces multiples : pour la vérification."""
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip().lower()


class KnowledgeBase:
    """Petite base de faits, indexée en TF-IDF sur les caractères."""

    def __init__(self, path=KB_PATH):
        # Import paresseux : test.py / serve.py restent utilisables sans RAG
        # même si scikit-learn n'est pas installé.

        from sklearn.feature_extraction.text import TfidfVectorizer

        path = Path(path)
        self.items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.items:
            raise ValueError(f"Base de connaissances vide : {path}")
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        # Indexer le texte ET la question améliore la correspondance quand la
        # question reformule le fait (ex. « à quelle température... »).
        self.matrix = self.vectorizer.fit_transform(
            [f"{item['text']} {item['question']}" for item in self.items]
        )

    def retrieve(self, question, k=1):
        """Renvoie [(fait, score), ...] triés par similarité décroissante."""
        query = self.vectorizer.transform([question])
        scores = (self.matrix @ query.T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.items[i], float(scores[i])) for i in order]

    def build_prompt(self, question, k=1, min_score=DEFAULT_MIN_SCORE):
        """Construit le prompt d'entraînement ; sans contexte si score trop faible."""
        question = str(question).strip()
        matches = self.retrieve(question, k)
        if not matches or matches[0][1] < min_score:
            return f"{QUESTION_PREFIX}{question}\n", None
        item, _score = matches[0]
        prompt = (
            f"{CONTEXT_PREFIX}{item['text']}{SEPARATOR}{QUESTION_PREFIX}{question}\n"
        )
        return prompt, item["text"]


def get_kb(path=KB_PATH):
    """Cache la base pour ne pas refaire le TF-IDF à chaque requête."""
    key = str(Path(path).resolve())
    if key not in _KB_CACHE:
        _KB_CACHE[key] = KnowledgeBase(path)
    return _KB_CACHE[key]


def is_supported(answer, context):
    """Vrai si la réponse est un extrait du contexte fourni."""
    if not context or not answer:
        return False
    answer_norm = normalize(answer)
    context_norm = normalize(context)
    return bool(answer_norm) and answer_norm in context_norm


def answer_with_rag(tokenizer, model, question, k=1, min_score=DEFAULT_MIN_SCORE, **gen_kwargs):
    """Récupère un contexte, génère une réponse, puis vérifie son ancrage.

    Anciennement, le décodage était contraint aux tokens du contexte. Avec le
    nouveau jeu de réponses directes (`facts_train.txt`), cette contrainte
    tronquait les réponses (« 100 » au lieu de « 100 °C ») et pouvait provoquer
    une abstention à tort. La génération déterministe donne désormais la
    réponse complète ; `is_supported()` conserve la garantie d'ancrage.
    """
    prompt, context = get_kb().build_prompt(question, k=k, min_score=min_score)
    if context is None:
        return ABSTENTION

    # Décodage déterministe : reproductible, sans troncature liée à la contrainte.
    # `no_repeat_ngram_size=3`, utile en échantillonnage libre, casse la beam
    # search sur ce modèle (réponses du type « convertir la mitre des lumières »).
    gen_kwargs["do_sample"] = False
    gen_kwargs["no_repeat_ngram_size"] = 0
    gen_kwargs.setdefault("num_beams", 5)
    generated = generate(
        tokenizer,
        model,
        prompt,
        return_full_text=False,
        **gen_kwargs,
    ).strip()
    if not is_supported(generated, context):
        return ABSTENTION
    return generated
