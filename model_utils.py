"""Fonctions partagées pour charger le modèle et générer une réponse."""

from config import MODEL_DIR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

MAX_LEN = 256  # doit rester identique à MAX_LEN dans train.py

DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


class StopAtNewline(StoppingCriteria):
    """Arrête la génération dès qu'une nouvelle ligne apparaît dans la réponse.

    Le modèle a été entraîné avec ``prompt\\nréponse<|endoftext|>``. Un second
    saut de ligne annonce une nouvelle paire question/réponse inventée : on
    s'en sert comme critère d'arrêt, en plus de l'EOS.
    """

    def __init__(self, tokenizer, prompt_len):
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self._prompt_len:]
        if generated.numel() == 0:
            return False
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        # Si le prompt ne finit pas par `\n`, le modèle peut en générer un
        # avant de répondre : on ignore les sauts de ligne initiaux.
        return "\n" in text.lstrip()


def load_model(model_dir=MODEL_DIR):
    """Charge le tokenizer et le modèle, avec un message clair si les poids manquent."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    except OSError as exc:
        raise SystemExit(
            f"Poids introuvables dans {model_dir!r}. Lancez d'abord `python train.py`."
        ) from exc

    model.eval()
    # Le generation_config sauvegardé contient max_length=20, ce qui entre en conflit
    # avec max_new_tokens. On le neutralise pour garder un seul réglage.
    model.generation_config.max_length = None
    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    do_sample=True,
    temperature=DEFAULT_TEMPERATURE,
    top_p=DEFAULT_TOP_P,
    top_k=DEFAULT_TOP_K,
    repetition_penalty=DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
    return_full_text=True,
    prefix_allowed_tokens_fn=None,
    num_beams=1,
):
    """Génère une réponse.

    Par défaut, renvoie le texte complet (prompt + réponse). Avec
    ``return_full_text=False``, renvoie uniquement la réponse générée.

    L'échantillonnage est activé par défaut : le décodage glouton provoque des
    boucles sur ce petit modèle. La génération s'arrête à l'EOS ou au premier
    saut de ligne produit après la réponse.
    """
    # À l'entraînement, le `\n` entre prompt et réponse était toujours fourni
    # (et masqué dans les labels) : le modèle n'a donc pas appris à le générer.
    # On le garantit ici, sinon la réponse démarre au milieu de la ligne.
    if not prompt.endswith("\n"):
        prompt = prompt + "\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    )
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "stopping_criteria": StoppingCriteriaList(
            [StopAtNewline(tokenizer, prompt_len)]
        ),
    }
    if do_sample:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    if prefix_allowed_tokens_fn is not None:
        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

    output_ids = model.generate(**inputs, **gen_kwargs)
    if return_full_text:
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return tokenizer.decode(
        output_ids[0, prompt_len:], skip_special_tokens=True
    ).lstrip()
