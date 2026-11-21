"""Entraîne le mini GPT-2, ou ajuste un GPT-2 pré-entraîné.

Usage :
    python train.py                          # GPT-2 pré-entraîné (recommandé)
    python train.py --init scratch           # ancien comportement, depuis zéro
    python train.py --epochs 5 --lr 3e-5     # réglages personnalisés

Le modèle pré-entraîné est celui qui donne des réponses cohérentes dans le
GGUF / llama.app : avec 745 exemples, un modèle entraîné de zéro (même plus
grand) ne peut pas apprendre la langue correctement. Le mode ``scratch`` est
conservé pour la démonstration pédagogique.
"""

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

DATA_PATHS = [
    Path("train.txt"),
    Path("rag_train.txt"),
    Path("facts_train.txt"),
]
OUTPUT_DIR = Path("PicoGPT1-Daisy")

MAX_LEN = 256          # taille du contexte
MAX_EXAMPLES = 3000    # assez pour 2 000 exemples Alpaca + les exemples RAG
BATCH_SIZE = 4

# Réglages par défaut : fine-tuning court et faible taux d'apprentissage,
# ou entraînement depuis zéro plus long avec un taux plus élevé.
EPOCHS_FINETUNE = 3
EPOCHS_SCRATCH = 10
LR_FINETUNE = 5e-5
LR_SCRATCH = 5e-4

# Architecture utilisée uniquement avec ``--init scratch``.
N_LAYER = 4
N_EMBD = 256
N_HEAD = 4

PRETRAINED_MODEL = "gpt2"


def build_example(tokenizer, prompt, completion, max_len=MAX_LEN):
    """Tokenise un couple prompt/réponse et masque le prompt dans les labels."""
    text = f"{prompt}\n{completion}{tokenizer.eos_token}"
    tokens = tokenizer(text, truncation=True, max_length=max_len)
    input_ids = tokens["input_ids"]

    prefix_ids = tokenizer(f"{prompt}\n", truncation=True, max_length=max_len)["input_ids"]
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }


def load_datasets(tokenizer, paths=DATA_PATHS, max_examples=MAX_EXAMPLES, max_len=MAX_LEN):
    """Lit les fichiers de données : deux lignes par exemple, une ligne vide entre eux."""
    examples = []

    for path in paths:
        path = Path(path)
        if not path.is_file():
            print(f"{path} introuvable, ignoré.")
            continue
        content = path.read_text(encoding="utf-8")

        for block in content.split("\n\n"):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue

            examples.append(build_example(tokenizer, lines[0], lines[1], max_len))
            if len(examples) >= max_examples:
                return Dataset.from_list(examples)

    return Dataset.from_list(examples)


def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne ou ajuste PicoGPT1-Daisy")
    parser.add_argument("--data-paths", nargs="+", type=Path, default=DATA_PATHS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES)
    parser.add_argument("--epochs", type=float, default=None,
                        help="défaut : 3 en fine-tuning, 10 depuis zéro")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None,
                        help="défaut : 5e-5 en fine-tuning, 5e-4 depuis zéro")
    parser.add_argument("--init", choices=("pretrained", "scratch"),
                        default="pretrained",
                        help="partir de gpt2 (recommandé) ou de poids aléatoires")
    parser.add_argument("--pretrained-model", default=PRETRAINED_MODEL,
                        help="identifiant Hugging Face du modèle de départ")
    parser.add_argument("--n-layer", type=int, default=N_LAYER,
                        help="couches, uniquement avec --init scratch")
    parser.add_argument("--n-embd", type=int, default=N_EMBD,
                        help="dimensions, uniquement avec --init scratch")
    parser.add_argument("--n-head", type=int, default=N_HEAD,
                        help="têtes, uniquement avec --init scratch")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="entraînement en demi-précision (uniquement si CUDA est disponible)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    epochs = args.epochs if args.epochs is not None else (
        EPOCHS_FINETUNE if args.init == "pretrained" else EPOCHS_SCRATCH
    )
    lr = args.lr if args.lr is not None else (
        LR_FINETUNE if args.init == "pretrained" else LR_SCRATCH
    )

    # Le tokenizer local est prioritaire : même vocabulaire que les poids
    # existants, et aucune requête réseau (les erreurs TLS sont fréquentes).
    tokenizer_source = (
        args.output_dir if (args.output_dir / "tokenizer.json").is_file()
        else (args.pretrained_model if args.init == "pretrained" else "gpt2")
    )
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_source))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_len
    # Format appris pendant l'entraînement : "prompt\nreponse<|endoftext|>"
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% if not loop.last %}{{ '\\n' }}{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '\\n' }}{% endif %}"
    )
    print(f"Vocab size : {len(tokenizer)}")
    print(f"PAD token  : {tokenizer.pad_token!r}")

    dataset = load_datasets(
        tokenizer, args.data_paths, args.max_examples, args.max_len
    )
    sources = ", ".join(str(path) for path in args.data_paths)
    print(f"{len(dataset)} exemples chargés depuis {sources}")
    print(f"Initialisation : {args.init} | epochs : {epochs} | lr : {lr}")

    if args.init == "pretrained":
        print(f"Chargement des poids pré-entraînés {args.pretrained_model!r}...")
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
        # On garde les 1 024 positions du modèle d'origine : les exemples sont
        # tronqués à max_len, donc aucun embedding au-delà n'est utilisé.
        print(
            f"Architecture héritée : {model.config.n_layer} couches, "
            f"{model.config.n_embd} dims, {model.config.n_head} têtes"
        )
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=args.max_len,
            n_ctx=args.max_len,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
        )
        model = GPT2LMHeadModel(config)
        print(
            f"Architecture : {args.n_layer} couches, {args.n_embd} dims, "
            f"{args.n_head} têtes (poids aléatoires)"
        )

    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres totaux : {total_params:,}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=50,
        save_steps=args.save_steps,
        save_total_limit=1,
        fp16=args.fp16,
        report_to="none",
        dataloader_num_workers=0,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Modèle sauvegardé dans {args.output_dir}")
    print("Pensez à régénérer le GGUF : python export_gguf.py")


if __name__ == "__main__":
    main()
