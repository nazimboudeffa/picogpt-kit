"""Entraîne le mini GPT-2 à partir de zéro sur train.txt.

Usage :
    python train.py

Équivalent de train.ipynb, sans Jupyter. Modifie les constantes ci-dessous
pour changer la taille ou le nombre d'exemples.
"""

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

DATA_PATHS = [Path("train.txt"), Path("rag_train.txt")]
OUTPUT_DIR = Path("PicoGPT1-Daisy")

MAX_LEN = 256          # taille du contexte
MAX_EXAMPLES = 2000    # nombre maximum d'exemples, tous fichiers confondus
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 5e-4


def build_example(tokenizer, prompt, completion):
    """Tokenise un couple prompt/réponse et masque le prompt dans les labels."""
    text = f"{prompt}\n{completion}{tokenizer.eos_token}"
    tokens = tokenizer(text, truncation=True, max_length=MAX_LEN)
    input_ids = tokens["input_ids"]

    prefix_ids = tokenizer(f"{prompt}\n", truncation=True, max_length=MAX_LEN)["input_ids"]
    labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }


def load_datasets(tokenizer):
    """Lit les fichiers de données : deux lignes par exemple, une ligne vide entre eux."""
    examples = []

    for path in DATA_PATHS:
        if not path.is_file():
            print(f"{path} introuvable, ignoré.")
            continue
        content = path.read_text(encoding="utf-8")

        for block in content.split("\n\n"):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue

            examples.append(build_example(tokenizer, lines[0], lines[1]))
            if len(examples) >= MAX_EXAMPLES:
                return Dataset.from_list(examples)

    return Dataset.from_list(examples)


def main():
    # Le tokenizer est déjà sauvegardé avec le modèle : lire ce dossier évite
    # une requête réseau (et ses erreurs TLS) et garantit le même vocabulaire.
    tokenizer_source = (
        OUTPUT_DIR if (OUTPUT_DIR / "tokenizer.json").is_file() else "gpt2"
    )
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_source))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_LEN
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

    dataset = load_datasets(tokenizer)
    sources = ", ".join(str(path) for path in DATA_PATHS)
    print(f"{len(dataset)} exemples chargés depuis {sources}")

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=MAX_LEN,
        n_ctx=MAX_LEN,
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres totaux : {total_params:,}")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=1,
        fp16=False,
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
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Modèle sauvegardé dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
