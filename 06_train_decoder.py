import os
import json
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from utils import seed_all, save_json

TRAIN_PATH = os.environ.get("TRAIN_PATH", "data/decoder_train.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "out/decoder")
MODEL = os.environ.get("MODEL", "google/flan-t5-small")
MAX_IN = int(os.environ.get("MAX_IN", "512"))
MAX_OUT = int(os.environ.get("MAX_OUT", "96"))

CODE_MODE = os.environ.get("CODE_MODE", "special")  # special | text | none
K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
ADD_SPECIAL_TOKENS = os.environ.get("ADD_SPECIAL_TOKENS", "1") == "1"

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
LR = float(os.environ.get("LR", "2e-5"))
TRAIN_BS = int(os.environ.get("TRAIN_BS", "8"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "50"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "-1"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def code_token(k: int, v: int, width: int) -> str:
    return f"<c{k}_{v:0{width}d}>"


def codes_to_special(codes: List[int], width: int) -> str:
    return " ".join([code_token(i, c, width) for i, c in enumerate(codes)])


def codes_to_text(codes: List[int]) -> str:
    return " ".join([f"c{i}={c}" for i, c in enumerate(codes)])


def parse_codes_str(code_str: str) -> List[int]:
    codes = []
    for part in code_str.strip().split():
        if "=" not in part:
            continue
        try:
            codes.append(int(part.split("=")[1]))
        except ValueError:
            continue
    return codes


def build_code_tokens(k: int, v: int) -> List[str]:
    width = len(str(v - 1))
    tokens = []
    for i in range(k):
        for j in range(v):
            tokens.append(code_token(i, j, width))
    return tokens


def format_prompt(ctx_text: str, codes: List[int]) -> str:
    if CODE_MODE == "none":
        return f"CTX: {ctx_text}\nWrite exactly one next sentence:"
    if CODE_MODE == "text":
        code_str = codes_to_text(codes)
        return f"CTX: {ctx_text}\nCODES: {code_str}\nWrite exactly one next sentence:"
    if CODE_MODE == "special":
        width = len(str(V - 1))
        code_str = codes_to_special(codes, width)
        return f"CTX: {ctx_text}\nCODES: {code_str}\nWrite exactly one next sentence:"
    raise ValueError(f"Unknown CODE_MODE: {CODE_MODE}")


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)

    rows = []
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ctx_text = r["ctx_text"]
            tgt_text = r["tgt_text"]
            codes = r.get("tgt_codes")
            if codes is None:
                codes = parse_codes_str(r.get("tgt_codes_str", ""))
            inp = format_prompt(ctx_text, codes)
            rows.append({"input": inp, "target": tgt_text})
            if MAX_ROWS > 0 and len(rows) >= MAX_ROWS:
                break

    ds = Dataset.from_list(rows)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    tokens_added = 0
    if CODE_MODE == "special" and ADD_SPECIAL_TOKENS:
        code_tokens = build_code_tokens(K, V)
        tokens_added = tok.add_special_tokens({"additional_special_tokens": code_tokens})
        if tokens_added > 0:
            model.resize_token_embeddings(len(tok))

    def tok_fn(batch):
        x = tok(batch["input"], truncation=True, max_length=MAX_IN)
        y = tok(batch["target"], truncation=True, max_length=MAX_OUT)
        x["labels"] = y["input_ids"]
        return x

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=TRAIN_BS,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        report_to=[],
        fp16=DEVICE == "cuda",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    meta = {
        "train_path": TRAIN_PATH,
        "out_dir": OUT_DIR,
        "model": MODEL,
        "code_mode": CODE_MODE,
        "K": K,
        "V": V,
        "max_in": MAX_IN,
        "max_out": MAX_OUT,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "train_batch_size": TRAIN_BS,
        "tokens_added": tokens_added,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "device": DEVICE,
    }
    save_json(os.path.join(OUT_DIR, "config.json"), meta)

    print("Saved:", OUT_DIR)
    print("Meta:", os.path.join(OUT_DIR, "config.json"))


if __name__ == "__main__":
    main()
