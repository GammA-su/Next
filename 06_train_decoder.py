import os
import json
import hashlib
from typing import List

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from utils import seed_all, save_json

TRAIN_PATH = os.environ.get("TRAIN_PATH", "data/decoder_train.jsonl")
VAL_PATH = os.environ.get("VAL_PATH", "data/decoder_val.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "out/decoder")
MODEL = os.environ.get("MODEL", "google/flan-t5-small")
MAX_IN = int(os.environ.get("MAX_IN", "512"))
MAX_OUT = int(os.environ.get("MAX_OUT", "96"))

CODE_MODE = os.environ.get("CODE_MODE", "special")  # special | text | none
K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
ADD_SPECIAL_TOKENS = os.environ.get("ADD_SPECIAL_TOKENS", "1") == "1"

NUM_PROC = int(os.environ.get("NUM_PROC", "0"))
CACHE_TOKENIZED = os.environ.get("CACHE_TOKENIZED", "1") == "1"
TOKENIZED_CACHE_DIR = os.environ.get(
    "TOKENIZED_CACHE_DIR", "out/cache/tokenized_decoder"
)
KEEP_IN_MEMORY = os.environ.get("KEEP_IN_MEMORY", "0") == "1"
PRETOKENIZE_ONLY = os.environ.get("PRETOKENIZE_ONLY", "0") == "1"
MAP_BATCH_SIZE = int(os.environ.get("MAP_BATCH_SIZE", "1000"))

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


def file_signature(path: str) -> dict:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def compute_cache_key(
    train_path: str,
    val_path: str,
    tokenizer_name: str,
    vocab_size: int,
    code_mode: str,
    k: int,
    v: int,
    add_special_tokens: bool,
    max_in: int,
    max_out: int,
    max_rows: int,
) -> str:
    payload = {
        "train": file_signature(train_path),
        "val": file_signature(val_path),
        "tokenizer_name": tokenizer_name,
        "vocab_size": vocab_size,
        "code_mode": code_mode,
        "k": k,
        "v": v,
        "add_special_tokens": add_special_tokens,
        "max_in": max_in,
        "max_out": max_out,
        "max_rows": max_rows,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_decoder_rows(path: str, max_rows: int) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ctx_text = r["ctx_text"]
            tgt_text = r["tgt_text"]
            codes = r.get("tgt_codes")
            if codes is None:
                codes = parse_codes_str(r.get("tgt_codes_str", ""))
            inp = format_prompt(ctx_text, codes)
            rows.append({"input": inp, "target": tgt_text})
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def tokenize_batch(batch, tokenizer, max_in: int, max_out: int) -> dict:
    x = tokenizer(
        batch["input"],
        padding="max_length",
        truncation=True,
        max_length=max_in,
    )
    y = tokenizer(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=max_out,
    )
    x["labels"] = y["input_ids"]
    return x


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)

    num_proc = NUM_PROC
    if os.name == "nt" and num_proc > 0:
        print("NUM_PROC>0 is not supported on Windows; falling back to 0.")
        num_proc = 0

    train_rows = load_decoder_rows(TRAIN_PATH, MAX_ROWS)
    val_rows = load_decoder_rows(VAL_PATH, -1)
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    tokens_added = 0
    if CODE_MODE == "special" and ADD_SPECIAL_TOKENS:
        code_tokens = build_code_tokens(K, V)
        tokens_added = tok.add_special_tokens({"additional_special_tokens": code_tokens})
        if tokens_added > 0:
            model.resize_token_embeddings(len(tok))

    cache_key = compute_cache_key(
        TRAIN_PATH,
        VAL_PATH,
        tok.name_or_path,
        len(tok),
        CODE_MODE,
        K,
        V,
        ADD_SPECIAL_TOKENS,
        MAX_IN,
        MAX_OUT,
        MAX_ROWS,
    )
    cache_root = os.path.join(TOKENIZED_CACHE_DIR, cache_key)
    train_cache_dir = os.path.join(cache_root, "train")
    val_cache_dir = os.path.join(cache_root, "val")

    print(
        "Tokenization setup: "
        f"train_rows={len(train_ds)}, val_rows={len(val_ds)}, "
        f"NUM_PROC={num_proc}, MAP_BATCH_SIZE={MAP_BATCH_SIZE}, "
        f"cache_key={cache_key}, cache_dir={cache_root}"
    )

    cache_hit = False
    if CACHE_TOKENIZED and os.path.isdir(train_cache_dir) and os.path.isdir(val_cache_dir):
        try:
            print(f"Tokenized cache hit: loading from {cache_root}")
            train_ds = load_from_disk(train_cache_dir, keep_in_memory=KEEP_IN_MEMORY)
            val_ds = load_from_disk(val_cache_dir, keep_in_memory=KEEP_IN_MEMORY)
            cache_hit = True
        except Exception as exc:
            print(f"Tokenized cache load failed ({exc}); rebuilding.")
            cache_hit = False

    if not cache_hit:
        cache_note = "and saving cache" if CACHE_TOKENIZED else "without caching"
        print(f"Tokenized cache miss: building {cache_note}")
        train_ds = train_ds.map(
            tokenize_batch,
            batched=True,
            batch_size=MAP_BATCH_SIZE,
            num_proc=num_proc if num_proc > 0 else None,
            remove_columns=train_ds.column_names,
            load_from_cache_file=False,
            keep_in_memory=KEEP_IN_MEMORY,
            fn_kwargs={"tokenizer": tok, "max_in": MAX_IN, "max_out": MAX_OUT},
        )
        val_ds = val_ds.map(
            tokenize_batch,
            batched=True,
            batch_size=MAP_BATCH_SIZE,
            num_proc=num_proc if num_proc > 0 else None,
            remove_columns=val_ds.column_names,
            load_from_cache_file=False,
            keep_in_memory=KEEP_IN_MEMORY,
            fn_kwargs={"tokenizer": tok, "max_in": MAX_IN, "max_out": MAX_OUT},
        )
        if CACHE_TOKENIZED:
            try:
                os.makedirs(train_cache_dir, exist_ok=True)
                os.makedirs(val_cache_dir, exist_ok=True)
                train_ds.save_to_disk(train_cache_dir)
                val_ds.save_to_disk(val_cache_dir)
                print(f"Tokenized cache saved to {cache_root}")
            except Exception as exc:
                print(f"Tokenized cache save failed ({exc}); continuing without cache.")

    needed_columns = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=needed_columns)
    val_ds.set_format(type="torch", columns=needed_columns)

    if PRETOKENIZE_ONLY:
        print("PRETOKENIZE_ONLY=1, exiting after tokenization.")
        return

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
        train_dataset=train_ds,
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
