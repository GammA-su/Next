import os
import json
import pysbd
from datasets import load_dataset

from utils import seed_all, save_json

OUT = os.environ.get("SENT_PATH", "data/sentences.jsonl")
META = os.environ.get("SENT_META", "data/sentences.meta.json")
DATASET = os.environ.get("HF_DATASET", "wikitext")
CONFIG = os.environ.get("HF_CONFIG", "wikitext-2-raw-v1")
SPLIT = os.environ.get("HF_SPLIT", "train")
DOCS_JSONL = os.environ.get("DOCS_JSONL", "")
LANG = os.environ.get("LANG", "en")
MIN_LEN = int(os.environ.get("MIN_LEN", "15"))
MAX_LEN = int(os.environ.get("MAX_LEN", "300"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

segmenter = pysbd.Segmenter(language=LANG, clean=True)


def segment(text: str):
    sents = [s.strip() for s in segmenter.segment(text) if s.strip()]
    # basic filters (tune)
    sents = [s for s in sents if MIN_LEN <= len(s) <= MAX_LEN]
    return sents


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    sent_id = 0
    num_docs = 0

    if DOCS_JSONL:
        docs_seen = 0
        docs_invalid = 0
        docs_skipped = 0

        with open(DOCS_JSONL, "r", encoding="utf-8") as fin, open(
            OUT, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                docs_seen += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    docs_invalid += 1
                    continue

                text = rec.get("text", "")
                if not text or not text.strip():
                    docs_skipped += 1
                    continue

                sents = segment(text)
                if not sents:
                    docs_skipped += 1
                    continue

                fallback_id = docs_seen - 1
                doc_val = rec.get("doc_id", fallback_id)
                try:
                    doc_val = int(doc_val)
                except (TypeError, ValueError):
                    doc_val = fallback_id

                extra = {}
                if "title" in rec:
                    extra["title"] = rec.get("title")
                if "url" in rec:
                    extra["url"] = rec.get("url")

                for s in sents:
                    out_rec = {"doc_id": doc_val, "sent_id": sent_id, "text": s}
                    if extra:
                        out_rec.update(extra)
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    sent_id += 1
                num_docs += 1

        meta = {
            "mode": "docs_jsonl",
            "docs_jsonl": DOCS_JSONL,
            "lang": LANG,
            "min_len": MIN_LEN,
            "max_len": MAX_LEN,
            "sent_path": OUT,
            "num_docs": num_docs,
            "num_sents": sent_id,
            "docs_seen": docs_seen,
            "docs_invalid": docs_invalid,
            "docs_skipped": docs_skipped,
            "seed": SEED,
            "deterministic": DETERMINISTIC,
        }
        save_json(META, meta)

        print("Wrote:", OUT)
        print("Meta:", META)
        return

    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    doc_id = 0
    buf = []

    def flush():
        nonlocal doc_id, sent_id, buf, num_docs
        if not buf:
            return
        text = " ".join(buf)
        sents = segment(text)
        if sents:
            for s in sents:
                rec = {"doc_id": doc_id, "sent_id": sent_id, "text": s}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sent_id += 1
            doc_id += 1
            num_docs += 1
        buf = []

    with open(OUT, "w", encoding="utf-8") as f:
        for ex in ds:
            t = ex.get("text", "")
            if not t.strip():
                flush()
                continue
            # wikitext has short lines; we buffer them into a doc until blank line
            buf.append(t.strip())
        flush()

    meta = {
        "mode": "hf",
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "lang": LANG,
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "sent_path": OUT,
        "num_docs": num_docs,
        "num_sents": sent_id,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }
    save_json(META, meta)

    print("Wrote:", OUT)
    print("Meta:", META)


if __name__ == "__main__":
    main()
