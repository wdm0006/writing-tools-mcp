#!/usr/bin/env python3
"""Build the committed RAID/Wikipedia benchmark corpus from RAID-train.

This is a one-time provenance script, not part of running the benchmark: the
corpus it produces is committed, so scoring it needs no download. It is kept in
the repository so the selection rule is auditable and re-runnable.

The source file is ~11.8 GB and is streamed, never held in memory. Pass
``--source-csv`` to read an already-downloaded copy instead.

Usage:
    uv run benchmarks/extract_raid_corpus.py --out benchmarks/corpora/raid_wiki_v1
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: RAID-train pinned to an explicit dataset revision rather than a moving branch.
RAID_REVISION = "865cac74188466cb0c3b7574a10204007b57a459"
RAID_TRAIN_URL = f"https://huggingface.co/datasets/liamdugan/raid/resolve/{RAID_REVISION}/train.csv"

#: Only unmodified Wikipedia-domain rows are eligible. RAID's adversarial
#: attacks (homoglyph substitution, zero-width spaces, paraphrase, ...) are a
#: deliberately harder, separate question and are out of scope here.
DOMAIN = "wiki"
ATTACK = "none"

CSV_FIELDS = [
    "id",
    "adv_source_id",
    "source_id",
    "model",
    "decoding",
    "repetition_penalty",
    "attack",
    "domain",
    "title",
    "prompt",
    "generation",
]


def normalize_text(raw: str) -> str:
    """Line-ending normalization only - the analyzers see the source prose."""
    return raw.replace("\r\n", "\n").replace("\r", "\n").strip()


def eligible(text: str, min_words: int, max_words: int) -> bool:
    return min_words <= len(text.split()) <= max_words


def iter_rows(handle):
    # RAID generations routinely exceed csv's default 128 KiB field cap, which
    # aborts the parse mid-file with "field larger than field limit".
    csv.field_size_limit(sys.maxsize)
    reader = csv.DictReader(handle)
    for row in reader:
        if row.get("domain") == DOMAIN and row.get("attack") == ATTACK:
            yield row


def _open_source(source_csv, url):
    if source_csv is not None:
        return open(source_csv, encoding="utf-8", newline="")

    import requests  # imported lazily: the local-file path needs no HTTP client

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    raw = response.raw

    class _Reader(io.RawIOBase):
        def readable(self):
            return True

        def readinto(self, buffer):
            chunk = raw.read(len(buffer))
            if not chunk:
                return 0
            buffer[: len(chunk)] = chunk
            return len(chunk)

    return io.TextIOWrapper(io.BufferedReader(_Reader(), 1 << 22), encoding="utf-8", newline="")


def collect(handle, min_words: int, max_words: int):
    """Group eligible rows by source article."""
    humans = {}
    machines = {}
    seen_row_ids = set()
    for row in iter_rows(handle):
        if row["id"] in seen_row_ids:
            continue
        seen_row_ids.add(row["id"])
        text = normalize_text(row["generation"])
        if not eligible(text, min_words, max_words):
            continue
        row["_text"] = text
        if row["model"] == "human":
            # Ties on source_id resolve to the lexicographically smallest row id.
            existing = humans.get(row["source_id"])
            if existing is None or row["id"] < existing["id"]:
                humans[row["source_id"]] = row
        else:
            machines.setdefault(row["source_id"], {}).setdefault(row["model"], []).append(row)
    return humans, machines


def select(humans, machines, pairs: int):
    """Deterministically pick `pairs` topic-matched human/machine document pairs.

    Source articles are walked in ascending ``source_id`` order and assigned a
    generator model round-robin from the sorted model list, so every model
    contributes a near-equal share and no source article contributes more than
    one human and one machine document. A source article whose assigned model
    has no eligible generation is skipped rather than reassigned, which keeps
    the assignment a pure function of position.
    """
    models = sorted({model for by_model in machines.values() for model in by_model})
    if not models:
        raise ValueError("No machine-generated rows survived filtering")

    selected = []
    seen_texts = set()
    for index, source_id in enumerate(sorted(humans)):
        if len(selected) >= pairs:
            break
        model = models[index % len(models)]
        candidates = machines.get(source_id, {}).get(model)
        if not candidates:
            continue
        human_row = humans[source_id]
        machine_row = min(candidates, key=lambda row: row["id"])
        # Exact-duplicate text is rejected in both directions so no string can
        # appear under two labels, or twice under one.
        if human_row["_text"] in seen_texts or machine_row["_text"] in seen_texts:
            continue
        if human_row["_text"] == machine_row["_text"]:
            continue
        seen_texts.add(human_row["_text"])
        seen_texts.add(machine_row["_text"])
        selected.append((source_id, human_row, machine_row))
    return models, selected


def to_records(selected, prefix: str):
    records = []
    for pair_index, (source_id, human_row, machine_row) in enumerate(selected, start=1):
        for label, row in (("human", human_row), ("machine", machine_row)):
            records.append(
                {
                    "doc_id": f"{prefix}-{pair_index:04d}-{label}",
                    "label": label,
                    "text": row["_text"],
                    "source": {
                        "dataset": "liamdugan/raid",
                        "revision": RAID_REVISION,
                        "split": "train",
                        "raid_id": row["id"],
                        "raid_source_id": source_id,
                        "model": row["model"],
                        "decoding": row["decoding"],
                        "repetition_penalty": row["repetition_penalty"],
                        "attack": row["attack"],
                        "domain": row["domain"],
                        "title": row["title"],
                    },
                }
            )
    records.sort(key=lambda record: record["doc_id"])
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Corpus directory to write into")
    parser.add_argument("--source-csv", type=Path, default=None, help="Local RAID train.csv instead of downloading")
    parser.add_argument("--url", default=RAID_TRAIN_URL, help="Pinned RAID train.csv URL")
    parser.add_argument("--pairs", type=int, default=110, help="Human/machine document pairs to select")
    parser.add_argument("--min-words", type=int, default=120, help="Minimum whitespace-token count per document")
    parser.add_argument("--max-words", type=int, default=500, help="Maximum whitespace-token count per document")
    parser.add_argument("--prefix", default="raid-wiki", help="doc_id prefix")
    args = parser.parse_args(argv)

    with _open_source(args.source_csv, args.url) as handle:
        humans, machines = collect(handle, args.min_words, args.max_words)

    models, selected = select(humans, machines, args.pairs)
    records = to_records(selected, args.prefix)

    args.out.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    per_model = {}
    for record in records:
        if record["label"] == "machine":
            per_model[record["source"]["model"]] = per_model.get(record["source"]["model"], 0) + 1

    extraction = {
        "source": {
            "dataset": "liamdugan/raid",
            "split": "train",
            "revision": RAID_REVISION,
            "url": args.url,
        },
        "filters": {
            "domain": DOMAIN,
            "attack": ATTACK,
            "min_words": args.min_words,
            "max_words": args.max_words,
        },
        "selection": {
            "pairs_requested": args.pairs,
            "pairs_selected": len(selected),
            "models_available": models,
            "documents_per_generator_model": dict(sorted(per_model.items())),
            "eligible_human_source_articles": len(humans),
        },
        "counts": {
            "documents": len(records),
            "human": sum(1 for record in records if record["label"] == "human"),
            "machine": sum(1 for record in records if record["label"] == "machine"),
        },
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
    }
    (args.out / "extraction.json").write_text(json.dumps(extraction, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(extraction["counts"] | extraction["selection"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
