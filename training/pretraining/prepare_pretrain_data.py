import itertools
import json
import re
from typing import BinaryIO

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from global_info.global_info import (
    PRETRAIN_CORPUS_PATH,
    PRETRAIN_TOKENS_PATH,
    PRETRAIN_TOKENS_META_PATH,
    PRETRAIN_DATA_CONFIG,
    TOKENIZER_PATH,
)
from tokenizer.built_tokenizer_utils import load_tokenizer
from training.collator import clean_text
from training.utils.dialogue import extract_turns, render_dialogue

RE_HTML = re.compile(r'<[^<>]{1,64}>')
RE_WHITESPACE = re.compile(r'\s+')

TOKENIZER = load_tokenizer(TOKENIZER_PATH)


def _is_mostly_english(line: str) -> bool:
    # >=20 chars, ascii ratio >=0.85, backslash density <0.5% (drops LaTeX)
    if len(line) < PRETRAIN_DATA_CONFIG["min_line_chars"]:
        return False
    
    ascii_chars = sum(1 for c in line if ord(c) < 128)
    
    return (ascii_chars / len(line)) >= PRETRAIN_DATA_CONFIG["min_ascii_ratio"]


def _general_docs():
    buf = []
    with open(PRETRAIN_CORPUS_PATH, "r", encoding="utf-8") as src:
        for line in src:
            line = line.rstrip("\n")

            end = line.endswith("<EOS>")
            buf.append(line[:-len("<EOS>")] if end else line)

            if end:
                doc, buf = " ".join(buf), []

                if _is_mostly_english(doc): 
                    yield doc


def _dialogue_lines(row_counter : dict):
    """Multi-turn SODA dialogues rendered into the <USR>/<BOT> protocol"""

    dataset = load_dataset(PRETRAIN_DATA_CONFIG["dialogue_dataset"], split=PRETRAIN_DATA_CONFIG["dialogue_split"], streaming=True)

    for row in dataset:
        row_counter["rows"] += 1
        # rendered by the shared helper so pretraining and the fine-tune cannot drift into showing the model two different turn protocols
        line = render_dialogue(
            extract_turns(row),
            min_turns=PRETRAIN_DATA_CONFIG["min_dialogue_turns"],
            max_turns=PRETRAIN_DATA_CONFIG["max_dialogue_turns"],
        )

        if line:
            yield line

def _write_batch(lines: list, out: BinaryIO):
    cleaned = [clean_text(line, RE_HTML, RE_WHITESPACE) for line in lines]
    cleaned = [c for c in cleaned if c]

    if not cleaned:
        return 0
    
    written = 0

    for enc in TOKENIZER.encode_batch(cleaned, add_special_tokens=False):
        if not enc.ids:
            continue

        ids = enc.ids + [TOKENIZER.token_to_id("<EOS>")]

        np.array(ids, dtype=np.uint16).tofile(out)
        written += len(ids)

    return written

def prepare():
    if not PRETRAIN_CORPUS_PATH.exists():
        raise FileNotFoundError(f"Pretrain corpus not found at {PRETRAIN_CORPUS_PATH}.") 
    
    target_tokens = PRETRAIN_DATA_CONFIG["target_tokens"]
    batch_lines = PRETRAIN_DATA_CONFIG["batch_lines"]

    # Split the budget between general web text (language knowledge) and dialogue (turn-taking).
    dialogue_target = int(target_tokens * PRETRAIN_DATA_CONFIG["dialogue_fraction"])
    general_target = target_tokens - dialogue_target

    streams = {}

    row_counter = {"rows" : 0}
    streams["general"] = {"gen": _general_docs(), "target": general_target, "total": 0, "lines": 0, "done": False}
    streams["dialogue"] = {"gen": _dialogue_lines(row_counter), "target": dialogue_target, "total": 0, "lines": 0, "done": False}

    print(f"General Dialogue: {dialogue_target:,}")
    print(f"Token budget: {general_target:,}")

    with open(PRETRAIN_TOKENS_PATH, "wb") as out, tqdm(total=target_tokens, unit="tok", unit_scale=True, desc="Tokenizing pretrain corpus") as pbar:

        total_tokens = 0

        # If one runs out, the other keeps going until the overall target is met.
        while total_tokens < target_tokens and any(not s["done"] for s in streams.values()):
            alive = [k for k, s in streams.items() if not s["done"]]
            key = min(alive, key=lambda k: streams[k]["total"] / max(1, streams[k]["target"]))
            stream = streams[key]

            # Shrink the pull as a stream nears its quota
            n_lines = batch_lines

            if stream["lines"]:
                per_line = stream["total"] / stream["lines"]
                remaining = stream["target"] - stream["total"]
                n_lines = max(1, min(batch_lines, int(remaining / max(per_line, 1e-9)) + 1))

            lines = list(itertools.islice(stream["gen"], n_lines))
            if not lines:
                stream["done"] = True
                continue

            written = _write_batch(lines, out)

            stream["total"] += written
            stream["lines"] += len(lines)

            total_tokens += written
            pbar.update(total_tokens - pbar.n)

    meta = {
        "num_tokens": total_tokens,
        "dtype": "uint16",
        "vocab_size": TOKENIZER.get_vocab_size(),
        "general_tokens": streams.get("general", {}).get("total", 0),
        "dialogue_tokens": streams.get("dialogue", {}).get("total", 0),
        "dialogue_dataset": PRETRAIN_DATA_CONFIG["dialogue_dataset"] if dialogue_target > 0 else None,
        "dialogue_rows_consumed" : row_counter["rows"]
    }

    with open(PRETRAIN_TOKENS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {total_tokens:,} tokens to {PRETRAIN_TOKENS_PATH}")
    print(f"Meta saved to {PRETRAIN_TOKENS_META_PATH}")


if __name__ == "__main__":
    prepare()
