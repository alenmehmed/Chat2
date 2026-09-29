import re
import zlib

from datasets import load_dataset
from torch.utils.data import Dataset

from global_info.global_info import CHAT_DATA_CONFIG, CHAT_DIALOGUE_CACHE_PATH, TOKENIZER_PATH
from tokenizer.built_tokenizer_utils import load_tokenizer
from training.utils.dialogue import extract_turns, render_dialogue

# Alpaca-cleaned is GPT-3.5 output and 1,249 of its rows answer as the assistant. Remove for conciseness.
_AI_DISCLAIMER = re.compile(
    r"\b(as an ai|an ai (language )?model|a language model|i am an ai|i'm an ai|"
    r"as a digital assistant|as an artificial intelligence|openai)\b",
    re.IGNORECASE,
)

def _load_dialogues(CHAT_DATA_CONFIG):
    """Rendered multi-turn dialogues"""
    n = CHAT_DATA_CONFIG["dialogue_samples"]
    if n <= 0:
        return []

    if CHAT_DIALOGUE_CACHE_PATH.exists():
        cached = CHAT_DIALOGUE_CACHE_PATH.read_text(encoding="utf-8").splitlines()
        if len(cached) >= n:
            return cached[:n]

    dataset = load_dataset(CHAT_DATA_CONFIG["dialogue_dataset"], split=CHAT_DATA_CONFIG["dialogue_split"], streaming=True)
    rendered = []
    for row in dataset:
        line = render_dialogue(
            extract_turns(row),
            min_turns=CHAT_DATA_CONFIG["min_dialogue_turns"],
            max_turns=CHAT_DATA_CONFIG["max_dialogue_turns"],
        )
        if line:
            rendered.append(line)
        if len(rendered) >= n:
            break

    CHAT_DIALOGUE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAT_DIALOGUE_CACHE_PATH.write_text("\n".join(rendered), encoding="utf-8")
    return rendered

def _load_instruct(CHAT_DATA_CONFIG):
    """Single-turn instruction/response rows to sit alongside Dolly."""
    n = CHAT_DATA_CONFIG["instruct_samples"]
    if n <= 0:
        return []

    dataset = load_dataset(CHAT_DATA_CONFIG["instruct_dataset"], split=CHAT_DATA_CONFIG["instruct_split"])
    return dataset.select(range(min(n, len(dataset))))


def _keep_rows(responses, max_tokens, tokenizer, contexts=None):
    """Row indices which read like a conversational reply"""
    responses = [r.strip() for r in responses]
    contexts = list(contexts) if contexts is not None else None
    lengths = [len(e.ids) for e in tokenizer.encode_batch(responses, add_special_tokens=False)]

    keep = []
    for i, (response, n) in enumerate(zip(responses, lengths)):
        if not response or n > max_tokens or _AI_DISCLAIMER.search(response):
            continue
        if contexts is not None and contexts[i].strip():
            continue
        keep.append(i)
    return keep

def _in_val(source, row):
    """Held-out membership from a hash of (source, row) alone."""
    h = zlib.crc32(f"{CHAT_DATA_CONFIG['split_seed']}:{source}:{row}".encode())

    return h % 10_000 < CHAT_DATA_CONFIG["val_fraction"] * 10_000


class ChatDataset(Dataset):
    """Answer-shaped instruction data (Dolly + Alpaca) plus multi-turn SODA dialogues, all rendered as USR/BOT turns."""
    SRC_DOLLY, SRC_DIALOGUE, SRC_INSTRUCT = 0, 1, 2
    SOURCE_NAMES = {SRC_DOLLY: "dolly", SRC_DIALOGUE: "soda", SRC_INSTRUCT: "alpaca"}

    def __init__(self, split="train"):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.dolly = load_dataset(CHAT_DATA_CONFIG["dolly_dataset"], split=CHAT_DATA_CONFIG["dolly_split"])
        self.dialogues = _load_dialogues(CHAT_DATA_CONFIG)
        self.instruct = _load_instruct(CHAT_DATA_CONFIG)

        tokenizer = load_tokenizer(TOKENIZER_PATH)
        cap = CHAT_DATA_CONFIG["max_response_tokens"]

        dolly_rows = _keep_rows(self.dolly["response"], cap, tokenizer, contexts=self.dolly["context"])
        instruct_rows = (_keep_rows(self.instruct["output"], cap, tokenizer, contexts=self.instruct["input"])
            if len(self.instruct)
            else []
        )

        # flat index so DataLoader(shuffle=True) interleaves the sources
        base = (
            [(self.SRC_DOLLY, i) for i in dolly_rows]
            + [(self.SRC_DIALOGUE, j) for j in range(len(self.dialogues))]
            + [(self.SRC_INSTRUCT, k) for k in instruct_rows]
        )

        keep = [entry for entry in base if _in_val(*entry) == (split == "val")]
        if split == "val":
            self._index = keep
            return

        # Oversample Dolly only after the holdout is complete
        repeats = max(1, CHAT_DATA_CONFIG.get("dolly_repeat", 1)) - 1
        self._index = keep + [e for e in keep if e[0] == self.SRC_DOLLY] * repeats

    def __len__(self):
        return len(self._index)

    def source_of(self, idx):
        return self._index[idx][0]

    def __getitem__(self, idx):
        source, i = self._index[idx]

        if source == self.SRC_DIALOGUE:
            return {"text": self.dialogues[i]}

        if source == self.SRC_INSTRUCT:
            row = self.instruct[i]
            prompt, context, response = row["instruction"], row["input"], row["output"]
        else:
            row = self.dolly[i]
            prompt, context, response = row["instruction"], row["context"], row["response"]

        prompt, context, response = prompt.strip(), context.strip(), response.strip()
        if context:
            prompt = f"{prompt}\n{context}"

        return {"text": f"<USR> {prompt} <BOT> {response}"}
    