import torch
import re
import unicodedata

from global_info.global_info import SPECIAL_TOKENS

# Ignore our own special tokens
_PROTECTED_TOKENS = set(SPECIAL_TOKENS.values())

def clean_text(text, html_re, ws_re):
    text = unicodedata.normalize("NFKC", text)

    text = html_re.sub(lambda m: m.group(0) if m.group(0) in _PROTECTED_TOKENS else ' ', text)
    text = ws_re.sub(' ', text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)  # remove control characters

    return text.strip()

def bot_content_mask(ids, usr_id, bot_id, eos_id):
    """True at every index holding a token the model must learn to produce."""
    mask = [False] * len(ids)
    in_bot = False

    for index, token in enumerate(ids):
        if token == bot_id:
            in_bot = True
            continue

        if token == usr_id:
            in_bot = False
            continue

        if not in_bot:
            continue

        mask[index] = True
        if token == eos_id:
            in_bot = False  # this bot turn is closed

    return mask


class Collator:
    def __init__(self, tokenizer, context_length):
        self.tokenizer = tokenizer
        self.context_length = context_length

        self.eos_id = tokenizer.token_to_id("<EOS>")
        self.bot_id = tokenizer.token_to_id("<BOT>")
        self.usr_id = tokenizer.token_to_id("<USR>")

        assert self.eos_id is not None, "<EOS> not found in tokenizer"
        assert self.bot_id is not None, "<BOT> not found in tokenizer"
        assert self.usr_id is not None, "<USR> not found in tokenizer"

        self._re_html = re.compile(r'<[^>]+>')
        self._re_ws   = re.compile(r'\s+')

    def __call__(self, batch_orig):
        # Filter dataset
        good = []
        for i, b in enumerate(batch_orig):
            txt = b.get("text") if isinstance(b, dict) else None
            if not (isinstance(txt, str) and len(txt.strip()) >= 5):
                # print(f"[SKIP] bad sample {i}: {repr(txt)[:60]}")
                continue
            good.append(txt)

        if not good:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        # Clean + tokenize
        texts = []
        for txt in good:
            txt = clean_text(txt, self._re_html, self._re_ws).strip()
            texts.append(txt if txt.endswith(SPECIAL_TOKENS["eos_token"]) else txt + " <EOS>")

        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        
        samples = []
        for enc in encodings:
            ids = enc.ids
            if not ids:
                continue
            ids = ids[: self.context_length]  # Limit token count, not raw text
            if ids[-1] != self.eos_id:
                ids = ids[:-1] + [self.eos_id] if len(ids) == self.context_length else ids + [self.eos_id]

            # Loss covers bot turns only
            content = bot_content_mask(ids, self.usr_id, self.bot_id, self.eos_id)
            if not any(content[1:]):
                continue

            samples.append((ids, content))

        if not samples:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        bl = max(len(ids) for ids, _ in samples)

        inputs = []
        targets = []

        for ids, content in samples:
            # Pad to the longest sequence in this batch rather than always to context_length
            pad_len = bl - len(ids)
            inp = ids + [0] * pad_len

            # Teacher forcing: tgt[j] is what the model must predict at position j, i.e. ids[j+1].
            # See ignore_index=-100 in the cross entropy loss in training/train.py and training/pretraining/pretrain.py.
            tgt = [
                ids[j + 1] if content[j + 1] else -100
                for j in range(len(ids) - 1)
            ] + [-100] * (pad_len + 1)

            inputs.append(inp)
            targets.append(tgt)

        input_tensor  = torch.tensor(inputs,  dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)

        return input_tensor, target_tensor
