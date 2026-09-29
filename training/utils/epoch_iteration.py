import math

import torch
from torch.amp.autocast_mode import autocast
import torch.nn.functional as F

from global_info.global_info import TRAINING_CONFIG, GENERATION_CONFIG
from tokenizer.built_tokenizer_utils import Tokenizer, detokenize
from language_model.model import LanguageModel

def lr_lambda(step : int, warmup_steps : int, total_steps : int):
    if step < warmup_steps:
        return step / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / decay_steps))

    return max(TRAINING_CONFIG["min_lr_ratio"], cosine_decay)

def sample(model : LanguageModel, tokenizer : Tokenizer, prompt : str) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([ids]).cuda()

    out = model.generate(input_ids)

    # Show the reply only
    reply = out[0, input_ids.size(1):].tolist()
    decoded = detokenize(tokenizer.decode(reply, skip_special_tokens=True))

    print(f"[SAMPLE] {prompt} -> {decoded}\n")

    return decoded

@torch.no_grad()
def validate(model : LanguageModel, val_loaders : dict, vocab_size : int):
    """Mean cross-entropy per supervised token on held-out data and per source"""

    model.eval()
    total, count, per_source = 0.0, 0, {}

    for name, source_loader in val_loaders.items():
        s_total, s_count = 0.0, 0
        for x, y in source_loader:
            if x.numel() == 0:
                continue

            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(x)

            s_total += F.cross_entropy(
                logits.view(-1, vocab_size).float(),
                y.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).item()
            s_count += (y != -100).sum().item()

        if s_count:
            per_source[name] = s_total / s_count
        total += s_total
        count += s_count

    model.train()

    return total / max(count, 1), per_source
