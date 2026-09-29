import sys
import math
import multiprocessing
from functools import partial

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from global_info.global_info import (
    TOKENIZER_PATH,
    PRETRAIN_DIR,
    PRETRAIN_TOKENS_PATH,
    PRETRAIN_TOKENS_META_PATH,
    PRETRAIN_CHECKPOINT_PATH,
    PRETRAIN_CONFIG,
    MODEL_CONFIG
)
from training.pretraining.pretrain_dataset import PackedTokenDataset
from training.utils.epoch_iteration import (
    lr_lambda as make_lr_lambda, # Alias used for partial function construction
    sample
)
from tokenizer.built_tokenizer_utils import load_tokenizer
from language_model.model import LanguageModel

# Use all of the threads
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

# sample output can contain characters outside Windows' default cp1252 console codepage.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

def pretrain():
    if not PRETRAIN_TOKENS_PATH.exists() or not PRETRAIN_TOKENS_META_PATH.exists():
        raise FileNotFoundError("Packed pretrain tokens not found. Run training/prepare_pretrain_data.py first.")

    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    model = LanguageModel(
        vocab_size=vocab_size,
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        heads=MODEL_CONFIG["heads"],
        context_length=MODEL_CONFIG["context_length"],
        eos_token_id=tokenizer.token_to_id("<EOS>"),
    ).cuda()

    data = PackedTokenDataset(PRETRAIN_TOKENS_PATH, PRETRAIN_TOKENS_META_PATH, MODEL_CONFIG["context_length"])
    loader = DataLoader(
        data,
        batch_size=PRETRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=PRETRAIN_CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    iters_per_epoch = len(loader)

    total_steps = PRETRAIN_CONFIG["epochs"] * iters_per_epoch // PRETRAIN_CONFIG["grad_accum_steps"]
    warmup_steps = max(100, int(0.05 * total_steps))

    lr_lambda = partial(make_lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps)

    optimizer = AdamW(model.parameters(), lr=PRETRAIN_CONFIG["lr"], weight_decay=PRETRAIN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    accum_steps = PRETRAIN_CONFIG["grad_accum_steps"]
    epoch_losses = []
    perplexities = []
    scaler = GradScaler("cuda")

    for epoch in range(PRETRAIN_CONFIG["epochs"]): # Only one epoch for now
        model.train()
        losses = []
        pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch+1}")

        for i, (input_tensor, target_tensor) in enumerate(pbar):
            input_tensor = input_tensor.cuda(non_blocking=True)
            target_tensor = target_tensor.cuda(non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(input_tensor)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    target_tensor.view(-1),
                    label_smoothing=PRETRAIN_CONFIG["label_smoothing"],
                ) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                # see train.py: unscale before clipping, not after
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item() * accum_steps)
            pbar.set_postfix(loss=loss.item() * accum_steps)

        avg_loss = sum(losses) / (len(losses) + 1e-8)
        perplexity = math.exp(avg_loss)
        perplexities.append(perplexity)
        epoch_losses.append(avg_loss)

        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % PRETRAIN_CONFIG["checkpoint_every"] == 0:
            torch.save(model.state_dict(), PRETRAIN_DIR / f"pretrain_epoch{epoch+1}.pth")

        sample(model, tokenizer, prompt="The weather today is")
        sample(model, tokenizer, prompt="<USR> Hey, how was your weekend? <BOT>")

    torch.save(model.state_dict(), PRETRAIN_CHECKPOINT_PATH)
    print(f"Saved final pretrained weights to {PRETRAIN_CHECKPOINT_PATH}")

if __name__ == "__main__":
    pretrain()
