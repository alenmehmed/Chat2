import sys
import math
import multiprocessing
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

from global_info.global_info import (
    TOKENIZER_PATH,
    CHAT_MODELS_DIR,
    TRAINING_CONFIG,
    MODEL_CONFIG,
    PRETRAIN_CHECKPOINT_PATH,
    PRETRAIN_TOKENS_PATH,
    PRETRAIN_TOKENS_META_PATH,
    TEST_PROMPTS
)
from training.data_loader import ChatDataset
from training.training_plots.plot import plot_training_and_perplexity
from training.utils.epoch_iteration import (
    lr_lambda as make_lr_lambda, # Alias's used for partial function construction
    validate,
    sample
)
from training.collator import Collator
from training.pretraining.pretrain_dataset import PackedTokenDataset
from tokenizer.built_tokenizer_utils import load_tokenizer
from language_model.model import LanguageModel

# pin PyTorch to use all hardware threads
torch.set_num_threads(multiprocessing.cpu_count())
torch.set_num_interop_threads(multiprocessing.cpu_count())

# The vocab can produce characters outside Windows' default cp1252 console codepage
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

def train():
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    model = LanguageModel(
        vocab_size=vocab_size,
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_layers=MODEL_CONFIG["num_layers"],
        heads=MODEL_CONFIG["heads"],
        context_length=MODEL_CONFIG["context_length"],
        dropout=TRAINING_CONFIG["model_dropout"], # Prevent overfitting
        eos_token_id=tokenizer.token_to_id("<EOS>"),
    ).cuda()

    if PRETRAIN_CHECKPOINT_PATH.exists():
        print(f"Loading pretrained weights from {PRETRAIN_CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(PRETRAIN_CHECKPOINT_PATH, map_location="cuda"))
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found at {PRETRAIN_CHECKPOINT_PATH}")

    print("Sampling pre-trained model:")
    idx = 0
    for prompt in TEST_PROMPTS:
        idx = idx + 5
        torch.manual_seed(123 + idx)
        # model.eval() called in sample()
        sample(model, tokenizer, prompt=prompt)

    data = ChatDataset(split="train")
    val_data = ChatDataset(split="val")
    collate = Collator(tokenizer, MODEL_CONFIG["context_length"])

    loader = DataLoader(
        data,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loaders = {
        name: DataLoader(
            Subset(val_data, [k for k in range(len(val_data)) if val_data.source_of(k) == src]),
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False, # No need to shuffle held-out data
            collate_fn=collate,
            pin_memory=True,
        )
        for src, name in ChatDataset.SOURCE_NAMES.items()
    }

    replay_iter = iter(DataLoader(
        dataset=PackedTokenDataset(PRETRAIN_TOKENS_PATH, PRETRAIN_TOKENS_META_PATH, MODEL_CONFIG["context_length"]),
        batch_size=TRAINING_CONFIG["batch_size"], 
        shuffle=True, 
        pin_memory=True
    ))

    iters_per_epoch = len(loader)

    total_steps = TRAINING_CONFIG["epochs"] * iters_per_epoch // TRAINING_CONFIG["grad_accum_steps"]
    warmup_steps = max(100, int(0.05 * total_steps))

    lr_lambda = partial(make_lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps)

    decay    = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]   # biases, LayerNorm
    optimizer = AdamW([{"params": decay, "weight_decay": TRAINING_CONFIG["weight_decay"]},
                       {"params": no_decay, "weight_decay": 0.0}],
                       lr=TRAINING_CONFIG["lr"], betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    accum_steps = TRAINING_CONFIG["grad_accum_steps"]

    epoch_losses = []  
    perplexities = []
    val_losses = []
    val_scores = []

    best_val = float("inf")
    epochs_since_best = 0
    scaler = GradScaler('cuda')

    for epoch in range(TRAINING_CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)  # drop any partial accumulation
        losses = []
        loader_iter = iter(loader)
        pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}")

        for i in pbar:
            try:
                input_tensor, target_tensor = next(loader_iter)
            except StopIteration:
                break
            if input_tensor is None or input_tensor.numel() == 0:
                continue
            if not (target_tensor != -100).any():
                continue  # nothing supervised in this batch -> CE would be nan

            input_tensor = input_tensor.cuda(non_blocking=True)
            target_tensor = target_tensor.cuda(non_blocking=True)

            with autocast(device_type="cuda"):
                logits = model(input_tensor)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    target_tensor.view(-1),
                    ignore_index=-100, # Ignore tokens of -100 in the target tensor
                ) / accum_steps

            scaler.scale(loss).backward()

            # Replay pretraining data to prevent forgetting of general knowledge
            # dropout of 0.1 and weight decay should be enough to prevent overfitting 
            if i % TRAINING_CONFIG["replay_every"] == 0:
                rx, ry = next(replay_iter)
                
                with autocast(device_type="cuda"):
                    r_logits = model(rx.cuda(non_blocking=True))
                    r_loss = F.cross_entropy(r_logits.view(-1, vocab_size), ry.cuda(non_blocking=True).view(-1))

                scaler.scale(r_loss * TRAINING_CONFIG["replay_weight"] / accum_steps).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer) # Unscale before clipping
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

        val_loss, val_by_source = validate(model=model, val_loaders=val_loaders, vocab_size=vocab_size)
        val_losses.append(val_loss)

        val_score = sum(val_by_source.values()) / len(val_by_source) # Validation score is average loss among sources
        val_scores.append(val_score)

        print(f"[Epoch {epoch+1}]")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        print("\n")
        print(f"Avg Training Loss: {avg_loss:.4f}")
        print(f"Perplexity: {math.exp(avg_loss):.2f}")
        print("\n")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Score: {val_score:.4f}")
        print(f"Gap: {val_loss - avg_loss:+.3f}")
        print(f"\n")
        print("Validation by source: " + " | ".join(f"{name} {loss:.4f}" for name, loss in val_by_source.items()))

        if (epoch + 1) % TRAINING_CONFIG["checkpoint_every"] == 0:
            torch.save(model.state_dict(), CHAT_MODELS_DIR / f"model_epoch{epoch+1}.pth")

        # Held-out loss, not training loss, is what says which checkpoint to ship.
        if val_score < best_val:
            best_val = val_score
            epochs_since_best = 0

            torch.save(model.state_dict(), CHAT_MODELS_DIR / "model_best.pth")
            print(f"New best held-out macro loss - saved at {CHAT_MODELS_DIR / 'model_best.pth'}")
        else:
            epochs_since_best += 1

        for prompt in TEST_PROMPTS:
            idx = idx + 5
            torch.manual_seed(123 + idx)
            # model.eval() called in sample()
            sample(model, tokenizer, prompt=prompt)
            
        if epochs_since_best >= TRAINING_CONFIG["early_stop_patience"]:
            print(f"No held-out improvement for {epochs_since_best} epochs - stopping early.")
            break

    best_epoch = min(range(len(val_losses)), key=lambda i: val_losses[i]) + 1
    print(f"\nBest held-out loss {best_val:.4f} (ppl {math.exp(best_val):.2f}) "f"at epoch {best_epoch} -> model_best.pth")

    plot_training_and_perplexity(epoch_losses, perplexities, iters_per_epoch)

if __name__ == "__main__":
    train()
    