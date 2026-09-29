from global_info.global_info import GENERATION_CONFIG

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.attn_dropout = nn.Dropout(0.05)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_mult * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(ff_mult * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask):
        x_norm = self.norm1(x)
        a, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        a = self.attn_dropout(a)
        x = a + x
        f = self.ff(self.norm2(x))
        return f + x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, heads, context_length, eos_token_id):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, context_length, embed_dim))
        self.emb_dropout = nn.Dropout(0.05)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, heads) for _ in range(num_layers)
        ])
        self.layernorm_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.context_length = context_length
        self.eos_token_id = eos_token_id

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) # std=0.02 keeps logits from exploding (same as GPT-2)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        b, t = input_ids.size()
        x = self.token_emb(input_ids) + self.pos_emb[:, :t]
        x = self.emb_dropout(x)

        # Build a [t × t] mask of –inf above the diagonal (ignore future tokens)
        mask = torch.triu(torch.ones((t, t), device=x.device), diagonal=1).bool()  

        # Now pass this mask into each block
        for layer in self.layers:
            x = layer(x, attn_mask=mask)

        x = self.layernorm_f(x)

        return self.head(x)

    @torch.no_grad()
    def generate(self, input_ids):
        max_new_tokens = GENERATION_CONFIG["max_new_tokens"]
        temperature = GENERATION_CONFIG["temperature"]
        top_k = GENERATION_CONFIG["top_k"]
        top_p = GENERATION_CONFIG["top_p"]
        repetition_penalty = GENERATION_CONFIG["repetition_penalty"]
        repetition_cap = GENERATION_CONFIG["repetition_cap"]
        no_repeat_ngram_size = GENERATION_CONFIG["no_repeat_ngram_size"]
        
        self.eval()
        generated = input_ids.clone()
        prompt_len = input_ids.size(1)

        # keep track of which samples are "done"
        done = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            cond = generated[:, -self.context_length:]
            logits = self(cond)[:, -1, :].float()

            # Penalise only what the model itself has emitted.
            emitted = generated[:, prompt_len:]

            # Global repetition
            if repetition_penalty != 1.0 and emitted.size(1) > 0:
                counts = torch.zeros_like(logits)
                counts.scatter_add_(1, emitted, torch.ones_like(emitted, dtype=logits.dtype))
                factor = repetition_penalty ** counts.clamp(max=repetition_cap)
                logits = torch.where(logits > 0, logits / factor, logits * factor)

            # Local repetition
            if no_repeat_ngram_size >= 2 and emitted.size(1) >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                for b in range(emitted.size(0)):
                    seq = emitted[b].tolist()
                    prefix = seq[-(n - 1):]
                    for k in range(len(seq) - n + 1):
                        if seq[k : k + n - 1] == prefix:
                            logits[b, seq[k + n - 1]] = float("-inf")

            logits = logits / temperature

            # top-k: keep only the k highest-scoring tokens
            if top_k:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            # top-p :keep the shortest prefix of the sorted distribution whose mass exceeds top_p. 
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                # exclusive cumsum, so the token that crosses the threshold is kept
                drop_sorted = (sorted_probs.cumsum(dim=-1) - sorted_probs) > top_p
                drop = torch.zeros_like(drop_sorted).scatter(1, sorted_idx, drop_sorted)
                logits = logits.masked_fill(drop, float("-inf"))

            probs = F.softmax(logits, dim=-1) # probabilities vector!
            next_t = torch.multinomial(probs, num_samples=1)

            # rows that already emitted EOS keep emitting it rather than drifting
            # on into unrelated text while the rest of the batch finishes.
            next_t = torch.where(
                done[:, None], torch.full_like(next_t, self.eos_token_id), next_t
            )
            generated = torch.cat([generated, next_t], dim=1)

            # mark samples that just generated EOS
            done |= (next_t.squeeze(1) == self.eos_token_id)
            if done.all():
                break

        return generated
