from pathlib import Path

# All commonly refereneced paths 
ROOT_DIR : Path = Path(__file__).resolve().parent.parent # Chat2/

TOKENIZER_PATH : Path = ROOT_DIR / "models" / "bpe.json"
PRETRAIN_DIR : Path = ROOT_DIR / "models" / "Pretrain"
CHAT_MODELS_DIR : Path = ROOT_DIR / "models" / "Chat"

# rendered SODA dialogues, cached so every run does not re-stream them
CHAT_DIALOGUE_CACHE_PATH : Path = ROOT_DIR / "models" / "chat_dialogues.txt"

PRETRAIN_CORPUS_PATH : Path = ROOT_DIR / "tokenizer" / "corpus.txt"  # RedPajama sample 
PRETRAIN_TOKENS_PATH : Path = PRETRAIN_DIR / "pretrain_tokens.bin"
PRETRAIN_TOKENS_META_PATH : Path = PRETRAIN_DIR / "pretrain_tokens_meta.json"
PRETRAIN_CHECKPOINT_PATH : Path = PRETRAIN_DIR / "pretrain_final.pth"

CHAT_MODEL_PATH = CHAT_MODELS_DIR / "model_best.pth"

SPECIAL_TOKENS = { # Special token config (reused in train + load)
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "eos_token": "<EOS>",
    "usr_token": "<USR>", 
    "bot_token": "<BOT>"
}

CHAT_DATA_CONFIG = { # Used by training/data_loader.py to build the fine-tune mix
    "dolly_dataset": "databricks/databricks-dolly-15k",
    "dolly_split": "train",
    "dolly_repeat": 1,

    "max_response_tokens": 128, # alpaca median response is 114
    "instruct_dataset": "yahma/alpaca-cleaned",
    "instruct_split": "train",
    "instruct_samples": 52_000,

    "dialogue_dataset": "allenai/soda",
    "dialogue_split": "train",
    "dialogue_samples": 15_000, # Served from cache (chat_dialogues.txt)

    "assistant_dataset": "HuggingFaceTB/smoltalk",
    "assistant_config": "everyday-conversations",
    "assistant_repeat": 3,
     
    "val_fraction": 0.02, # Training can be scored on data it never saw.
    "split_seed": 1234,
    "min_dialogue_turns": 4,
    "max_dialogue_turns": 12,     # keep a rendered dialogue inside the 512 context
}

PRETRAIN_DATA_CONFIG = { # Used by training/pretraining/prepare_pretrain_data.py to build the packed token file
    "target_tokens": 600_000_000,  # 3hr pretrain on an RTX 2060
    "min_line_chars": 20,
    "min_ascii_ratio": 0.85,
    "batch_lines": 256, # lines per tokenizer.encode_batch call
    "dialogue_dataset": "allenai/soda", # Multi-turn dialogue (SODA) mixed into the pretrain stream alongside RedPajama.
    "dialogue_split": "train",
    "dialogue_fraction": 0.25,  # Fraction of target_tokens drawn from dialogue rather than general web text.
    "min_dialogue_turns": 4,
    "max_dialogue_turns": 16,   # keep a rendered dialogue near the 512 context
}

PRETRAIN_CONFIG = { # Training values used in training/pretraining/pretrain.py
    "lr": 6e-4,
    "weight_decay": 0.01,
    "epochs": 1,          # one pass over the packed token stream is already 130x Dolly's token count
    "batch_size": 8, # logit tensor at 8 x 512 x 16000 (batch_size x context_length x vocab_size)
    "num_workers": 1,
    "grad_accum_steps": 4,
    "checkpoint_every": 1,
    "model_dropout" : 0.0, # 1 inital epoch, nothing to overfit
}

MODEL_CONFIG = { # Model values to be passed into LanguageModel constructor in language_model/model.py   
    "embed_dim": 512,
    "num_layers": 6,
    "heads": 8,
    "context_length": 512 
}

TRAINING_CONFIG = { # Training values used in train() in training/train.py
    "lr": 1e-4,
    "weight_decay": 0.01,
    "epochs": 6,
    "early_stop_patience": 3, # epochs with no new best held-out loss
    "min_lr_ratio": 0.1,
    "batch_size": 4,
    "num_workers": 1,
    "grad_accum_steps": 8, # effective batch size = batch_size * grad_accum_steps = 32
    "checkpoint_every": 2,
    "model_dropout" : 0.1, # Prevent overfitting
    "replay_every": 3,     # Replay pretrain data during training, ~33% more compute time per epoch
    "replay_weight": 1.0,
}

# Decoding defaults
GENERATION_CONFIG = {
    "max_new_tokens": 140,
    "temperature": 0.7, # Creativity not allowed!
    "top_k": 24, # Cut rank 1000+
    "top_p": 0.85, # Cut rank 1000+
    "repetition_penalty": 1.05,
    "repetition_cap": 2.0,
    "no_repeat_ngram_size": 4,
}

TEST_PROMPTS = [
    "<USR> Hi! How are you doing today? <BOT>",
    "<USR> How do you stay healthy? <BOT>",
    "<USR> What is the capital of France? <BOT>", # Quite useful to detect repetition
    "<USR> What can I do to motivate myself? <BOT>",
    "<USR> What is photosynthesis? <BOT>",
    # multi-turn: the reply has to use the earlier turn, not just the last one
    "<USR> What should I make for dinner tonight? <BOT> How about pasta? It's quick and easy. <EOS>"
    "<USR> I don't have any tomatoes though. <BOT>",
    "<USR> How often should I exercise? <BOT> Starting out with 3 times a week is healthy. <EOS>"
    "<USR> I'm lazy though <BOT> You should motivate yourself to start! The first step is the hardest. <EOS>"
    "<USR> What can I do to motivate myself? <BOT>",
]