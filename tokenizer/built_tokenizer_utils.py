from pathlib import Path

from global_info.global_info import SPECIAL_TOKENS

from tokenizers import Tokenizer, decoders, pre_tokenizers, normalizers, AddedToken

SPECIAL_TOKEN_LIST = list(SPECIAL_TOKENS.values())


def configure_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    return tokenizer


def special_added_tokens():
    # lstrip lets a marker absorb the space in front of it
    # rstrip stays off so the first word of a reply keeps its leading-space form
    return [
        AddedToken(token, lstrip=True, rstrip=False, single_word=False, normalized=False)
        for token in SPECIAL_TOKEN_LIST
    ]


def load_tokenizer(path : Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at: {path}")

    tokenizer = configure_tokenizer(Tokenizer.from_file(str(path)))
    tokenizer.add_special_tokens(special_added_tokens())

    return tokenizer


def detokenize(text: str) -> str:
    return text.strip() # To be concise :-)
