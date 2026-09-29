from global_info.global_info import SPECIAL_TOKENS, TOKENIZER_PATH

from tokenizer.built_tokenizer_utils import load_tokenizer

# Plain text must round-trip exactly. Byte-level BPE makes that a hard guarantee
# (for NFKC-normalised input), so every sample is asserted, including the ones
# the WordPiece vocab could only approximate.
SAMPLES = [
    "Hello, world!",
    "This is a test of tokenizer coverage.",
    "E-mail addresses like test@example.com should tokenize reasonably.",
    "Numbers 123 and symbols $%&* shouldn't break things.",
    "CamelCaseAnd_snake_case should be split smartly.",
    "Non-ASCII: café naïve résumé — what happens?",
    "Multilingual: Привет мир, 你好世界, مرحبا بالعالم",
]

# Markers are registered with lstrip=True, so the space in front of one is
# absorbed into the marker and does not come back on decode. That is the
# intended behaviour, so it is asserted explicitly rather than skipped.
MARKER_SAMPLES = [
    ("Should keep <EOS> if special tokens are included <EOS>",
     "Should keep<EOS> if special tokens are included<EOS>"),
    ("<USR> How are you? <BOT> I'm doing fine! <EOS>",
     "<USR> How are you?<BOT> I'm doing fine!<EOS>"),
]


def test_tokenizer_coverage():
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    unk_id = tokenizer.token_to_id(SPECIAL_TOKENS["unk_token"])

    print("Loaded vocab size:", tokenizer.get_vocab_size(), "\n")
    assert tokenizer.get_vocab_size() > 0

    cases = [(text, text) for text in SAMPLES] + MARKER_SAMPLES
    for text, expected in cases:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)

        print(f"{text!r}\n  tokens : {encoded.tokens}\n  decoded: {decoded!r}\n")

        assert encoded.ids, f"no tokens for {text!r}"
        assert unk_id not in encoded.ids, f"[UNK] in {text!r} - byte-level should never produce it"
        assert decoded == expected, f"round-trip for {text!r}: got {decoded!r}, want {expected!r}"


def test_markers_do_not_cost_a_space_token():
    """The inference prompt ends '<BOT>' and the model continues from there, so the
    token before a marker must be the last real character, not a lone 'Ġ'."""
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    bot_id = tokenizer.token_to_id(SPECIAL_TOKENS["bot_token"])

    encoded = tokenizer.encode("<USR> How are you? <BOT>", add_special_tokens=False)

    assert encoded.ids[-1] == bot_id
    assert encoded.tokens[-2] == "?", encoded.tokens


if __name__ == "__main__":
    test_tokenizer_coverage()
    test_markers_do_not_cost_a_space_token()
    