import itertools
from collections.abc import Mapping

from global_info.global_info import SPECIAL_TOKENS, ROOT_DIR, PRETRAIN_CORPUS_PATH
from tokenizer.built_tokenizer_utils import configure_tokenizer, special_added_tokens

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tqdm import tqdm


if __name__ == "__main__":
    tokenizer = configure_tokenizer(Tokenizer(models.BPE()))
    corpus_file = PRETRAIN_CORPUS_PATH

    if not corpus_file.exists(): 
        # Save corpus to local disk
        dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", trust_remote_code=True)

        with open(corpus_file, "w", encoding="utf-8") as f:
            for example in tqdm(itertools.islice(dataset, 1_000_000), total=1_000_000):
                text = example.get("text", "") if isinstance(example, Mapping) else ""
                text = text.strip() if isinstance(text, str) else ""  # ignore empty or missing text

                if text:
                    f.write(text + " <EOS>\n")
    else:
        print(f"Reusing existing corpus at {corpus_file}")

    # Train tokenizer
    print("Training tokenizer")
    trainer = trainers.BpeTrainer(
        vocab_size=16_000,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS.values()),
        # Seed with all 256 byte symbols, including ones the corpus never contains
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train([str(corpus_file)], trainer=trainer)

    # Re-register with the same flags load_tokenizer() uses, so the saved file and
    # the loaded tokenizer cannot disagree about marker whitespace handling.
    tokenizer.add_special_tokens(special_added_tokens())

    # Save tokenizer as bpe.json to Chat2/models/
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    bpe_path = models_dir / "bpe.json"
    tokenizer.save(str(bpe_path))
    print(f"Tokenizer saved to {bpe_path}")
    