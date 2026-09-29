# Table of Contents
1. [Summary](#summary)
2. [Components](#components)
   - [Depenedencies](#dependencies)
   - [Tokenizer](#tokenizer)
   - [Pretraining](#pretraining)
   - [Training (fine-tuning)](#training-(fine-tuning))
   - [Testing and Development](#testing-and-development)
3. [Future Additions](#future-additions)

# Summary

Global info

# Installation

# Usage 

# Components 

In each section, we review each component's particular design, including the why and how each component was made. 

## Depenedencies 

Our Python package manager of choice is `uv`, as it's fast to download these big modules used. 

Main dependencies of Chat2:
- `numpy`
- `datasets`
- `tokenizers`

## Tokenizer

### build_tokenizer.py 

The tokenizer is HuggingFace's Byte-Pair Encoding (BPE), with a 16 thousand word vocabulary size. It's trained on the `RedPajama-Data-1T-Sample` data set, using a minimum of 2 token pairs to filter rare merges and keep the dataset broad. 

Note that we write `corpus.txt` (a ~5 GiB download of `RedPajama-Data-1T-Sample`) before training the model. So we check if it exists to avoid re-downloading it.

The final output is saved at `Chat2/models/bpe.json`.

### built_tokenizer_utils.py

Now for using the tokenizer, we look at `tokenizer/built_tokenizer_utils.py`, which contains our normalization and `load_tokenizer()` which is used during the training pipeline. 

NFKC normalization was used for dataset standardization, ensuring precise string matching.

## Pretraining

## Training (fine-tuning)

### Validation

## Testing and Development 

# Future Additions
