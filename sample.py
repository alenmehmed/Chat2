from torch import load

from global_info.global_info import CHAT_MODEL_PATH, TOKENIZER_PATH, MODEL_CONFIG, TEST_PROMPTS
from tokenizer.built_tokenizer_utils import load_tokenizer
from language_model.model import LanguageModel
from training.utils.epoch_iteration import sample

tokenizer = load_tokenizer(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

chat = LanguageModel(
    vocab_size=vocab_size,
    embed_dim=MODEL_CONFIG["embed_dim"],
    num_layers=MODEL_CONFIG["num_layers"],
    heads=MODEL_CONFIG["heads"],
    context_length=MODEL_CONFIG["context_length"],
    dropout=0.1,
    eos_token_id=tokenizer.token_to_id("<EOS>"),
).cuda()

chat.load_state_dict(load(CHAT_MODEL_PATH))

print("Speak with Chat: ")

for chat_prompt in TEST_PROMPTS:
    sample(chat, tokenizer, chat_prompt)

# while True:
#     try:
#         user_input = input("> ")

#         chat_input = f"<USR> {user_input} <BOT>" # Match the training protocol

#         output = sample(chat, tokenizer, chat_input)
        
#     except EOFError:
#         print("\nExit signal received. Goodbye!")
#         break