"""Shared <USR>/<BOT> dialogue rendering."""

from global_info.global_info import SPECIAL_TOKENS

USR = SPECIAL_TOKENS["usr_token"]
BOT = SPECIAL_TOKENS["bot_token"]
EOS = SPECIAL_TOKENS["eos_token"]


def extract_turns(row):
    """Flatten one dataset row into an ordered list of utterances.

    Handles the two shapes worth supporting: SODA's plain list of alternating
    strings, and the OpenAI-style [{"role", "content"}] list used by UltraChat.
    """
    dialogue = row.get("dialogue")
    if isinstance(dialogue, list) and dialogue and isinstance(dialogue[0], str):
        return dialogue

    messages = row.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        # Only keep user/assistant messages which alternate
        msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        expected = ["user", "assistant"] * (len(msgs) // 2 + 1)
        
        if not msgs or [m["role"] for m in msgs] != expected[:len(msgs)]:
            return []
        
        return [m.get("content", "") for m in msgs]

    return []


def render_dialogue(turns, min_turns=4, max_turns=16):
    """Render alternating utterances as '<USR> u <BOT> b <EOS> <USR> u <BOT> b <EOS>'.

    Every bot turn is closed with its own <EOS> rather than only terminating the
    whole conversation. That is what lets generate() stop after a single reply at
    inference: the model is shown, every turn, that a bot turn ends at <EOS>.
    """
    turns = [t for t in turns if isinstance(t, str) and t.strip()]
    turns = turns[:max_turns]

    # keep whole USR/BOT pairs only, so a rendered dialogue always ends on <EOS>
    if len(turns) % 2:
        turns = turns[:-1]
    if len(turns) < min_turns:
        return None

    parts = []
    for i, turn in enumerate(turns):
        turn = " ".join(turn.split())
        if i % 2 == 0:
            parts.append(f"{USR} {turn}")
        else:
            parts.append(f"{BOT} {turn} {EOS}")

    return " ".join(parts)
