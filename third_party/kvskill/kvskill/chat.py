"""Chat tokenization helpers: stop tokens and chat-template rendering with a length cap."""

from __future__ import annotations


class PromptTooLongError(ValueError):
    pass


def stop_token_ids(tokenizer, generation_config=None) -> set[int]:
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    if generation_config is not None:
        eos = generation_config.eos_token_id
        if eos is None:
            eos = []
        elif isinstance(eos, int):
            eos = [eos]
        ids.update(int(e) for e in eos)
    if not ids:
        raise ValueError("no EOS token id could be derived")
    return ids


def tokenize_prompt(tokenizer, messages, *, max_tokens: int | None = None) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if max_tokens is not None and len(ids) > max_tokens:
        raise PromptTooLongError(f"prompt has {len(ids)} tokens > cap {max_tokens}")
    return list(ids)
