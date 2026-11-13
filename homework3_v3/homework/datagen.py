import json
from pathlib import Path

from .data import DATA_DIR, Dataset, is_answer_valid


def _auto_batch_size(model, items, oversample: int, new_tokens: int = 50) -> int:
    """
    Pick how many questions fit in one generate call. The KV cache dominates memory: every one of the
    `oversample` sequences per question stores keys/values for the whole (long CoT) prompt plus new tokens.
    """
    import torch

    if not torch.cuda.is_available():
        return 2

    cfg = model.model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    dtype_bytes = next(model.model.parameters()).element_size()
    kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * dtype_bytes

    longest = max(len(model.tokenizer(model.format_prompt(q))["input_ids"]) for q, _ in items[:64])
    bytes_per_question = oversample * (longest + new_tokens) * kv_bytes_per_token

    free, _ = torch.cuda.mem_get_info()  # model weights are already loaded at this point
    return max(1, int(0.6 * free // bytes_per_question))  # 0.6 leaves room for activations / logits


def generate_dataset(
    output_json: str = str(DATA_DIR / "rft.json"),
    oversample: int = 10,
    temperature: float = 0.6,
    checkpoint: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    num_questions: int | None = None,
    batch_size: int | None = None,
):
    """
    Rejection sampling: sample `oversample` chain-of-thought completions per training question from
    a stronger model, keep ONE completion whose answer is correct, and drop the question if none are.

    Writes a json list of [question, correct_answer, reasoning] to `output_json`.
    num_questions is for smoke tests. batch_size is the number of questions per generate call
    (each produces batch_size * oversample sequences); if None it is chosen from the free GPU memory.
    """
    from .cot import CoTModel

    model = CoTModel(checkpoint)
    train = Dataset("train")
    items = [train[i] for i in range(len(train) if num_questions is None else min(num_questions, len(train)))]

    if batch_size is None:
        batch_size = _auto_batch_size(model, items, oversample)
        print(f"Using batch_size={batch_size} questions ({batch_size * oversample} sequences) per call", flush=True)

    rows = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompts = [model.format_prompt(question) for question, _ in chunk]
        generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

        for (question, truth), candidates in zip(chunk, generations):
            best = None  # (relative_error, length, text)
            for text in candidates:
                text = text.strip()
                # Need a complete answer tag; drop anything the model rambled after it
                if "</answer>" not in text:
                    continue
                text = text[: text.index("</answer>") + len("</answer>")]

                answer = model.parse_answer(text)
                if answer != answer or not is_answer_valid(answer, truth):
                    continue

                # Among correct candidates, prefer the most accurate, then the shortest
                error = abs(answer - truth) / abs(truth) if truth else abs(answer - truth)
                if best is None or (error, len(text)) < best[:2]:
                    best = (error, len(text), text)

            if best is not None:
                rows.append([question, truth, best[2]])

        print(f"{min(start + batch_size, len(items))}/{len(items)} questions, {len(rows)} kept", flush=True)

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Kept {len(rows)}/{len(items)} questions ({len(rows) / max(len(items), 1):.1%}) -> {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
