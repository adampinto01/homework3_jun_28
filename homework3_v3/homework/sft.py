from math import floor, log10
from pathlib import Path

from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # The grader allows 5% relative error, so 4 significant figures lose nothing and strip float
    # noise like 5999.999999999999. Avoid scientific notation (e.g. 2.087e+04), which is harder to learn.
    answer = float(answer)
    digits = 3 - floor(log10(abs(answer))) if answer != 0 else 0
    rounded = round(answer, digits)
    text = str(int(rounded)) if rounded == int(rounded) else str(rounded)
    return {"question": prompt, "answer": f"<answer>{text}</answer>"}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = str(Path(__file__).parent / "sft_model"),
    epochs: int = 5,
    lr: float = 2e-4,
    rank: int = 8,
    max_steps: int = -1,
    num_train: int | None = None,
    run_test: bool = True,
):
    """
    Fine-tune BaseLLM with a LoRA adapter on the training set and save only the adapter.

    max_steps / num_train are for quick smoke tests (e.g. max_steps=2 num_train=32 run_test=False).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()

    # Rank 8 over all linear layers is ~17MB in fp32 for SmolLM2-360M (limit: 20MB)
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
        r=rank,
        lora_alpha=4 * rank,
        lora_dropout=0.05,
    )
    model = get_peft_model(llm.model, config)
    model.enable_input_require_grads()  # required for gradient_checkpointing with a frozen embedding layer
    model.print_trainable_parameters()

    train_data = Dataset("train")
    if num_train is not None:
        train_data.data = train_data.data[:num_train]
    train_dataset = TokenizedDataset(llm.tokenizer, train_data, format_example)

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",  # avoid checkpoint-* folders that would bloat the submission
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()

    trainer.save_model(output_dir)  # saves only the LoRA adapter
    if run_test:
        test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
