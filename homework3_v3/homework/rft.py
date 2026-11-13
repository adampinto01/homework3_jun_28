from pathlib import Path

from .base_llm import BaseLLM
from .data import Dataset
from .sft import TokenizedDataset, test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_rft_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    """
    RFT rows are [question, correct_answer, reasoning]. The completion is the full reasoning
    (which already ends in <answer>...</answer>), so the model learns to reason before answering.
    """
    return {"question": question, "answer": reasoning}


def train_model(
    output_dir: str = str(Path(__file__).parent / "rft_model"),
    epochs: int = 5,
    lr: float = 2e-4,
    rank: int = 8,
    data: str = "rft",
    max_steps: int = -1,
    num_train: int | None = None,
    run_test: bool = True,
):
    """
    Fine-tune BaseLLM with a LoRA adapter on the rejection-sampled question / reasoning data
    (data/<data>.json, produced by datagen.py) and save only the adapter.

    max_steps / num_train are for quick smoke tests (e.g. max_steps=2 num_train=32 run_test=False).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()

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

    train_data = Dataset(data)
    if num_train is not None:
        train_data.data = train_data.data[:num_train]
    train_dataset = TokenizedDataset(llm.tokenizer, train_data, format_rft_example)

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


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
