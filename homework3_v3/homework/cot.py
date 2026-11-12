from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You convert units. Always answer in exactly this form: "
                    "'1 <from> = <factor> <to>. <N> * <factor> = <answer>RESULT</answer>'. "
                    "Always finish with the answer tag; put only digits inside it, never units.\n"
                    "Facts: 1 kB = 1000 B (not 1024); 1 B = 8 bit; 1 MB = 1000 kB; 1 GB = 1000 MB; "
                    "1 year = 12 months = 52.18 weeks = 365.24 days; 1 month = 730.5 h; 1 decade = 10 years; "
                    "1 century = 100 years; 1 h = 3600 s; 1 in = 2.54 cm; 1 ft = 12 in; 1 yd = 3 ft; "
                    "1 mi = 1760 yd = 1609.34 m; 1 kg = 2.2046 pound = 35.274 ounce; 1 pound = 16 ounce; "
                    "1 gallon = 4 quart = 128 fluid ounce; 1 quart = 2 pint; 1 fluid ounce = 29.57 mL; "
                    "1 m/s = 3.6 km/h; 1 mi/h = 1.4667 ft/s = 0.44704 m/s."
                ),
            },
            # Three short worked examples that all complete the full pattern
            {"role": "user", "content": "How many min are there in 2 days?"},
            {"role": "assistant", "content": "1 day = 1440 min. 2 * 1440 = <answer>2880</answer>"},
            {"role": "user", "content": "Convert 4 pound to ounce."},
            {"role": "assistant", "content": "1 pound = 16 ounce. 4 * 16 = <answer>64</answer>"},
            {"role": "user", "content": "What is 10 km in mi?"},
            {"role": "assistant", "content": "1 km = 0.6214 mi. 10 * 0.6214 = <answer>6.214</answer>"},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

    #Analyze responses to the first 10 questions in the validation set
    # n = 10
    # questions = [testset[i][0] for i in range(n)]
    # prompts = [model.format_prompt(q) for q in questions]
    # responses = model.batched_generate(prompts)

    # for i, (question, response) in enumerate(zip(questions, responses)):
    #     correct = testset[i][1]
    #     parsed = model.parse_answer(response)
    #     print(f"[{i}] Q: {question}")
    #     print(f"    Model response: {response.strip()}")
    #     print(f"    Parsed answer:  {parsed}")
    #     print(f"    Real answer:    {correct}")
    #     print()

    # Temporary: inspect how format_prompt renders a question
    # model = CoTModel()
    # print(model.format_prompt("How many centimeters are there in 3 meters?"))


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
