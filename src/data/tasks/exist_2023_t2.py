from typing import List, Tuple
from src.data.base import Dataset, TaskPromptBuilder, PromptSyntax
from src.preprocessing.exist_2023_t2 import get_counts


class Exist2023T2PromptBuilder(TaskPromptBuilder):
    def __init__(
        self,
        prompt_start="For this task, solve the following classification problem:\n",
    ):
        self.prompt_start = prompt_start

    def build(
        self,
        prompt_syntax: PromptSyntax,
        input: dict,
        retrieved: List[dict],
        answer: bool,
    ) -> Tuple[str, str]:
        retrieved_prompts = [
            prompt_syntax.build(
                formated_question=self.format_input(entry),
                formated_answer=self.format_output(entry),
            )
            for entry in retrieved
        ]

        prompt = prompt_syntax.build(self.format_input(input))
        return (
            self.prompt_start + "\n".join(retrieved_prompts) + "\n" + prompt,
            self.format_output(input) if answer else None,
        )

    def format_input(self, entry):
        return entry["tweet"]

    def format_output(self, entry):
        counts = get_counts(entry)
        return str(counts)
