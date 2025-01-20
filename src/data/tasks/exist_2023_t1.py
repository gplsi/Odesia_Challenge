from typing import List
from src.data.base import Dataset, TaskPromptBuilder, PromptSyntax
from src.preprocessing.exist_2023_t1 import get_counts


class Exist2023T1PromptBuilder(TaskPromptBuilder):
    def build(
        self,
        prompt_syntax: PromptSyntax,
        input: dict,
        retrieved: List[dict],
        answer: bool,
    ) -> str:
        retrieved_prompts = [
            prompt_syntax.build(self.format_input(entry), "", self.format_output(entry))
            for entry in retrieved
        ]
        formated_answer = self.format_output(input) if answer else None
        prompt = prompt_syntax.build(self.format_input(input), "", formated_answer)
        return (
            "For this task, solve the following classification problem:\n"
            + "\n".join(retrieved_prompts)
            + "\n"
            + prompt
        )

    def format_input(self, entry):
        return entry["tweet"]

    def format_output(self, entry):
        counts = get_counts(entry)
        return str(counts)
