from typing import List
from src.data.base import Dataset, TaskPromptBuilder, PromptSyntax


class Exist2022T1PromptBuilder(TaskPromptBuilder):
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
    ) -> str:
        retrieved_prompts = [
            prompt_syntax.build(self.format_input(entry), "", self.format_output(entry))
            for entry in retrieved
        ]
        formated_answer = self.format_output(input) if answer else None
        prompt = prompt_syntax.build(self.format_input(input), "", formated_answer)
        return self.prompt_start + "\n".join(retrieved_prompts) + "\n" + prompt

    def format_input(self, entry):
        return entry["text"]
    
    def format_output(self, entry):
        return entry["value"]


