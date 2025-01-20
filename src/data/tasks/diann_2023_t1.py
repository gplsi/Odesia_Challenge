from typing import List
from src.data.base import Dataset, TaskPromptBuilder, PromptSyntax
from src.preprocessing.diann_2023_t1 import extract_bio_tokens

class Diann2023T1PromptBuilderBIO(TaskPromptBuilder):
    def __init__(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
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
        return str(entry["tokens"])
    
    def format_output(self, entry):
        return str(entry["value"])


class Diann2023T1PromptBuilderTokenIdentification(TaskPromptBuilder):
    def __init__(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
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
        return str(entry["tokens"])
    
    def format_output(self, entry):
        return extract_bio_tokens(entry)


