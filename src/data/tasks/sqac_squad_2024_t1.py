from typing import List
from src.data.base import TaskPromptBuilder, PromptSyntax


PROMPT_START = """
For this task, find the shortest span needed to answer the question.
The texts are academic news from CSIC (for Spanish) and Cambridge University (for English).
In all cases, the answers are fragments of the text and all questions can be answered from the text.
"""


class SqacSquad2024PromptBuilder(TaskPromptBuilder):
    def __init__(
        self,
        prompt_start=PROMPT_START,
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
            prompt_syntax.build(
                self.format_input(entry),
                self.format_context(entry),
                self.format_output(entry),
            )
            for entry in retrieved
        ]
        prompt = prompt_syntax.build(
            self.format_input(input),
            self.format_context(input),
            self.format_output(input) if answer else None,
        )
        return self.prompt_start + "\n".join(retrieved_prompts) + "\n" + prompt

    def format_input(self, entry):
        return entry["question"]

    def format_context(self, entry):
        return f"{entry['title']}\n{entry['context']}"

    def format_output(self, entry):
        return entry["value"]
