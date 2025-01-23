from src.data.tasks.base import GenericTaskPromptBuilder


PROMPT_START = """
For this task, find the shortest span needed to answer the question.
The texts are academic news from CSIC (for Spanish) and Cambridge University (for English).
In all cases, the answers are fragments of the text and all questions can be answered from the text.
"""


class SqacSquad2024PromptBuilder(GenericTaskPromptBuilder):
    def __init__(
        self,
        prompt_start=PROMPT_START,
        prompt_guide="",
        prompt_end="",
    ):
        super().__init__(prompt_start, prompt_guide, prompt_end)

    def format_input(self, entry):
        return entry["question"]

    def format_context(self, entry):
        return f"{entry['title']}\n{entry['context']}"

    def format_output(self, entry):
        return entry["value"]
