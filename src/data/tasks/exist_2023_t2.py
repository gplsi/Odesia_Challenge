from src.data.tasks.base import GenericTaskPromptBuilder
from src.preprocessing.exist_2023_t2 import get_counts


class Exist2023T2PromptBuilder(GenericTaskPromptBuilder):
    def __init__(
        self,
        prompt_start="For this task, solve the following classification problem:\n",
        prompt_guide="",
        prompt_end="",
    ):
        super().__init__(prompt_start, prompt_guide, prompt_end)

    def format_input(self, entry):
        return entry["text"]

    def format_output(self, entry):
        counts = get_counts(entry)
        return str(counts)
