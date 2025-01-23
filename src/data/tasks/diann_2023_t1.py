from src.preprocessing.diann_2023_t1 import extract_bio_tokens
from src.data.tasks.base import GenericTaskPromptBuilder


class Diann2023T1PromptBuilderBIO(GenericTaskPromptBuilder):
    def __init__(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
        prompt_guide="",
        prompt_end="",
    ):
        super().__init__(prompt_start, prompt_guide, prompt_end)

    def format_input(self, entry):
        return str(entry["tokens"])

    def format_output(self, entry):
        return str(entry["value"])


class Diann2023T1PromptBuilderTokenIdentification(GenericTaskPromptBuilder):
    def __init__(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
        prompt_guide="",
        prompt_end="",
    ):
        super().__init__(prompt_start, prompt_guide, prompt_end)

    def format_input(self, entry):
        return str(entry["tokens"])

    def format_output(self, entry):
        return extract_bio_tokens(entry)
