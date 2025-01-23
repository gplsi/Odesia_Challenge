from src.data.tasks.base import GenericTaskPromptBuilder


class DipromatsT1PromptBuilder(GenericTaskPromptBuilder):

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
        return entry["value"]
