from src.preprocessing.diann_2023_t1 import extract_bio_tokens, extract_bio_spans
from src.data.tasks.base import GenericTaskPromptBuilder, ContextualTaskPromptBuilder


class Diann2023T1PromptBuilderBIO(GenericTaskPromptBuilder):
    def init(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
        prompt_guide="",
        prompt_end="",
    ):
        super().init(prompt_start, prompt_guide, prompt_end)

    def format_input(self, entry):
        return str(entry["tokens"])

    def format_output(self, entry):
        return str(entry["value"])


class Diann2023T1PromptBuilderTokenIdentification(GenericTaskPromptBuilder):
    def init(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
        prompt_guide="",
        prompt_end="",
    ):
        super().init(prompt_start, prompt_guide, prompt_end)

    def format_input(self, entry):
        return str(entry["tokens"])

    def format_output(self, entry):
        return extract_bio_tokens(entry)


class Diann2023T1PromptBuilderGenerativeNER(GenericTaskPromptBuilder):
    def init(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
        prompt_guide="",
        prompt_end="",
    ):
        super().init(prompt_start, prompt_guide, prompt_end)

    def format_input(self, entry):
        return str.join(" ", entry["tokens"])

    def format_output(self, entry):
        return extract_bio_spans(entry)


class Diann2023T1ContextualPromptBuilderNER(ContextualTaskPromptBuilder):
    def init(
        self,
        prompt_start="For this task, solve the following Named Entity Recognition problem:\n",
    ):
        super().init(prompt_start)

    def format_input(self, entry) -> str:
        return str.join(" ", entry["tokens"])

    def format_output(self, entry) -> str | None:
        return extract_bio_spans(entry)

    def format_context(self, entry, retrieved) -> str | None:
        retrieved_outputs = [self.format_output(item) for item in retrieved]
        return str(retrieved_outputs)