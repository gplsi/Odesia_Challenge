from typing import List, Tuple
from abc import ABC, abstractmethod

from src.data.base import TaskPromptBuilder, PromptSyntax


class GenericTaskPromptBuilder(TaskPromptBuilder):
    def __init__(
        self,
        prompt_start="",
        prompt_guide="",
        prompt_end="",
    ):
        self.prompt_start = prompt_start
        self.prompt_guide = prompt_guide
        self.prompt_end = prompt_end

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
                formated_context=self.format_context(entry),
                formated_answer=self.format_output(entry),
            )
            for entry in retrieved
        ]
        prompt = prompt_syntax.build(
            formated_question=self.format_input(input),
            formated_context=self.format_context(input),
        )
        examples_prompt = (
            "\n" + self.prompt_guide + "\n".join(retrieved_prompts)
            if retrieved_prompts
            else ""
        )
        return (
            self.prompt_start + examples_prompt + "\n" + self.prompt_end + prompt,
            self.format_output(input) if answer else None,
        )

    @abstractmethod
    def format_input(self, entry) -> str:
        pass

    @abstractmethod
    def format_output(self, entry) -> str | None:
        pass

    def format_context(self, entry) -> str | None:
        return None


class ContextualTaskPromptBuilder(TaskPromptBuilder):
    def __init__(
        self,
        prompt_start="",
    ):
        self.prompt_start = prompt_start

    def build(
        self,
        prompt_syntax: PromptSyntax,
        input: dict,
        retrieved: List[dict],
        answer: bool,
    ) -> Tuple[str, str]:
        prompt = prompt_syntax.build(
            formated_question=self.format_input(input),
            formated_context=self.format_context(input, retrieved),
        )
        return (
            f"{self.prompt_start}\n{prompt}",
            self.format_output(input) if answer else None,
        )

    @abstractmethod
    def format_input(self, entry) -> str:
        pass

    @abstractmethod
    def format_output(self, entry) -> str | None:
        pass

    @abstractmethod
    def format_context(self, entry, retrieved) -> str | None:
        pass
