from src.data.base import PromptSyntax


class BasicSyntax(PromptSyntax):
    def build(
        self,
        formated_question: str,
        formated_context: str = None,
        formated_answer: str = None,
    ) -> str:
        prompt = ""
        prompt += f"\nText: {formated_question}\n"
        prompt += f"Context: {formated_context}\n" if formated_context else ""
        prompt += "Answer: " + (f"{formated_answer}" if formated_answer else "")
        return prompt


class CustomSyntax(PromptSyntax):
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
        formated_question: str,
        formated_context: str = None,
        formated_answer: str = None,
    ) -> str:
        prompt = self.prompt_start
        prompt += f"\nQuestion: {formated_question}\n"
        prompt += self.prompt_guide
        prompt += f"\nContext: {formated_context}\n" if formated_context else ""
        prompt += self.prompt_end
        prompt += "\nAnswer: " + (f"{formated_answer}" if formated_answer else "")
        return prompt
