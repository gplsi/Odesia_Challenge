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
        prompt += (f"Context: {formated_context}\n" if formated_context else "")
        prompt += "Answer: " + (f"{formated_answer}" if formated_answer else "")
        return prompt