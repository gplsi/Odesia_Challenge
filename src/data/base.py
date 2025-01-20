import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Tuple, List, Dict


# Base class for datasets, which defines the structure for loading and iterating over dataset items.
class Dataset:
    def __init__(self, data: List[dict], text_key: str, transform: Callable[[dict], str] = None):
        self.data = data
        self.transform = (lambda row: row[text_key]) if transform is None else transform

    @classmethod
    def load(cls, path: str, text_key: str, transform: Callable[[dict], str] = None) -> "Dataset":
        data = json.load(open(path))
        return Dataset(data, text_key, transform)

    def items(self) -> List[Tuple[str, dict]]:
        return [(self.transform(row), row) for row in self.data]


# Base class for retrievers, which defines the structure for retrieving data based on keys and tasks.
class Retriever(ABC):
    # Abstract method to retrieve data given a key, task name, and optional limit for the number of items to return.
    @abstractmethod
    def add_data(self, task: str, dataset: Dataset) -> None:
        pass
    
    @abstractmethod
    def set_retrieve_mode(self, task: str) -> None:
        pass
    
    @abstractmethod
    def retrieve(self, query: str, limit=5) -> object:
        pass


# Base class for defining the syntax of prompts, including how they are built.
class PromptSyntax(ABC):
    # Abstract method to build a prompt by combining the formatted question, context, and optionally, an answer.
    @abstractmethod
    def build(
        self,
        formated_question: str,
        formated_context: str,
        formated_answer: str = None,
    ) -> str:
        pass


# Base class for task prompt builders, which define how to assemble prompts for specific tasks.
class TaskPromptBuilder(ABC):
    # Abstract method to build a prompt based on a prompt syntax, input data, retrieved data, and whether an answer is required.
    @abstractmethod
    def build(
        self,
        prompt_syntax: PromptSyntax,
        input: dict,
        retrieved: List[dict],
        answer: bool,
    ) -> str:
        pass


# Base class for data encoders, which include the logic to encode a dataset into prompts.
class DataEncoder(ABC):
    # Method to encode the system prompt and associated prompts for the given dataset, retriever, prompt builder, etc.
    def encode(
        self,
        source: Dataset,
        retriever: Retriever,
        prompt_builder: TaskPromptBuilder,
        prompt_syntax: PromptSyntax,
        system_prompt: str,
    ) -> Dict[str, str | List[str]]:
        return {
            "system": system_prompt,
            "prompts": self.build_prompt(
                source,
                retriever,
                prompt_builder,
                prompt_syntax,
            ),
        }

    # Abstract method to build prompts for the dataset, retriever, prompt builder, and prompt syntax.
    @abstractmethod
    def build_prompt(
        source: Dataset,
        retriever: Retriever,
        prompt_builder: TaskPromptBuilder,
        prompt_syntax: PromptSyntax,
    ) -> List[str]:
        pass
