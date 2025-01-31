from abc import ABC, abstractmethod
import json
from typing import List
import uuid
import os
import ast
from src.data.config import (
    CLASSES_DIPROMATS_2023_T1,
    CLASSES_DIPROMATS_2023_T2,
    CLASSES_DIPROMATS_2023_T3,
    CLASSES_EXIST_2022_T1,
    CLASSES_EXIST_2022_T2,
    CLASSES_EXIST_2023_T1,
    CLASSES_EXIST_2023_T2,
    CLASSES_EXIST_2023_T3,
)
from src.utils import post_processing_error_handler
import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt", quiet=True)


class PostProcessing(ABC):
    @abstractmethod
    def process_ner(
        self, text: str, task_name: str, language: str, start_id: int, shot_count: int
    ):
        """Postprocesa la salida del modelo para tareas de NER."""
        pass

    @abstractmethod
    def process_classification(
        self, text: str, classes: list, task_name: str, language: str, start_id: int
    ):
        """Postprocesa la salida del modelo para tareas de clasificación."""
        pass

    @abstractmethod
    def process_qa(self, text: str, task_name: str, language: str, start_id: int):
        """Postprocesa la salida del modelo para tareas de QA."""
        pass


class PostProcessingImplementation(PostProcessing):

    def process_ner(
        self,
        tokens: List[List[str]],
        outputs: List,
        task_name: str,
        language: str,
        partition: str,
        shot_count: int = 5,
        use_bio_format: bool = True,
        tag: str='.',
    ):
        task_name = task_name.lower()
        if not os.path.exists(f"data/{tag}/results_{partition}"):
            os.makedirs(f"data/{tag}/results_{partition}")
        output_file = f"data/{tag}/results_{partition}/{task_name}_{language}_{partition}_{shot_count}.json"
        results = []

        for i in range(len(outputs)):
            token_list = tokens[i]
            output = outputs[i]
            text = output["out"]
            ids = output["id"]
            text = text[0]["generated_text"][2]["content"]

            text_processed = (
                PostProcessingImplementation._extract_and_convert_to_list(
                    token_list, text
                )
                if use_bio_format
                else PostProcessingImplementation._extract_entities(token_list, text)
            )

            result = {"id": str(ids), "test_case": "DIANN2023", "value": text_processed}
            results.append(result)

        os.makedirs("data/results", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        return results

    def process_classification(
        self,
        outputs: dict,
        task_name: str,
        language: str,
        partition: str,
        shot_count: int,
        tag: str='.'
    ):
        if not os.path.exists(f"data/{tag}/results_{partition}"):
            os.makedirs(f"data/{tag}/results_{partition}")

        results = []
        for output in outputs:
            task_name = task_name.lower()
            output_file = f"data/{tag}/results_{partition}/{task_name}_{language}_{partition}_{shot_count}.json"
            text = output["out"][0]["generated_text"][2]["content"]
            ids = output["id"]
            try:
                if task_name == "exist_2023_t1":
                    text_processed = (
                        PostProcessingImplementation.find_classes_and_convert_to_dict(
                            text, CLASSES_EXIST_2023_T1
                        )
                    )
                    test_case = "EXIST2023"

                if task_name == "exist_2023_t2":
                    text_processed = (
                        PostProcessingImplementation.find_classes_and_convert_to_dict(
                            text, CLASSES_EXIST_2023_T2
                        )
                    )
                    test_case = "EXIST2023"

                if task_name == "exist_2023_t3":
                    text_processed = (
                        PostProcessingImplementation.find_classes_and_convert_to_dict(
                            text, CLASSES_EXIST_2023_T3
                        )
                    )
                    test_case = "EXIST2023"

                if task_name == "exist_2022_t1":
                    text_processed = PostProcessingImplementation.find_last_class(
                        text, CLASSES_EXIST_2022_T1
                    )
                    test_case = "EXIST2022"

                if task_name == "exist_2022_t2":
                    text_processed = PostProcessingImplementation.find_last_class(
                        text, CLASSES_EXIST_2022_T2
                    )
                    test_case = "EXIST2022"

                if task_name == "dipromats_2023_t1":
                    text_processed = PostProcessingImplementation.find_last_class(
                        text, CLASSES_DIPROMATS_2023_T1
                    )
                    if not text_processed:
                        text_processed = "false"

                    test_case = "DIPROMATS2023"

                if task_name == "dipromats_2023_t2":
                    text_processed = (
                        PostProcessingImplementation.find_classes_and_convert_to_list(
                            text, CLASSES_DIPROMATS_2023_T2
                        )
                    )
                    test_case = "DIPROMATS2023"

                if task_name == "dipromats_2023_t3":
                    text_processed = (
                        PostProcessingImplementation.find_classes_and_convert_to_list(
                            text, CLASSES_DIPROMATS_2023_T3
                        )
                    )
                    test_case = "DIPROMATS2023"
                result = {
                    "test_case": test_case,
                    "id": str(ids),
                    "value": text_processed,
                }
                results.append(result)

            except Exception as e:
                post_processing_error_handler(
                    e, text, ids, task_name, language, partition
                )
                continue

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    def process_qa(
        self,
        outputs: dict,
        task_name: str,
        language: str,
        partition: str,
        shot_count: int,
        tag: str='.'
    ):
        if not os.path.exists(f"data/{tag}/results_{partition}"):
            os.makedirs(f"data/{tag}/results_{partition}")

        task_name = task_name.lower()
        output_file = f"data/{tag}/results_{partition}/{task_name}_{language}_{partition}_{shot_count}.json"
        results = []

        for output in outputs:
            text = output["out"][0]["generated_text"][2]["content"]
            ids = output["id"]
            text_processed = text

            try:
                result = {
                    "id": str(ids),
                    "test_case": "SQAC_SQUAD_2024",
                    "value": text_processed,
                }
                results.append(result)
            except Exception as e:
                post_processing_error_handler(
                    e, text, ids, task_name, language, partition
                )
                continue

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    def find_last_class(text, classes):
        """
        Encuentra el primer valor en la lista `classes` recorriendo el texto desde el final hacia el principio.

        :param text: Texto a analizar.
        :param classes: Lista de valores a buscar en el texto.
        :return: El primer valor encontrado desde el final o None si no se encuentra.
        """
        # Convertimos el texto en una lista de palabras manteniendo el orden inverso
        reversed_words = text.split()[::-1]

        # Recorremos las palabras en orden inverso buscando el primer match
        for word in reversed_words:
            if word in classes:
                return word

        return classes[0]  # Si no se encuentra ninguna clase

    """ def extract_and_convert_to_list(text, ids):
        try:
            # Buscar el índice del primer y último corchete
            start_index = text.find('[')
            end_index = text.rfind(']')

            if start_index == -1:
                post_processing_error_handler(ValueError("No se encontró ningún '[' en el texto."), text, ids, "DIANN_2023_T1")
                raise ValueError("No se encontró ningún '[' en el texto.")
            
            # Fix potential unterminated string element (add missing closing single quote)
            if text.strip()[-1] != "'" and text.strip()[-2:] != "']":
                # Add the missing single quote only if necessary
                end_bracket_pos = text.rfind("]")
                if end_bracket_pos != -1:
                    text = text[:end_bracket_pos] + "'" + text[end_bracket_pos:]
                else:
                    text = text.strip() + "'"

            # Si no hay un ] al final, lo añade
            if end_index == -1 or end_index < start_index:
                end_index = len(text)
                text = text[:end_index] + ']'

            # Extraer el contenido desde el primer [ hasta el último ]
            list_str = text[start_index:end_index + 1]

            # Usar ast.literal_eval para convertir el string en una lista
            result_list = ast.literal_eval(list_str)
            
            # Validar que el resultado sea una lista
            if not isinstance(result_list, list):
                post_processing_error_handler(ValueError("El contenido extraído no es una lista."), text, ids, "DIANN_2023_T1")
                raise ValueError("El contenido extraído no es una lista.")
            
            return result_list
        except (SyntaxError, ValueError) as e:
            post_processing_error_handler(e, text, ids, "DIANN_2023_T1")
            print(f"Error al procesar el texto: {e}")
            return None """

    def _extract_and_convert_to_list(tokens: List[str], text: str):
        result_list = ["O"] * len(tokens)

        # Trim any leading or trailing whitespace
        text = text.strip()
        text = text.replace(" ", "")
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("'", "")

        # Split the text by commas
        outputed_tokens = text.split(",")

        # Iterate over the outputed tokens and replace the tokens in the result list
        for i in range(len(outputed_tokens)):
            if i >= len(result_list):
                break

            result_token_i = outputed_tokens[i]
            if result_token_i.startswith("B-") and result_token_i != "B-DIS":
                result_token_i = "B-DIS"
            elif result_token_i.startswith("I-") and result_token_i != "I-DIS":
                result_token_i = "I-DIS"
            if result_token_i == "":
                continue

            result_list[i] = result_token_i

        # Ensure the result is a list
        if not isinstance(result_list, list):
            raise ValueError("El contenido corregido no es una lista válida.")
        return result_list

    def _extract_entities(tokens: List[str], text: str):
        result_list = ["O"] * len(tokens)

        # Trim any leading or trailing whitespace
        text = text.strip()
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("'", "")

        # Split the text by commas
        outputed_tokens = text.split(",")
        outputed_tokens = [y for t in outputed_tokens for y in t.split(" ") if y != "" and y != " "]  # Split by spaces

        # Iterate over the outputed tokens and replace the tokens in the result list
        outputed_tokens_set = set(outputed_tokens)
        result = []
        prev_label = "O"
        for token in tokens:
            if token in outputed_tokens_set:
                label = "B-DIS" if prev_label == "O" else "I-DIS"
            else:
                label = "O"
            result.append(label)
            prev_label = label

        return result_list

    def find_classes_and_convert_to_list(text, classes):
        """
        Clasifica un texto dado en función de las clases proporcionadas.

        Args:
            text (str): El texto a evaluar.
            classes (list): Lista de posibles clases.

        Returns:
            list: Lista con las clases encontradas o ["false"] si la clase es "false".
        """
        # Convertir el texto a minúsculas para evitar problemas con mayúsculas/minúsculas
        text = text.lower()
        # Convertir las clases a minúsculas para que coincidan
        classes = [cls.lower() for cls in classes]

        # Evaluar si cada clase está presente en el texto
        detected_classes = [cls for cls in classes if cls in text]

        # Si no se encuentra ninguna clase, devolver ["false"]
        return detected_classes if detected_classes else ["false"]

    def find_classes_and_convert_to_dict(text, classes):
        return ast.literal_eval(text)


# Ejemplo de uso
if __name__ == "__main__":
    text_example_ner = ["O", "O", "B-DIS", "I-DIS", "O"]
    text_example_classification = (
        "This is a test text with CLASS_A and CLASS_B appearing."
    )
    text_example_qa = "The answer to the question is this."

    processor = PostProcessingImplementation()

    # Procesamiento de NER
    processor.process_ner(text_example_ner, "DIANN_2023_T1", "es", "val", shot_count=5)

    # Procesamiento de Clasificación
    processor.process_classification(
        text_example_classification,
        ["CLASS_A", "CLASS_B", "CLASS_C"],
        "EXIST_2023_T1",
        "en",
        500,
    )

    # Procesamiento de QA
    processor.process_qa(text_example_qa, "SQAC_SQUAD_2024", "en", 1)
