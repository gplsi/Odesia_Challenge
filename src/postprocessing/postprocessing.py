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
            #PostProcessingImplementation._extract_entities(token_list, text)

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

    @staticmethod
    def _extract_entities(tokens: List[str], text: str):
        result_list = ["O"] * len(tokens)

        # Trim any leading or trailing whitespace
        text = text.strip()
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("'", "")

        # Split the text by commas
        outputed_tokens = text.split(",")
        
        response_token_dict = dict()
        for t in outputed_tokens:
            splitted_tokens = [y for y in t.split(" ") if y != "" and y != " "]  # Split by spaces
            if splitted_tokens:
                response_token_dict[splitted_tokens[-1]] = splitted_tokens
                
        # Iterate over the outputed tokens and replace the tokens in the result list
        for i in range(len(tokens)):
            token = tokens[i]
            if token in response_token_dict.keys():
                span = response_token_dict[token]
                
                all_match = True
                # check if the tokens from before match the full detected span
                for j in range(len(span)):
                    token_i = i - len(span) + j + 1
                    if token_i < 0:
                        all_match = False
                        break
                    
                    if tokens[token_i] != span[j]:
                        all_match = False
                        break
                    
                if all_match:
                    prev_label = "O"
                    for j in range(len(span)):
                        token_i = i - len(span) + j + 1
                        label = "B-DIS" if prev_label == "O" else "I-DIS"
                        result_list[token_i] = label
                        prev_label = label
                        
        
        prev_label = "O"
        processed_result = []
        for result_label in result_list:
            if result_label == "B-DIS":
                # fix the B token
                if prev_label == 'O':
                    result_label = "B-DIS"
                else:
                    result_label = "I-DIS"
                    
            prev_label = result_label
            processed_result.append(result_label)
            
        return processed_result

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


    def _extract_entities_JP(tokens: List[str], text: str):
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
        pointer = 0
        for i in range(len(tokens)):
            if pointer >= len(outputed_tokens):
                break

            token_i = tokens[i]

            if token_i == outputed_tokens[pointer]:
                result_list[i] = (
                    "B-DIS" if i == 0 or result_list[i - 1] == "O" else "I-DIS"
                )
                pointer += 1
                continue

        return result_list


    def _test_extract_entities(input_tokens: List[str], output_text: str):
        """
        input_tokens: List[str]
            List of the original input tokens to the model.
            i.e. ["This", "is", "a", "test", "text", "with", "CRONIC", "DISABILITY", ...]
        
        output_text: str
            The output text from the model. It might have errors and need to be formatted to a list.
            i.e. "['CRONIC DISABILITY', 'Alzheimer' ,']"
        """
        
        result_list = ["O"] * len(input_tokens)

        # --- Preprocessing of output_text ---
        # Note: This preprocessing is brittle and may need improvements in a real-world scenario.
        output_text = output_text.strip()
        output_text = output_text.replace("[", "")
        output_text = output_text.replace("]", "")
        output_text = output_text.replace("'", "")
        
        # Split the text by commas
        outputed_tokens = output_text.split(",")
        
        # --- Build a robust dictionary of candidate spans ---
        # Instead of a dictionary that maps the last token to a single span, we map it to a list of spans.
        # This avoids overwriting if multiple entity spans end with the same token.
        response_token_dict = {}
        for current_tokens in outputed_tokens:
            # Split by any whitespace and filter out empty tokens
            splitted_tokens = [token for token in current_tokens.split() if token]
            if splitted_tokens:
                last_token = splitted_tokens[-1]
                response_token_dict.setdefault(last_token, []).append(splitted_tokens)
        
        # --- Iterate over the input tokens and substitute labels if a span is matched ---
        # When matching, convert input tokens to lowercase for comparison
        for i, token in enumerate(input_tokens):
            lower_token = token.lower()
            if lower_token in response_token_dict:
                candidate_spans = response_token_dict[lower_token]
                for span in candidate_spans:
                    span_length = len(span)
                    start_index = i - span_length + 1
                    # Ensure there are enough tokens before the current token
                    if start_index < 0:
                        continue
                    # Check if the candidate span matches the tokens in input_tokens in lowercase
                    if [t.lower() for t in input_tokens[start_index:i+1]] == span:
                        # Label the tokens: the first token gets "B-DIS", and the rest get "I-DIS"
                        result_list[start_index] = "B-DIS"
                        for k in range(1, span_length):
                            result_list[start_index + k] = "I-DIS"
                        # Once a match is found for this candidate, no need to check further candidates
                        break
        return result_list

# Ejemplo de uso
if __name__ == "__main__":
    text_example_ner = ["O", "O", "B-DIS", "I-DIS", "O"]
    text_example_classification = (
        "This is a test text with CLASS_A and CLASS_B appearing."
    )
    text_example_qa = "The answer to the question is this."

    processor = PostProcessingImplementation()

    tokens = ["This", "is", "a", "test", "text", "with", "CRONIC", "DISABILITY", "and", "another", "CRONIC", "thing", "appearing", "DISABILITY"]
    output = "['CRONIC DISABILITY', 'a test']"

    print("here")
    # Procesamiento de NER
    result = PostProcessingImplementation._extract_entities(tokens, output)
    print(result)
