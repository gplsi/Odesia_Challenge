from abc import ABC, abstractmethod
import json
import uuid
import os

class PostProcessing(ABC):
    
    @abstractmethod
    def process_ner(self, text: str, task_name: str, language: str, start_id: int):
        """Postprocesa la salida del modelo para tareas de NER."""
        pass

    @abstractmethod
    def process_classification(self, text: str, classes: list, task_name: str, language: str, start_id: int):
        """Postprocesa la salida del modelo para tareas de clasificación."""
        pass

    @abstractmethod
    def process_qa(self, text: str, task_name: str, language: str, start_id: int):
        """Postprocesa la salida del modelo para tareas de QA."""
        pass

class PostProcessingImplementation(PostProcessing):

    def process_ner(self, text: str, task_name: str, language: str, ids: int, partition: str):
        task_name = task_name.lower()
        output_file = f"./outputs/{task_name[:-3]}/{partition}_{task_name[-3:]}_{language}.json"
        # output_file = f"{task_name}_{language}.json"
        results = []

        text_processed = text

        result = {
            "id": str(ids),
            "test_case": "DIANN2023",
            "value": text_processed
        }
        results.append(result)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    def process_classification(self, outputs: str, classes: list, task_name: str, language: str, partition: str):
        results = []
        for output in outputs:
            task_name = task_name.lower()
            # output_file = f"./outputs/{task_name[:-3]}/{partition}_{task_name[-3:]}_{language}.json"
            output_file = f"{task_name}_{language}.json"
            text=output["out"]
            ids=output["id"]
            if task_name == "exist_2023_t1":
                text_processed=text
                test_case="EXIST2023"
            if task_name == "exist_2023_t2":
                text_processed=text
                test_case="EXIST2023"
            if task_name == "exist_2023_t3":
                text_processed=text
                test_case="EXIST2023"
            if task_name == "exist_2022_t1":
                text_processed=PostProcessingImplementation.find_last_class(text, classes)
                test_case="EXIST2022"
            if task_name == "exist_2022_t2":
                text_processed=PostProcessingImplementation.find_last_class(text, classes)
                test_case="EXIST2022"
            if task_name == "dipromats_2023_t1":
                text=text[0]['generated_text'][2]['content']
                text_processed=PostProcessingImplementation.find_last_class(text, classes)
                if not text_processed:
                    text_processed="false"
                test_case="DIPROMATS2023"
            if task_name == "dipromats_2023_t2":
                text_processed=PostProcessingImplementation.find_last_class(text, classes)
                test_case="DIPROMATS2023"
            if task_name == "dipromats_2023_t3":
                text_processed=PostProcessingImplementation.find_last_class(text, classes)
                test_case="DIPROMATS2023"
            result = {
                    "test_case": test_case,
                    "id": str(ids),
                    "value": text_processed
                }
            results.append(result)                  

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    def process_qa(self, text: str, task_name: str, language: str, ids: int, partition: str):
        task_name = task_name.lower()
        output_file = f"./outputs/{task_name[:-3]}/{partition}_{task_name[-3:]}_{language}.json"
        # output_file = f"{task_name}_{language}.json"
        results = []

        text_processed=text

        result = {
            "id": str(ids),
            "test_case": "SQAC_SQUAD_2024",
            "value": text_processed
        }
        results.append(result)

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

        return None  # Si no se encuentra ninguna clase

# Ejemplo de uso
if __name__ == "__main__":
    text_example_ner = ["O", "O", "B-DIS", "I-DIS", "O"]
    text_example_classification = "This is a test text with CLASS_A and CLASS_B appearing."
    text_example_qa = "The answer to the question is this."

    processor = PostProcessingImplementation()

    # Procesamiento de NER
    processor.process_ner(text_example_ner, "DIANN_2023_T1", "es", 100)

    # Procesamiento de Clasificación
    processor.process_classification(text_example_classification, ["CLASS_A", "CLASS_B", "CLASS_C"], "EXIST_2023_T1", "en", 500)

    # Procesamiento de QA
    processor.process_qa(text_example_qa, "SQAC_SQUAD_2024", "en", 1)
