from abc import ABC, abstractmethod
import json
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

    def process_ner(self, outputs: dict, task_name: str, language: str, partition: str):
        task_name = task_name.lower()
        output_file = f"{task_name}_{language}_{partition}.json"
        results = []

        for output in outputs:
            text=output["out"]
            ids=output["id"]
            text=text[0]['generated_text'][2]['content']
            text_processed = PostProcessingImplementation.extract_and_convert_to_list(text)

            result = {
                "id": str(ids),
                "test_case": "DIANN2023",
                "value": text_processed
            }
            results.append(result)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    def process_classification(self, outputs: dict, task_name: str, language: str, partition: str):
        results = []
        for output in outputs:
            task_name = task_name.lower()
            output_file = f"{task_name}_{language}_{partition}.json"
            text=output["out"]
            ids=output["id"]
            
            if task_name == "exist_2023_t1":
                text_processed=PostProcessingImplementation.find_last_class(text, CLASSES_EXIST_2023_T1)
                test_case="EXIST2023"
            
            if task_name == "exist_2023_t2":
                text_processed=PostProcessingImplementation.find_last_class(text, CLASSES_EXIST_2023_T2)
                test_case="EXIST2023"
            
            if task_name == "exist_2023_t3":
                text_processed=PostProcessingImplementation.find_last_class(text, CLASSES_EXIST_2023_T3)
                test_case="EXIST2023"
            
            if task_name == "exist_2022_t1":
                text_processed=PostProcessingImplementation.find_last_class(text, CLASSES_EXIST_2022_T1)
                test_case="EXIST2022"
            
            if task_name == "exist_2022_t2":
                text_processed=PostProcessingImplementation.find_last_class(text, CLASSES_EXIST_2022_T2)
                test_case="EXIST2022"
            
            if task_name == "dipromats_2023_t1":
                text=text[0]['generated_text'][2]['content']
                text_processed=PostProcessingImplementation.find_last_class(text, CLASSES_DIPROMATS_2023_T1)
                if not text_processed:
                    text_processed="false"
                
                test_case="DIPROMATS2023"
            
            if task_name == "dipromats_2023_t2":
                # print("text: ",text)
                text=text[0]['generated_text'][2]['content']
                if text!="false":
                    print("text: ",text)
                text_processed=PostProcessingImplementation.find_classes_and_convert_to_list(text, CLASSES_DIPROMATS_2023_T2)
                test_case="DIPROMATS2023"
            
            if task_name == "dipromats_2023_t3":
                text_processed=PostProcessingImplementation.find_classes_and_convert_to_list(text, CLASSES_DIPROMATS_2023_T3)
                test_case="DIPROMATS2023"
            result = {
                    "test_case": test_case,
                    "id": str(ids),
                    "value": text_processed
                }
            results.append(result)                  

        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

    def process_qa(self, outputs: dict, task_name: str, language: str, partition: str):
        task_name = task_name.lower()
        output_file = f"{task_name}_{language}_{partition}.json"
        results = []

        for output in outputs:
            text=output["out"]
            ids=output["id"]
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
    
    def extract_and_convert_to_list(text):
        try:
            # Buscar el índice del primer y último corchete
            start_index = text.find('[')
            end_index = text.rfind(']')

            if start_index == -1:
                raise ValueError("No se encontró ningún '[' en el texto.")
            
            if text.strip().endswith("'"):
                text = text.strip()[:-1]

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
                raise ValueError("El contenido extraído no es una lista.")
            
            return result_list
        except (SyntaxError, ValueError) as e:
            print(f"Error al procesar el texto: {e}")
            return None


    def find_classes_and_convert_to_list(text,classes):
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
        
        # Si "false" está en la lista de clases y es la única opción, devolver ["false"]
        if "false" in classes and len(classes) == 1:
            return ["false"]

        # Evaluar si cada clase está presente en el texto
        detected_classes = [cls for cls in classes if cls in text]
        
        # Si no se encuentra ninguna clase, devolver ["false"]
        return detected_classes if detected_classes else ["false"]



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
