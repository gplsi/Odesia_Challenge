import ollama
from langchain_ollama import ChatOllama
import base64
from ollama import Client
from dotenv import load_dotenv
import os
import argparse
from src.data.base import Dataset, DataEncoder
from src.data.tasks import TASK_CONFIG
from src.data.tasks import *


load_dotenv()  # Loads variables from .env into environment

TASK_CONFIG = {
    "diann_2023_t1": {'class_builder': Diann2023T1PromptBuilderBIO},
}

def get_client(server: str, username: str, password: str) -> Client:

    print('Initialising ollama client...')

    auth_string = f'{username}:{password}'
    auth_base64 = base64.b64encode(auth_string.encode()).decode()
    headers = {
        "Authorization": f'Basic {auth_base64}'
    }
    return Client(host =server, headers =headers)

def main(args):
    dataset = args.dataset




    ollama_client = get_client(os.getenv("OLLAMA_SERVER"), os.getenv("OLLAMA_USERNAME"), os.getenv("OLLAMA_PASSWORD"))
    # Call language model


    # Eval
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language chain pipeline')
    parser.add_argument('dataset', type=str, help='Dataset file')
    parser.add_argument('partition', type=str, help='Partition file')
    args = parser.parse_args()
    main(args)







text = "Necesito un portatil nuevo"
messages = [
    {
        'role': 'user',
        'content': f"""A partir de ahora vas a clasificar la intención comunicativa de los mensajes que te voy a enviar.
           La intención del mensaje debe ser una de estas 13 categorías: ''informativa'', ''opinion personal'', ''elogio'', ''critica'', ''deseo'', ''peticion'', ''pregunta'', ''obligacion'', ''sugerencia'', ''sarcasmo / broma'', ''promesa'', ''amenaza'' o ''emotiva''.
           Quiero que tu respuesta sea única y solamente: entre corchetes [ ] la intención comunicativa seleccionada. Si puedes clasificar la intención responde con una cadena vacia. Mensaje: {text}""",
    },
]

ollama_model = 'llama3:instruct'
ollama_context = 6000
response = ollama_client.chat(model=ollama_model, options={'num_ctx': ollama_context}, messages=messages)
print(response['message']['content'])


