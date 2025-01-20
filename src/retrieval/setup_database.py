# setup_database.py
import weaviate
from weaviate.embedded import EmbeddedOptions

def create_weaviate_client(huggingface_api_key: str):
    client = weaviate.Client(
        embedded_options=EmbeddedOptions(),
        additional_headers={
            "X-HuggingFace-Api-Key": huggingface_api_key
        }
    )
    return client

def setup_collection(client: weaviate.Client, schema: dict):
    if client.collections.exists(schema["name"]):
        client.collections.delete(schema["name"])
    
    collection = client.collections.create(
        name=schema["name"],
        vectorizer_config=schema["vectorizer_config"],
        properties=schema["properties"],
        vector_index_config=schema["vectorIndexConfig"]
    )
    return collection
