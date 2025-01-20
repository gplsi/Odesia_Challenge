# setup_database.py
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.init import AdditionalConfig, Timeout, Auth
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure

def create_weaviate_client(huggingface_api_key: str):
    """
    Create a WeaviateClient (v4) pointing to your Weaviate deployment.
    Note: We do not pass 'url' directly; instead we use ConnectionParams(endpoint=...).
    """
    connection_params=ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    ),
    client = WeaviateClient(
        connection_params=connection_params[0],
        additional_headers={
            "X-HuggingFace-Api-Key": huggingface_api_key
        },
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120),  # Values in seconds
        ),
        skip_init_checks=False
        # embedded_options=EmbeddedOptions()  # Only if you need embedded mode
    )
    client.connect()
    return client

def setup_collection(client: WeaviateClient, schema: dict):
    """
    Drops the collection if it exists, then creates it based on the provided schema.
    """
    if client.collections.exists(schema["name"]):
        client.collections.delete(schema["name"])
    
    collection = client.collections.create(
        name=schema["name"],
        vectorizer_config=schema["vectorizer_config"],
        properties=schema["properties"],
        vector_index_config=schema["vector_index_config"],
    )
    return collection
