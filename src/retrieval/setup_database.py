# retrieval/setup_database.py
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from langchain_weaviate.vectorstores import WeaviateVectorStore
from src.retrieval.schema import schema
import contextlib

@contextlib.contextmanager
def get_weaviate_client():
    connection_params = ConnectionParams.from_params(
        http_host="127.0.0.1",
        http_port=8080,
        grpc_host="127.0.0.1",
        grpc_port=50051,
        http_secure=False,
        grpc_secure=False,
    )
    client = WeaviateClient(connection_params=connection_params)
    client.connect()
    try:
        yield client
    finally:
        client.close()


def get_vectorstore(
    client: WeaviateClient, embedding_function, task_id: str = "task_foo"
) -> WeaviateVectorStore:
    # Create or update the schema
    if not client.collections.exists(task_id):
        client.collections.create(name=task_id, **schema)

    return WeaviateVectorStore(
        client=client,
        index_name=task_id,
        text_key="text",
        embedding=embedding_function,
        attributes=["content"],
    )
