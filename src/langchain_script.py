# your_langchain_script.py
import weaviate
from langchain_community.document_loaders import json_loader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain.vectorstores import Weaviate


client = weaviate.WeaviateClient("http://localhost:8080")

""" dataset_schema = {
    "class": "dataset",
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "author", "dataType": ["text"]},
        {"name": "date", "dataType": ["date"]}
    ]
}

client.schema.create_class(dataset_schema) """

# Configurar embeddings de Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Configurar el vector store de Weaviate
vector_store = Weaviate(client=client, index_name="Noticias", text_key="content", embeddings=embeddings)


loader = json_loader.JSONLoader(
    file_path='./dataset.json',
    jq_schema='.documents[]',
    context_key='text',
    metadata_key="metadata"
)

documents = loader.load()
vector_store.add_documents(documents)
