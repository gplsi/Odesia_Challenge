# your_langchain_script.py
import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings

# Conectar a Weaviate (usando la red interna de Docker Compose)
client = weaviate.Client(url="http://weaviate:8080")

# Configurar embeddings de Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Crear/usar un vector store en Weaviate
vector_store = Weaviate(
    client=client,
    index_name="Documents",     # Nombre de la "clase" en Weaviate
    text_key="content",         # Nombre del campo de texto
    embedding_function=embeddings.embed_query,
)

# Insertar documentos
docs = ["Hola mundo", "Weaviate es una base de datos vectorial", "LangChain facilita RAG"]
vector_store.add_texts(docs)

# Hacer una búsqueda
query = "¿Qué es LangChain?"
retriever = vector_store.as_retriever()
results = retriever.get_relevant_documents(query)

print("Resultados de la búsqueda:\n")
for doc in results:
    print(doc.page_content)

