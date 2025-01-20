from weaviate.classes.config import Configure, Property, DataType

COLLECTION_NAME = "TaskExamples"

schema = {
    "name": COLLECTION_NAME,
    "vectorizer_config": Configure.Vectorizer.text2vec_openai(),
    "properties": [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="task_id", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),  # JSON-serialized content
        Property(name="metadata", data_type=DataType.OBJECT)  # Additional metadata
    ],
    "vectorIndexConfig": {
        "distance": "cosine",
        "ef": 100,  # Balance between search speed and accuracy
        "maxConnections": 64,
        "dynamicEfMin": 100,
        "dynamicEfMax": 500
    }
}
