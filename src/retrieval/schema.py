from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
)

COLLECTION_NAME = "ODESIA_TASKS"
VECTOR_SIZE = 384  # Adjust based on your chosen embedding model

schema = {
    "name": COLLECTION_NAME,
    "vectorizer_config": Configure.Vectorizer.none(),  # We'll handle vectorization ourselves
    "properties": [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="task_id", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="vector", data_type=DataType.VECTOR, vector_size=VECTOR_SIZE),
    ],
    "vector_index_config": Configure.VectorIndex.hnsw(
        quantizer=Configure.VectorIndex.Quantizer.pq(
            training_limit=10000
        )
    )
}