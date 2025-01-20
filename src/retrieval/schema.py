from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
    VectorDistances
)

COLLECTION_NAME = "ODESIA_TASKS"
VECTOR_SIZE = 768  # Adjust based on your chosen embedding model

schema = {
    # TURN OFF Weaviate’s built-in vectorization
    "vectorizer_config": Configure.Vectorizer.none(),
    "properties": [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="task_id", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
    ],
    "vector_index_config": Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE,
        max_connections=64,
        ef_construction=128,
        dynamic_ef_min=100,
        dynamic_ef_max=500,
        vector_cache_max_objects=1000000,
    ),
}