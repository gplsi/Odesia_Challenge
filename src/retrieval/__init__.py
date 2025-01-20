import json
from weaviate import WeaviateClient
 
from src.retrieval.embedding_service import get_embeddings
from src.retrieval.setup_database import get_weaviate_client, get_vectorstore
from src.retrieval.rag_service import RAGService, Example
from src.data.base import Dataset, Retriever


class ReRankRetriever(Retriever):
    """
    A retriever that uses a VectorStore and a reranking model to retrieve the most relevant
    examples from a stored dataset. It can optionally create a new dataset collection or
    retrieve from an existing one.

    :param dataset_id: The identifier for the dataset collection.
    :param dataset: An instance of the Dataset class containing data to be stored/retrieved.
    :param embeddings_model: Name of the embeddings model.
    :param rerank_model: Name of the reranking model.
    :param similarity_threshold: Threshold for filtering retrieved results.
    """
    def __init__(
        self,
        dataset_id: str = None,
        dataset: Dataset = None,
        embeddings_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        similarity_threshold = 0.6,
    ):
        # Instead of directly calling get_weaviate_client(), store the context manager:
        self.client_cm = get_weaviate_client()
        self.client = self.client_cm.__enter__()
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.embeddings_model = embeddings_model
        self.rerank_model = rerank_model
        self.similarity_threshold = similarity_threshold

        self._init_embeddings()
        self._init_data()

    def __del__(self):
        if getattr(self, "client_cm", None):
            self.client_cm.__exit__(None, None, None)

    def _init_embeddings(self):
        """
        Initialize the embeddings using the model name provided.
        """
        self.embeddings = get_embeddings(model_name=self.embeddings_model)

    def _init_data(self):
        """
        Setup the vector store and create a RAGService. If dataset info is available, add examples.
        """
        if self.dataset is None:
            if self.dataset_id is None:
                return
            else:
                # if dataset_id is provided but not the actual data, then we start in retrieval mode without writing new data
                # if the collection does not exist, raise an error as we do not have where to retrieve from
                if not self.client.collections.exists(self.dataset_id):
                    raise ValueError(f"Collection {self.dataset_id} does not exist")

                self.vectorstore = get_vectorstore(
                    self.client, self.embeddings, self.dataset_id
                )
                self.rag_service = RAGService(
                    self.vectorstore,
                    embedding_function=self.embeddings,
                    rerank_model=self.rerank_model,
                    similarity_threshold=self.similarity_threshold,
                )

        elif self.dataset_id is None:
            raise ValueError(
                "Dataset ID is required for adding new examples and creating the collection"
            )
        else:
            # if both dataset and dataset_id are provided, then we start in write mode
            # if the collection already exists, we overwrite the data
            if self.client.collections.exists(self.dataset_id):
                self.client.collections.delete(self.dataset_id)

            self.vectorstore = get_vectorstore(
                self.client, self.embeddings, self.dataset_id
            )
            self.rag_service = RAGService(
                self.vectorstore, embedding_function=self.embeddings
            )

            examples = [
                Example(text, self.dataset_id, json.dumps(content))
                for text, content in self.dataset.items()
            ]

            self.rag_service.add_examples(examples)

    def overwrite_data(self, dataset_id: str, dataset: Dataset):
        """
        Overwrite the existing collection data with a new dataset.
        :param dataset_id: The identifier for the new dataset collection.
        :param dataset: The new Dataset instance to store.
        """
        self.dataset_id = dataset_id
        self.dataset = dataset
        self._init_data()

    def set_retrieve_mode(self, dataset_id: str):
        """
        Switch the retriever to retrieve-only mode for the specified collection.
        :param dataset_id: The dataset collection to load for retrieval.
        """
        self.dataset_id = dataset_id
        self.dataset = None
        self._init_data()

    def retrieve(self, query: str, limit=5) -> object:
        """
        Retrieve the top documents related to the query.
        :param query: The input query string.
        :param limit: Number of documents to return.
        :return: The retrieved documents along with stats.
        """
        if self.dataset_id is None:
            raise ValueError("No dataset collection specified for retrieval")
        
        return self.rag_service.retrieve_examples(query, limit)

if __name__ == "__main__":
    # Simple test with mock data using the existing Dataset class
    mock_data = [
        {"text": "Hello world", "info": "Greeting"},
        {"text": "Hi how are you?", "info": "Greeting"},
        {"text": "How to bake a cake?", "info": "Recipe"},
        {"text": "Can we drink and eat?", "info": "Recipe"},
    ]
    
    mock_data_2 = [
        {"text": "My phone is pretty old", "info": "Phone"},
        {"text": "I dont hear you more", "info": "Phone"},
        {"text": "I want to watch TV", "info": "Cable"},
        {"text": "Turn off that thing, it is too bright", "info": "Cable"},
    ]
    
    mock_dataset = Dataset(data=mock_data, text_key="text")
    mock_dataset_2 = Dataset(data=mock_data_2, text_key="text")
    
    retriever = ReRankRetriever()
    
    retriever.overwrite_data("mock_collection", mock_dataset)
    retriever.overwrite_data("mock_collection_2", mock_dataset_2)
    
    print("Testing collection 1")
    retriever.set_retrieve_mode("mock_collection")
    results = retriever.retrieve("Recipe", limit=2)
    
    import pprint
    pprint.pprint(results)
    
    print()
    print("Testing collection 2")
    retriever.set_retrieve_mode("mock_collection_2")
    results = retriever.retrieve("Television", limit=2)
    
    import pprint
    pprint.pprint(results)
    print()
    
