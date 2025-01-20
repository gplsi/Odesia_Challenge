# main.py
from embedding_service import get_embeddings
from setup_database import get_weaviate_client, get_vectorstore
from rag_service import RAGService, Example
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_task_foo_examples():
    task_id = "task_foo"
    examples = [
        Example(
                text="TASK ANOTHER GORILLA!",
                task_id =task_id,
                content={"any_json": "structure"}
            ),
            Example(
                text="TASK FOO OMG a t-rex!",
                task_id =task_id,
                content={"key": "value"}
            ),
            
            Example(
                text="ANIMALS AND DINOSAURS WOW",
                task_id =task_id,
                content={"key": "value"}
            ),
            Example(
                text="I ran until i found a dinosaur",
                task_id =task_id,
                content={"key": "value"}
            ),
    ]
    return task_id, examples
    

def get_task_foo_2_examples():
    task_id = "task_foo_2"
    examples = [
        Example(
                text="my car is big",
                task_id =task_id,
                content={"any_json": "structure"}
            ),
            Example(
                text="a truck can run faster than a bird but not as much as a dwarf t-rex",
                task_id =task_id,
                content={"key": "value"}
            ),
            
            Example(
                text="a t-rex can play dirty with a cat, but the cat is faster",
                task_id =task_id,
                content={"key": "value"}
            ),
            Example(
                text="a fish once saw a scorpio and ran away",
                task_id =task_id,
                content={"key": "value"}
            ),
    ]
    return task_id, examples

def initialize_example_data(client, embeddings):
    task_id, examples = get_task_foo_examples()
    vectorstore = get_vectorstore(client, embeddings, task_id)
    rag_service = RAGService(vectorstore, embedding_function=embeddings)
    rag_service.add_examples(examples)
    
    task_id, examples = get_task_foo_2_examples()
    vectorstore = get_vectorstore(client, embeddings, task_id)
    rag_service = RAGService(vectorstore, embedding_function=embeddings)
    rag_service.add_examples(examples)

def main():
    # Initialize components
    with get_weaviate_client() as client:
        client.collections.delete_all()
        embeddings = get_embeddings("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        
        initialize_example_data(client, embeddings)
        
        task_id = "task_foo_2"
        # using the vectorstore for no memory leaks
        vectorstore = get_vectorstore(client, embeddings, task_id)
        rag_service = RAGService(vectorstore, embedding_function=embeddings)
        
        # Retrieve examples
        results, stats = rag_service.retrieve_examples(
            query_text="T-Rex!! What a surprise!",
            k=2
        )
        
        logger.info(f"Retrieved {len(results)} results:")
        for result in results:
            logger.info(f"Text: {result['text']}")
            logger.info(f"Similarity: {result['similarity']}")
            logger.info(f"Content: {result['content']}\n")

if __name__ == "__main__":
    main()
