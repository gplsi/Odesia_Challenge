docker:
	docker build -f Dockerfile -t langchain_inference .

langchain_container:
	docker compose up -d langchain-inference

weaviate_container:
	docker compose up -d weaviate