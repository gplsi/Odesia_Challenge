docker:
	docker build -f Dockerfile -t langchain_inference .

langchain_container:
	docker-compose up -d langchain-inference

weaviate_container: docker
	docker-compose up -d weaviate

run_weaviate:
	docker run -d \
		--name weaviate \
		-p 8080:8080 \
		-p 50051:50051 \
		-e QUERY_DEFAULTS_LIMIT=25 \
		-e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
		-e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
		-e CLUSTER_HOSTNAME=node1 \
		--gpus device=3\
		-v /raid/gplsi/robiert/docker_vol/Odesia_Challenge/weaviate:/var/lib/weaviate \
		--network="host" \
		semitechnologies/weaviate:latest

run_langchain_inference:
	docker run -d \
		--name langchain_inference \
		-p 8001:8001 \
		--gpus all \
		-v ./:/workspace/ \
		--network="host" \
		langchain_inference tail -f /dev/null

main: run_weaviate run_langchain_inference
	docker exec -it langchain_inference bash