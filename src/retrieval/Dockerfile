# Imagen base de PyTorch optimizada para GPU
FROM nvcr.io/nvidia/pytorch:23.02-py3

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar herramientas del sistema
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# (Opcional) Instalar Docker CLI si quieres manipular contenedores desde aquí
# RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
#     sh get-docker.sh && \
#     rm get-docker.sh

# Actualizar pip e instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install \
        langchain \
        langchain-community \
        sentence-transformers \
        weaviate-client \
        transformers \
        datasets \
        accelerate \
        faiss-gpu \
        python-dotenv \
        tiktoken \
        fastapi \
        uvicorn[standard] \
        gradio

# Crear directorio de trabajo
WORKDIR /workspace

# Copiar tu script de LangChain (opcional si vas a montarlo como volumen)
COPY langchain_script.py /workspace/

# (Opcional) Copiar un script de API
COPY inference_api.py /workspace/inference_api.py

# Exponer puertos, por ejemplo 8000 si vas a servir tu API con FastAPI
EXPOSE 8000

# Comando por defecto: podrías lanzar un script de Python o un servidor
CMD ["/bin/bash"]
