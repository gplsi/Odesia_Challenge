# Imagen base de PyTorch optimizada para GPU
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated git curl wget build-essential unzip && \
    rm -rf /var/lib/apt/lists/*

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

# Copy requirements earlier to leverage cache
COPY requirements.txt /tmp/requirements.txt

# Actualizar pip e instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

    # Provide defaults for Weaviate access
ENV WEAVIATE_HOST=weaviate
ENV WEAVIATE_PORT=8080

# Crear directorio de trabajo
WORKDIR /workspace

COPY . /workspace/
# Copiar tu script de LangChain (opcional si vas a montarlo como volumen)
# COPY langchain_script.py /workspace/

# # (Opcional) Copiar un script de API
# COPY inference_api.py /workspace/inference_api.py

# Exponer puertos, por ejemplo 8000 si vas a servir tu API con FastAPI
EXPOSE 8000

# Comando por defecto: podrías lanzar un script de Python o un servidor
CMD ["/bin/bash"]