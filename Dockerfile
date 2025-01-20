FROM python:3.10

# =====================
# User stuff
# ---------------------

RUN apt-get update \
 && apt-get install -y \
    curl \
    dumb-init \
    htop \
    locales \
    man \
    nano \
    git \
    procps \
    ssh \
    sudo \
    vim \
    graphviz \
  && rm -rf /var/lib/apt/lists/*
RUN sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen \
  && locale-gen
ENV LANG=en_US.UTF-8
RUN chsh -s /bin/bash
ENV SHELL=/bin/bash
RUN adduser --gecos '' --disabled-password coder && \
  echo "coder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd
USER coder

# =====================
# User stuff
# ---------------------

RUN curl -fsSL https://starship.rs/install.sh > ~/starship.sh
RUN sh ~/starship.sh --yes
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc
RUN rm ~/starship.sh

RUN pip install -U black

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /home/coder/huggingface_data/
ENV HF_HOME="/home/coder/huggingface_data/"

WORKDIR /home/coder/odesia