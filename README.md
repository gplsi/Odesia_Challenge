# Odesia Challenge - GPLSI Experiments

![License](https://img.shields. io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)
![CUDA](https://img.shields. io/badge/CUDA-12.1+-green.svg)

## 🚩 Purpose of This Repository

This repository is dedicated to the **Odesia Challenge**, an initiative that benchmarks language technologies across 10 distinct NLP tasks in Spanish. Developed and maintained by the **GPLSI group** (Grupo de Procesamiento del Lenguaje y Sistemas de la Información), the goal is to build and evaluate a **unified system capable of handling all these tasks efficiently**, using advanced machine learning strategies that do **not require separate architectures for each task**.

Rather than approaching each task as pure generation, the challenge allows solutions via **token classification**, enabling decoder-only, encoder-only, or encoder-decoder models. The GPLSI experiments documented and implemented here specifically focus on **decoder-only language models**, optimising their performance across tasks using:

- **Fine-tuning** on specific tasks
- **Zero-shot learning** without task-specific training
- **Retrieval-Augmented Generation (RAG)** with hybrid search capabilities

### The Core Question

> *"How far can decoder-only language models go when solving varied NLP tasks in Spanish, using strategic adaptation and retrieval techniques?"*

This repository documents the GPLSI group's comprehensive answer to this question through rigorous experimentation, systematic comparison, and production-ready implementation.

---

## 📋 Table of Contents

- [What is the Odesia Challenge?](#what-is-the-odesia-challenge)
- [Why This Repository Exists](#why-this-repository-exists)
- [Our Experimental Approaches](#our-experimental-approaches)
- [Repository Structure](#repository-structure)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Experimentation Workflows](#experimentation-workflows)
- [Results & Performance](#results--performance)
- [Technologies & Stack](#technologies--stack)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## 🏆 What is the Odesia Challenge?

The **Odesia Challenge** is a comprehensive Spanish NLP benchmark initiative designed to evaluate system performance across **10 diverse linguistic tasks** within a single, unified framework.

### Challenge Characteristics

- **10 Spanish NLP Tasks:** Diverse linguistic phenomena covering multiple NLP subtasks
- **Single System Requirement:** Build ONE unified system instead of separate models per task
- **Flexible Solution Approaches:** Tasks can be solved through generation, token classification, or sequence labeling, it can include complex systems with RAG and agents as well.
- **Practical Focus:** Emphasis on real-world applicability and computational efficiency
- **Multi-model Support:** Encoder, decoder, and encoder-decoder architectures welcome

### Why It Matters

The Odesia Challenge addresses a critical gap in NLP research:

> **"Can modern language models obtain a good performance across different competitive tasks in Spanish?"**

Spanish is a language with rich morphosyntactic complexity, regional variations, and specific linguistic phenomena.  The challenge bridges the gap between academic language understanding and practical, deployable NLP systems.

---

## 📖 Why This Repository Exists

### Primary Objectives

1. **Benchmark Development:** Create reference implementations for handling multiple Spanish NLP tasks
2. **Strategy Comparison:** Systematically compare fine-tuning, zero-shot, and RAG-based approaches
3. **Model Optimization:** Optimize decoder-only models for multi-task Spanish NLP
4. **Reproducibility:** Provide fully documented, reproducible experiments with version-controlled configurations
5. **Production Deployment:** Enable easy deployment and serving of models via containerized infrastructure
6. **Knowledge Sharing:** Document findings and best practices for the community

### What You'll Find Here

✅ **Complete training pipelines** for task-specific fine-tuning  
✅ **Zero-shot evaluation framework** to measure transfer learning capabilities  
✅ **Advanced RAG system** with hybrid search (vector + semantic) using Weaviate  
✅ **Experiment tracking** with systematic result documentation  
✅ **Production infrastructure** with Docker, FastAPI, and Gradio support  
✅ **Comprehensive evaluation metrics** and comparative analysis tools  
✅ **Reusable utility modules** for data processing and model management  

---

## 🔬 Our Experimental Approaches

The GPLSI group implements **three distinct strategies**, each testing different hypotheses about decoder-only model capabilities:

### 1️⃣ **Fine-Tuning on Specific Tasks**

**Hypothesis:** Adapting decoder models to task-specific Spanish data improves performance.

**Characteristics:**
- **Mechanism:** Task-specific training with token classification heads
- **Location:** `experiments/fine_tuning/`
- **Use Case:** Baseline performance and upper bound estimation
- **Training:** Gradient-based optimization on task-specific data

**Strengths:**
- Optimal individual task performance
- Leverages task-specific patterns and nuances
- Provides upper bound for comparison

**Limitations:**
- Requires training per task (computational cost)
- Higher data requirements
- Risk of overfitting to specific tasks

**When to Use:**
- Maximum performance is critical
- Sufficient task-specific training data available
- Computational resources permit per-task training

---

### 2️⃣ **Zero-Shot Experiments**

**Hypothesis:** Pretrained decoder models possess sufficient Spanish linguistic knowledge for direct task solving.

**Characteristics:**
- **Mechanism:** Direct inference on tasks without any task-specific training
- **Location:** `experiments/zero_shot/`
- **Use Case:** Measuring inherent model capabilities and transfer learning
- **Inference:** Single forward pass, no adaptation

**Strengths:**
- No task-specific training required (fastest path to results)
- Demonstrates transfer learning capability
- Minimal computational overhead
- Tests true generalization ability

**Limitations:**
- Lower individual task performance
- Depends heavily on pretraining quality
- Limited by model's inherent understanding
- No task-specific optimization

**When to Use:**
- Quick baseline results needed
- Limited training resources
- Evaluating model generalization
- Assessing multilingual capabilities

---

### 3️⃣ **Retrieval-Augmented Generation (RAG) with Few-Shot Learning**

**Hypothesis:** Augmenting inference with dynamically retrieved relevant examples dramatically improves performance across diverse tasks.

**Architecture:**