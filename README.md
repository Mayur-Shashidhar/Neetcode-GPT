# Neetcode-GPT: Building GPT From First Principles

A comprehensive implementation of machine learning and transformer fundamentals built while completing the NeetCode Machine Learning curriculum.

This repository progresses from core concepts such as gradient descent and backpropagation to modern transformer components including self-attention, KV Cache optimization, Grouped Query Attention (GQA), and a fully functional GPT-style language model.

The goal of this project was not simply to use existing frameworks, but to understand and implement the mathematical and architectural foundations behind modern large language models.

---

## Overview

Modern AI systems are built upon several layers of abstraction:

1. Mathematical optimization
2. Neural networks
3. Deep learning training pipelines
4. Natural language processing
5. Attention mechanisms
6. Transformer architectures
7. Generative language models

This repository implements each of these layers step-by-step, culminating in a GPT-style text generation system.

---

## Features

### Machine Learning Foundations

Implemented from scratch:

* Gradient Descent
* Linear Regression
* Activation Functions
* Loss Functions
* Single Neuron Models
* Backpropagation
* Multi-Layer Perceptrons (MLPs)
* Weight Initialization
* Dead ReLU Detection
* Training Diagnostics
* Training Loops

### Deep Learning Fundamentals

* Forward Propagation
* Gradient Computation
* Multi-Layer Backpropagation
* Classification Models
* Sentiment Analysis
* PyTorch Fundamentals

### NLP Pipeline

* Text Preprocessing
* Vocabulary Construction
* Character-Level Encoding
* Byte Pair Encoding (BPE) Tokenization
* Dataset Preparation
* Batch Data Loading

### Transformer Components

* Token Embeddings
* Positional Embeddings
* Single Head Attention
* Multi-Head Attention
* Transformer Blocks
* Layer Normalization
* Batch Normalization
* RMS Normalization

### GPT Optimizations

* KV Cache
* Cached Attention
* Grouped Query Attention (GQA)

### GPT Model

* Decoder-only Transformer Architecture
* Autoregressive Training
* Next Token Prediction
* Text Generation Pipeline

---

## Repository Structure

```text
Neetcode-GPT/

├── foundations/
│   ├── gradient_descent.py
│   ├── linear_regression.py
│   ├── neuron.py
│   ├── backprop.py
│   ├── multi_layer_backprop.py
│   ├── mlp.py
│   ├── activations.py
│   ├── loss.py
│   ├── training_loop.py
│   ├── training_diagnostics.py
│   ├── dead_relu_detector.py
│   ├── digit_classifier.py
│   ├── sentiment.py
│   ├── pytorch_basics.py
│   └── weight_init.py

├── data/
│   ├── tokenizer.py
│   ├── tokenizer_utils.py
│   ├── vocab.py
│   ├── dataset.py
│   ├── loader.py
│   └── nlp_preprocessing.py

├── model/
│   ├── attention.py
│   ├── multi_head_attention.py
│   ├── transformer.py
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── normalization.py
│   ├── batch_normalization.py
│   ├── rms_normalization.py
│   ├── kv_cache.py
│   ├── grouped_query_attention.py
│   └── gpt.py

├── train.py
├── generate.py
├── requirements.txt
└── README.md
```

---

## Learning Progression

The project follows the same progression used in modern LLM development:

```text
Gradient Descent
        ↓
Linear Models
        ↓
Neural Networks
        ↓
Backpropagation
        ↓
Deep Learning
        ↓
NLP Foundations
        ↓
Attention
        ↓
Transformers
        ↓
GPT
        ↓
Optimized Inference
```

---

## GPT Architecture

Implemented Components:

* Token Embedding Layer
* Positional Embedding Layer
* Multi-Head Self Attention
* Feed Forward Network
* Residual Connections
* Layer Normalization
* Transformer Blocks
* Vocabulary Projection Layer

Generation Strategy:

* Context Cropping
* Softmax Sampling
* Autoregressive Decoding
* Next Token Prediction

---

## Advanced Topics Implemented

### KV Cache

KV Cache stores previously computed Keys and Values during inference, reducing redundant computations and significantly improving generation speed.

Benefits:

* Faster autoregressive generation
* Reduced transformer recomputation
* Scalable inference

### Grouped Query Attention (GQA)

Grouped Query Attention reduces memory consumption by allowing multiple query heads to share key-value heads.

Benefits:

* Lower memory usage
* Faster inference
* Technique used in modern production LLMs

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/neetcode-gpt.git

cd neetcode-gpt
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run GPT training:

```bash
python train.py
```

The training pipeline:

1. Creates training batches
2. Performs forward propagation
3. Computes cross-entropy loss
4. Executes backpropagation
5. Updates parameters using AdamW

---

## Text Generation

Generate text using the trained model:

```bash
python generate.py
```

Generation process:

```text
Input Context
      ↓
Forward Pass
      ↓
Softmax Probabilities
      ↓
Sample Next Token
      ↓
Append To Context
      ↓
Repeat
```

---

## Technologies Used

* Python
* PyTorch
* NumPy
* TorchTyping

---

## Key Concepts Demonstrated

Machine Learning:

* Optimization
* Gradient Descent
* Regression
* Classification

Deep Learning:

* Neural Networks
* Backpropagation
* MLPs

Natural Language Processing:

* Tokenization
* Embeddings
* Language Modeling

Transformer Systems:

* Self Attention
* Multi Head Attention
* Positional Encoding
* Layer Normalization
* KV Cache
* Grouped Query Attention

Large Language Models:

* GPT Architecture
* Autoregressive Generation
* Next Token Prediction

---

## Educational Objective

This project was built to gain a first-principles understanding of how modern language models work internally.

Rather than relying solely on high-level libraries, each component was implemented individually to develop intuition for:

* Neural network training
* Transformer mechanics
* Attention computation
* GPT architecture
* LLM inference optimizations

---

## Acknowledgements

Built while completing the NeetCode Machine Learning curriculum.

Special thanks to the NeetCode team for creating a structured path from machine learning fundamentals to transformer-based language models.

---

## Author

**S Mayur**

Computer Science Student
Machine Learning • Deep Learning • NLP • Systems Research

Built on June 12, 2026.
