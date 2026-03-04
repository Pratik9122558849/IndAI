# IndAI – Custom Large Fine-Tuned AI Assistant

## Overview

**IndAI** is a custom AI assistant built by **Pratik in 2026** using a fine-tuned version of **GPT-2 Large**. The project demonstrates how to train a personal language model locally using modern machine learning techniques such as **LoRA (Low-Rank Adaptation)** and the **Hugging Face Transformers ecosystem**.

The goal of this project is to provide a practical guide and working implementation for developers who want to:

* Build their own AI assistant
* Fine-tune large language models locally
* Train models with custom datasets
* Control model identity and responses
* Continue training models incrementally
* Run inference through a simple chat interface

This repository contains everything required to reproduce the system:

* Training scripts
* Dataset generation tools
* Chat interface
* Identity training data
* Instructions for incremental model updates

The assistant trained in this project identifies itself as:

> **IndAI — an AI assistant created by Pratik in 2026.**

The system is designed to run on a **MacBook with Apple Silicon (M1/M2/M3)** but can also be trained on Linux or Windows machines with GPUs.

---

# Project Goals

The primary objectives of this project are:

1. Demonstrate how to fine-tune GPT-2 Large locally.
2. Implement LoRA to reduce training memory usage.
3. Enable incremental training without restarting from scratch.
4. Create a controllable AI identity (IndAI).
5. Provide a simple interactive chat interface.
6. Document the entire pipeline for learning and experimentation.

This project is meant for **learning, experimentation, and personal AI development**.

---

# Features

## Custom AI Identity

The model is trained to respond with a consistent identity:

* Name: **IndAI**
* Creator: **Pratik**
* Created in: **2026**

Example:

```
User: What is your name?
AI: My name is IndAI.

User: Who created you?
AI: I was created by Pratik in 2026.
```

---

## LoRA Fine-Tuning

Training large language models normally requires massive GPU memory.

This project uses **LoRA (Low-Rank Adaptation)** which allows training only a small number of parameters.

Benefits:

* Much lower memory usage
* Faster training
* Compatible with consumer hardware
* Incremental training support

---

## Incremental Training

The model can continue learning from new datasets.

Instead of restarting training each time, the system loads the previously trained LoRA adapter and continues training.

Training pipeline:

```
GPT-2 Large (base model)
        ↓
First fine-tuning
        ↓
Saved LoRA adapter
        ↓
Further fine-tuning with new datasets
```

This allows the model to continuously improve.

---

## Local Model Loading

All models are loaded using local cache files to avoid repeated downloads.

This ensures:

* Faster startup
* Offline usage
* Reproducible experiments

---

## Custom Dataset Training

The model is trained using JSONL datasets formatted like this:

```
{"input":"who are you","output":"I am IndAI created by Pratik in 2026."}
{"input":"what is your name","output":"My name is IndAI."}
{"input":"who created you","output":"I was created by Pratik in 2026."}
```

The system converts them into conversational format:

```
User: question
Assistant: answer
```

---

# Project Structure

Example repository structure:

```
IndAI/
│
├── train_indai.py
├── chat.py
├── generate_identity_dataset.py
│
├── datasets/
│   ├── indai_dataset.jsonl
│   └── conversations.jsonl
│
├── gpt2_large_lora/
│   ├── adapter_config.json
│   └── adapter_model.bin
│
├── README.md
└── requirements.txt
```

Explanation:

| File                         | Purpose              |
| ---------------------------- | -------------------- |
| train_indai.py               | Main training script |
| chat.py                      | Chat interface       |
| generate_identity_dataset.py | Dataset generator    |
| datasets                     | Training data        |
| gpt2_large_lora              | LoRA trained model   |
| requirements.txt             | Python dependencies  |

---

# Requirements

Minimum requirements:

* Python 3.10+
* PyTorch
* Transformers
* PEFT
* Datasets
* Apple Silicon or CUDA GPU

Recommended hardware:

| Component | Recommended               |
| --------- | ------------------------- |
| CPU       | Apple M1/M2 or modern CPU |
| RAM       | 16GB                      |
| Storage   | 10GB+                     |
| GPU       | Apple MPS or CUDA GPU     |

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/indai
cd indai
```

Create virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install torch transformers datasets peft accelerate
```

---

# Training the Model

Run the training script:

```
python train_indai.py
```

The script will:

1. Load GPT-2 Large
2. Apply LoRA adapters
3. Load dataset
4. Tokenize data
5. Train the model
6. Save LoRA adapter

Training output example:

```
Using device: mps
Loading tokenizer...
Loading base GPT-2 model...
Applying LoRA adapters...
Loading dataset...
Tokenizing dataset...
Starting training...
```

---

# Chat Interface

After training, run:

```
python chat.py
```

Example interaction:

```
You: hello
AI: Hello! I am IndAI.

You: who created you
AI: I was created by Pratik in 2026.
```

The chat interface controls generation parameters to reduce repetition and random outputs.

---

# Generation Settings

Important parameters used in the project:

```
max_new_tokens = 20
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.2
no_repeat_ngram_size = 3
```

These help prevent the model from generating long irrelevant text.

---

# Identity Dataset

The project includes a generator for creating large identity datasets.

Example script:

```
python generate_identity_dataset.py
```

This generates thousands of variations like:

```
who are you
what is your name
introduce yourself
who built you
```

Each response reinforces the IndAI identity.

---

# Continuing Training

To continue training with new data:

1. Add new dataset lines
2. Run training again

```
python train_indai.py
```

The script automatically loads the previous LoRA adapter.

---

# Known Limitations

Because GPT-2 is an older architecture:

* It sometimes generates extra sentences
* It may continue writing text beyond the intended answer
* It may require more dataset examples for stable identity

These issues can be mitigated using:

* stop tokens
* generation limits
* larger datasets

---

# Future Improvements

Planned improvements include:

* Conversation memory
* Streaming responses
* Web interface
* API integration
* Training with larger models
* Retrieval augmented generation

---

# Example Use Cases

This project can be used for:

* Personal AI assistants
* AI learning projects
* Chatbot experiments
* Dataset training research
* Local LLM experimentation

---

# Security Note

This model runs locally and does not send data to external servers.

However, developers should still ensure that sensitive data is not included in training datasets.

---

# Contributing

Contributions are welcome.

Possible contributions include:

* Improving training scripts
* Adding new datasets
* Optimizing model performance
* Improving documentation

---

# License

This project is released under the MIT License.

You are free to use, modify, and distribute the code with attribution.

---

# Credits

This project builds upon the work of the open-source AI community.

Key technologies used:

* Hugging Face Transformers
* PyTorch
* PEFT (LoRA)
* GPT-2 architecture

---

# Author

**Pratik**
Creator of **IndAI**
2026

---

# Final Notes

This repository demonstrates that it is possible to build and train a custom AI assistant locally with relatively modest hardware.

While GPT-2 is not the newest architecture, it remains a valuable model for experimentation and learning.

IndAI serves as a foundation for exploring:

* language model training
* custom AI identity
* incremental model improvement
* personal AI systems

Future versions may expand IndAI into a more capable assistant with memory, reasoning improvements, and advanced training methods.

---

End of README
