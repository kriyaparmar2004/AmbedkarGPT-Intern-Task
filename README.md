# AmbedkarGPT-Intern-Task

A RAG (Retrieval-Augmented Generation) Q&A system for answering questions about Dr. B.R. Ambedkar's speech on "Annihilation of Caste".

## 🎯 Project Overview

This is Assignment 1 for the Kalpit Pvt Ltd AI Intern position. The system uses LangChain, ChromaDB, HuggingFace embeddings, and Ollama's Mistral 7B model to create an intelligent Q&A system that can answer questions based on the provided speech text.

## ✨ Features

- **Document Loading**: Loads and processes the speech text
- **Text Chunking**: Splits text into optimal chunks for retrieval
- **Semantic Search**: Uses HuggingFace embeddings for semantic understanding
- **Local Vector Store**: ChromaDB for persistent storage (no cloud required)
- **Local LLM**: Ollama Mistral 7B (100% free, no API keys)
- **Interactive CLI**: User-friendly command-line interface

## 🛠️ Technology Stack

- **Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **LLM**: Ollama Mistral 7B
- **Language**: Python 3.8+

## 📋 Prerequisites

1. **Python 3.8 or higher**
2. **Ollama** with Mistral 7B model

## 🚀 Installation & Setup

### Step 1: Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the Mistral 7B model
ollama pull mistral

# Verify installation (should start an interactive chat)
ollama run mistral
# Type /bye to exit
```

### Step 2: Clone the Repository
```bash
git clone https://github.com/yourusername/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 3: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- LangChain (RAG framework)
- ChromaDB (vector database)
- sentence-transformers (embeddings)
- tiktoken (tokenization)

### Step 5: Start Ollama Service

**Important**: Ollama must be running before you start the Q&A system.
```bash
# In a separate terminal window:
ollama serve
```

Keep this terminal running while you use the system.

## 🎮 Usage

### Run the Q&A System
```bash
# Make sure venv is activated
python main.py
```
