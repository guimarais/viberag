# RAG (Retrieval-Augmented Generation) System

This is a Python-based RAG system that allows you to:
1. Load and process documents
2. Create embeddings and store them in a vector database
3. Perform semantic search and generate responses using LLM

## Setup

1. Create and activate a Conda environment:
```bash
conda create -n rag-env python=3.10
conda activate rag-env
```

2. Install dependencies using pip:
```bash
pip install -r requirements.txt
```

3. Make sure your OpenAI API key is a system variable.

## Usage

1. Place your documents in the `documents` folder
2. Run the ingestion script:
```bash
python ingest.py
```

3. Start the Streamlit interface:
```bash
streamlit run app.py
```

Alternatively, you can use the command-line interface:
```bash
python query.py
```

## Project Structure

- `app.py`: Streamlit web interface for the RAG system
- `ingest.py`: Script to process documents and create the vector store
- `query.py`: Script to perform RAG queries via command line
- `rag_utils.py`: Utility functions for the RAG system
- `documents/`: Folder to store your documents
- `vectorstore/`: Folder where the ChromaDB vector store is saved

## Note
Make sure to always activate your Conda environment before running the scripts:
```bash
conda activate rag-env
``` 