# RAG-CHAT

A Chatbot Application using LLM, langChain and chainlit

## Prerequisites

This application requires python>=3.8.

At the root of the application, run

`pip install -r requirements.txt`

## Preparing to run

### Create chunks from existing data file and insert embeddings to the FAISS VectorDB

`python ingest.py`

## Running the chainlit application

Run the command:
`chainlit run model-test.py -w`

The application should start at: <http://localhost:8000>.
