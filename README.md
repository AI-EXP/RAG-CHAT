# RAG-CHAT

A Chatbot Application using LLM, langChain and chainlit

## Prerequisites

- This application requires python>=3.8<=3.11.

### Download the LLM Library from Hugging Face

- Download the llama 2 GGML library from the [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) Web Site

### Preparation

- Create a folder at the project root (e.g. "model/") and put the `llama-2-7b-chat.ggmlv3.q8_0.bin` in it.
- Create another folder at the project root (e.g. "data/") to hold some of the pdf's you want to question.

At the root of the application, run

`pip install -r requirements.txt`

## Preparing to run

### Create chunks from existing data file and insert embeddings to the FAISS VectorDB

`python ingest.py`

## Running the chainlit application

Run the command:
`chainlit run model-test.py -w`

The application should start at: <http://localhost:8000>.
