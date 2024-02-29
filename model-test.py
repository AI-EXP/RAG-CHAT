from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
if you don't know the answer, please just say that you don't know the answer, don't try to make up
the answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt_template():
    """
    Prompt template for QA Retrieval for each vector stores
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    return prompt

def load_local_llm():
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    
    return llm

def retrieve_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = db.as_retriever(search_kwargs={'k':2}),
        # return_source_document = True,
        chain_type_kwargs={'prompt':prompt},
        return_source_documents = True,
       
    )

    return qa_chain

def qa_chatbot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-V2',
                                       model_kwargs = {'device': 'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_local_llm()
    qa_prompt = set_custom_prompt_template()

    qa = retrieve_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_chatbot()
    response = qa_result({'query': query})

    return response

## Chainlit stuffs
@cl.on_chat_start
async def start():
    chain = qa_chatbot()
    msg = cl.Message(content="Starting the RAG Chatbot...")
    await msg.send()
    msg.content = "Hi, Welcome to the RAG-Chatbot. What is your question?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    print("LangChain response:", res)

    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources: " + str(sources)
    else:
        answer += f"\nNo Sources found"

    await cl.Message(content=answer).send()

