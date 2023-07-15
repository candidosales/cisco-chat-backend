import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI

from app.models import Conversation, Message
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

from embedchain import App

load_dotenv()


db = Chroma(
    collection_name="embedchain_store",
    persist_directory="db",
    embedding_function=OpenAIEmbeddings(),
)

chatOpenAI = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    max_tokens=1000,
    verbose=True,
)

template = """
        As a digital security expert, I aim to educate laypeople in a clear and 
        simple way about Cisco security advisories.
        Use the following pieces of context to answer the query at the end.
        If you don't know the answer, just say that you don't know, 
        don't try to make up an answer.
        > Context
        {context}

        Can you suggest approaches how to fix it?
        If necessary, rewrite the answer to make it as didactic as possible.
        At the end of the answer add the link or URL for more information.
        > Question: {question}
        > Helpful Answer:
        """
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8})

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

chain = ConversationalRetrievalChain.from_llm(
    llm=chatOpenAI,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    return_generated_question=True,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
)

chat_bot = App()
app = FastAPI()


@app.get("/")
def read_root():
    return {"app": "Cisco ChatGPT - Security Advisories"}


@app.post("/conversation")
async def conversation(conversation: Conversation):
    return query(conversation.messages)


def query(messages: list[Message]):
    """
    Queries the vector database based on the given input query.
    Gets relevant doc based on the query and then passes it to an
    LLM as context to get the answer.

    :param input_query: The query to use.
    :return: The answer to the query.
    """
    return chain({"question": messages[-1].content, "chat_history": []})


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
