import uvicorn
import asyncio

from typing import AsyncIterable, Awaitable

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from models import Conversation
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

from embedchain import App

from langchain.vectorstores import Chroma

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
        As a digital security expert, I aim to educate laypeople in a clear and simple way about Cisco security advisories. 
        Use the following pieces of context to answer the query at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
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

qa = RetrievalQA.from_chain_type(
    llm=chatOpenAI,
    retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True,
)

chat_bot = App()

app = FastAPI()


@app.get("/")
def read_root():
    return {"app": "Cisco ChatGPT - Security Advisories"}


@app.post("/conversation")
async def conversation(conversation: Conversation):
    return query(conversation.messages[0].content.message)


def retrieve_from_database(input_query: str, number_relevant_documents: int):
    return db.search(
        query=input_query,
        search_type="mmr",
        kwargs={"k": number_relevant_documents, "fetch_k": 8},
    )


def generate_prompt_message(query: str, context: str):
    """
    Generates a prompt based on the given query and context, ready to be passed to an LLM

    :param query: The query to use.
    :param context: Similar documents to the query used as context.
    """

    prompt = """
        As a digital security expert, I aim to educate laypeople in a clear and simple way about Cisco security advisories. 
        Use the following pieces of context to answer the query at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        > Context
        ```
         {context}
        ```
        Can you suggest approaches how to fix it?
        If necessary, rewrite the answer to make it as didactic as possible.
        At the end of the answer add the link or URL for more information.
        > Query: ```{query}```
        > Helpful Answer:
        """

    return ChatPromptTemplate.from_template(prompt).format_messages(
        query=query, context=context
    )


def get_prompt_template():
    """
    Generates a prompt based on the given query and context, ready to be passed to an LLM

    :param query: The query to use.
    :param context: Similar documents to the query used as context.
    """

    template = """
        As a digital security expert, I aim to educate laypeople in a clear and simple way about Cisco security advisories. 
        Use the following pieces of context to answer the query at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        > Context
        ```
         {context}
        ```
        Can you suggest approaches how to fix it?
        If necessary, rewrite the answer to make it as didactic as possible.
        At the end of the answer add the link or URL for more information.
        > Query: ```{query}```
        > Helpful Answer:
        """
    return PromptTemplate(
        input_variables=["context", "query"],
        template=template,
    )


def query(input_query: str):
    """
    Queries the vector database based on the given input query.
    Gets relevant doc based on the query and then passes it to an
    LLM as context to get the answer.

    :param input_query: The query to use.
    :return: The answer to the query.
    """
    return qa({"query": input_query})


def query_old(input_query: str, streaming: bool):
    """
    Queries the vector database based on the given input query.
    Gets relevant doc based on the query and then passes it to an
    LLM as context to get the answer.

    :param input_query: The query to use.
    :return: The answer to the query.
    """
    documents = retrieve_from_database(input_query, 8)
    contexts = []

    for doc in documents:
        contexts.append(doc.page_content)

    message = generate_prompt_message(input_query, " | ".join(contexts))

    if streaming:
        return StreamingResponse(send_message(message), media_type="text/event-stream")
    else:
        return get_openai_answer(message)


def get_openai_answer(message: list[BaseMessage]):
    return chatOpenAI(message)


async def send_message(messages: list[BaseMessage]) -> AsyncIterable[str]:
    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    # Begin a task that runs in the background.
    task = asyncio.create_task(
        wrap_done(chatOpenAI.agenerate([messages]), callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield f"data: {token}\n\n"

    await task


if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8000, app=app)
