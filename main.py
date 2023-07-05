from dotenv import load_dotenv
from fastapi import FastAPI
from models import Conversation
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import openai

# from langchain.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

from embedchain import App

chat_bot = App()

# ----- 2v
# ## Initialize db from the disk
# embeddings = OpenAIEmbeddings(disallowed_special=())
# db = Chroma(embedding_function=embeddings, persist_directory="./db")

# ## Create a retriever function
# retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})

# ## Set up the model for the QA
# llm = ChatOpenAI(temperature=0)

# qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
# ----- 2v

app = FastAPI()


@app.get("/")
def read_root():
    return {"app": "Cisco ChatGPT - Security Advisories"}


@app.post("/conversation")
def conversation(conversation: Conversation):
    return query(conversation.messages[0].content.message)
    # return get_response(conversation.messages[0].content.message)


def retrieve_from_database(input_query: str, number_documents: int):
    """
    Queries the vector database based on the given input query.
    Gets relevant doc based on the query

    :param input_query: The query to use.
    :return: The content of the document that matched your query.
    """

    result = chat_bot.collection.query(
        query_texts=[
            input_query,
        ],
        n_results=number_documents,
    )

    result_formatted = _format_result(result)

    contents = [document[0].page_content for document in result_formatted]

    return contents


def _format_result(results):
    return [
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def generate_prompt_template():
    """
    Generates a prompt based on the given query and context, ready to be passed to an LLM

    :param query: The query to use.
    :param context: Similar documents to the query used as context.
    :return: The prompt
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
        > Query: ```{query}```
        > Helpful Answer:
        """

    return ChatPromptTemplate.from_template(prompt)


# ----- 2v https://twitter.com/cristobal_dev/status/1675745314592915456
# def get_response(question: str):
#     result = qa({"question": question, "chat_history": []})
#     return result["answer"]
# ----- 2v


def query(input_query: str):
    """
    Queries the vector database based on the given input query.
    Gets relevant doc based on the query and then passes it to an
    LLM as context to get the answer.

    :param input_query: The query to use.
    :return: The answer to the query.
    """
    contexts = retrieve_from_database(input_query, 3)
    prompt_template = generate_prompt_template()

    message = prompt_template.format_messages(
        query=input_query, context=" | ".join(contexts)
    )

    answer = get_openai_answer(message)
    return answer


def get_openai_answer(message):
    chat = ChatOpenAI(
        temperature=0.0,
        max_tokens=1000,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    response = chat(message)
    return response.content
