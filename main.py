from dotenv import load_dotenv
from fastapi import FastAPI
from models import Conversation
from langchain.docstore.document import Document

load_dotenv()

from embedchain import App

chat_bot = App()


app = FastAPI()


@app.get("/")
def read_root():
    return {"app": "Cisco Security Advisories"}


@app.post("/conversation")
def conversation(conversation: Conversation):
    # return {"conversation": conversation.messages[0].content.message}
    # return chat_bot.query(conversation.messages[0].content.message)
    return query(conversation.messages[0].content.message)


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


def generate_prompt(input_query: str, contexts: list[str]):
    """
    Generates a prompt based on the given query and context, ready to be passed to an LLM

    :param input_query: The query to use.
    :param context: Similar documents to the query used as context.
    :return: The prompt
    """

    if type(contexts) is list and len(contexts) > 0:
        prompt = f"""Use the following pieces of context to answer the query at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {" | ".join(contexts)}
            Query: {input_query}
            Helpful Answer:
            """

    else:
        prompt = f"""Use the following pieces of context to answer the query at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {contexts[0]}
            Query: {input_query}
            Helpful Answer:
            """
    return prompt


def query(input_query):
    """
    Queries the vector database based on the given input query.
    Gets relevant doc based on the query and then passes it to an
    LLM as context to get the answer.

    :param input_query: The query to use.
    :return: The answer to the query.
    """
    contexts = retrieve_from_database(input_query, 3)
    prompt = generate_prompt(input_query, contexts)
    answer = chat_bot.get_answer_from_llm(prompt)
    return answer
