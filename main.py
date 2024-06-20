import modal
import datetime

# definition of our container image for jobs on Modal
# Modal gets really powerful when you start using multiple images!
backend_image = (
    modal.Image.debian_slim(python_version="3.11.7")
    .poetry_install_from_file("pyproject.toml")
    .apt_install("sqlite3", force_build=True)
    .pip_install("pysqlite3-binary", "sqlite-utils")
)

# we define a Stub to hold all the pieces of our app
# most of the rest of this file just adds features onto this Stub
app = modal.App(
    name="cisco-chat-backend",
    image=backend_image,
    secrets=[
        # this is where we add API keys, passwords, and URLs, which are stored on Modal
        modal.Secret.from_name("openai-api-key"),
    ],
    mounts=[
        # we make our local modules available to the container
        modal.Mount.from_local_python_packages("models"),
        modal.Mount.from_local_dir("db", remote_path="/root/db"),
        # modal.Mount.from_local_dir(
        #     "db/ccea0d9e-c7d4-4354-9d57-f9f22baf346f",
        #     remote_path="/root/db/ccea0d9e-c7d4-4354-9d57-f9f22baf346f",
        # ),
    ],
)


@app.function(
    image=backend_image, container_idle_timeout=300, timeout=600, cpu=1.0, keep_warm=1
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from models import Conversation

    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.vectorstores import Chroma

    db = Chroma(
        collection_name="my-collection",
        persist_directory="/root/db",
        embedding_function=OpenAIEmbeddings(),
    )

    chatOpenAI = ChatOpenAI(
        model="gpt-4-turbo",
        max_tokens=1000,
        verbose=True,
    )

    template = """
        As a digital security expert, I aim to educate laypeople in a clear and
        simple way about Cisco security advisories.
        Use the following pieces of context to answer the query at the end.
        If you don't know the answer, just say that you don't know,
        don't try to make up an answer.
        Always say "thanks for asking!" at the end of the answer.
        > Context
        {context}

        If necessary, rewrite the answer to make it as didactic as possible.
        > Question: {question}
        > Helpful Answer:
        """
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # TODO - https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/self_query/
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

    web_app = FastAPI()

    @web_app.get("/")
    def index():
        return {
            "app": "Cisco ChatGPT - Security Advisories",
            "status": "ok",
            "timestamp": datetime.datetime.now(),
        }

    @web_app.post("/conversation")
    async def conversation(conversation: Conversation):
        """
        Queries the vector database based on the given input query.
        Gets relevant doc based on the query and then passes it to an
        LLM as context to get the answer.

        :param input_query: The query to use.
        :return: The answer to the query.
        """
        question = conversation.messages[-1].content
        conversation.messages.pop()

        return chain({"question": question, "chat_history": conversation.messages})

    return web_app
