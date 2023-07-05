import csv
import os
import openai

from dotenv import load_dotenv

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

from loaders.text import TextLoader
from chunckers.text import TextChunker
from advisory.models import Advisory
from advisory.utils import mount_advisory_content

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from embedchain import App

chat_bot = App()


def read_csv(max_items: int):
    with open("./data/output-2023-06-22T03:37:39.362Z.csv", "r") as file:
        csvreader = csv.reader(file)
        next(csvreader, None)  # Jump the header
        advisories = []
        for index, row in enumerate(csvreader):
            print()
            if index + 1 == max_items:
                break
            advisories.append(
                Advisory(
                    url=row[0],
                    id=row[1],
                    title=row[2],
                    severity=row[3],
                    cveList=row[4],
                    cvsScore=row[5],
                    summary=row[6],
                    affectedProducts=row[7],
                    firstPublished=row[8],
                    details=row[9],
                    workarounds=row[10],
                    fixedSoftware=row[11],
                    exploitationPublicAnnouncements=row[12],
                    source=row[13],
                )
            )
        return advisories


def add_advisories_db_sync(advisories: list[Advisory]):
    for advisory in advisories:
        embed_advisories_2v(advisory)
    print("Program finished! - sync")


def load_and_embed(loader, chunker, url):
    """
    Loads the data from the given URL, chunks it, and adds it to the database.

    :param loader: The loader to use to load the data.
    :param chunker: The chunker to use to chunk the data.
    :param url: The URL where the data is located.
    """
    embeddings_data = chunker.create_chunks(loader, url)
    print("[load_and_embed] embeddings_data", embeddings_data)
    documents = embeddings_data["documents"]
    metadatas = embeddings_data["metadatas"]
    ids = embeddings_data["ids"]
    # get existing ids, and discard doc if any common id exist.
    existing_docs = chat_bot.collection.get(
        ids=ids,
        # where={"url": url}
    )
    existing_ids = set(existing_docs["ids"])

    if len(existing_ids):
        data_dict = {
            id: (doc, meta) for id, doc, meta in zip(ids, documents, metadatas)
        }
        data_dict = {
            id: value for id, value in data_dict.items() if id not in existing_ids
        }

        if not data_dict:
            print(f"All data from {url} already exists in the database.")
            return

        ids = list(data_dict.keys())
        documents, metadatas = zip(*data_dict.values())

    chat_bot.collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(
        f"Successfully saved {url}. Total chunks count: {chat_bot.collection.count()}"
    )


def embed_advisories(advisory: Advisory):
    """
    Loads the data from the given URL, chunks it, and adds it to the database.

    :param loader: The loader to use to load the data.
    :param chunker: The chunker to use to chunk the data.
    :param url: The URL where the data is located.
    """

    chunker = TextChunker()
    embeddings_data = chunker.create_chunks(advisory)
    print("[load_and_embed] embeddings_data", embeddings_data)
    documents = embeddings_data["documents"]
    metadatas = embeddings_data["metadatas"]
    ids = embeddings_data["ids"]
    # get existing ids, and discard doc if any common id exist.
    existing_docs = chat_bot.collection.get(
        ids=ids,
        # where={"url": url}
    )
    existing_ids = set(existing_docs["ids"])

    if len(existing_ids):
        data_dict = {
            id: (doc, meta) for id, doc, meta in zip(ids, documents, metadatas)
        }
        data_dict = {
            id: value for id, value in data_dict.items() if id not in existing_ids
        }

        if not data_dict:
            print(f"All data from {advisory.url} already exists in the database.")
            return

        ids = list(data_dict.keys())
        documents, metadatas = zip(*data_dict.values())

    chat_bot.collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(
        f"Successfully saved {advisory.url}. Total chunks count: {chat_bot.collection.count()}"
    )


def embed_advisories_2v(advisory: Advisory):
    # Split the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )

    loader = TextLoader()
    content = mount_advisory_content(advisory)

    document = loader.load_data(advisory.id, advisory.url, content)
    documents = splitter.split_documents([document])

    # Save the texts in the vector database
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = Chroma.from_documents(documents, embeddings, persist_directory="./db")
    db.persist()


if __name__ == "__main__":
    advisories = read_csv(20)
    if len(advisories) > 0:
        add_advisories_db_sync(advisories)
