import csv
from multiprocessing import Pool
from dotenv import load_dotenv

load_dotenv()

from loaders.web_page import WebPageLoader
from chunckers.web_page import WebPageChunker

from embedchain import App

chat_bot = App()


def read_csv(max_items: int):
    with open("./data/output-2023-06-22T03:37:39.362Z.csv", "r") as file:
        csvreader = csv.reader(file)
        next(csvreader, None)
        urls = []
        for index, row in enumerate(csvreader):
            if index + 1 == max_items:
                break
            urls.append(row[0])
            # chat_bot.add("web_page", row[0])

        return urls


def add_into_db(url: str):
    print("add_into_db", url, flush=True)
    chat_bot.user_asks.append(["web_page", url])
    load_and_embed(WebPageLoader(), WebPageChunker(), url)


def add_into_db_parallel(urls: list[str]):
    with Pool() as pool:
        result = pool.map(add_into_db, urls)
    print("Program finished!", flush=True)


def add_into_db_sync(urls: list[str]):
    for url in urls:
        chat_bot.user_asks.append(["web_page", url])
        load_and_embed(WebPageLoader(), WebPageChunker(), url)
    print("Program finished! - sync")


def check_output(url: str):
    return WebPageLoader().load_data(url)


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


if __name__ == "__main__":
    urls = read_csv(20)
    if len(urls) > 0:
        add_into_db_sync(urls)
