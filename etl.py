import csv
from time import sleep
from multiprocessing import Pool

from dotenv import load_dotenv

load_dotenv()

from embedchain import App

chat_bot = App()


def add_into_db(url):
    print("add_into_db", url, flush=True)
    sleep(1)
    chat_bot.add("web_page", url)


def read_csv():
    with open("./data/output-2023-06-22T03:37:39.362Z.csv", "r") as file:
        csvreader = csv.reader(file)
        next(csvreader, None)
        urls = []
        for row in csvreader:
            urls.append(row[0])
            # chat_bot.add("web_page", row[0])

        return urls


if __name__ == "__main__":
    urls = read_csv()
    with Pool() as pool:
        result = pool.map(add_into_db, urls)
    print("Program finished!", flush=True)
