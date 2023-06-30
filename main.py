import csv
from time import sleep
from multiprocessing import Pool

# print("Number of processors: ", mp.cpu_count())

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

        return urls


if __name__ == "__main__":
    urls = read_csv()
    with Pool() as pool:
        result = pool.map(add_into_db, urls)
    print("Program finished!", flush=True)


# Step 1: Init multiprocessing.Pool()
# pool = mp.Pool(mp.cpu_count())

# urls = read_csv()
# # print(urls)

# # Step 2: `pool.apply` the `howmany_within_range()`
# results = [pool.apply(add_into_db, args=(row)) for row in urls]

# pool.close()


# def etl():
#     with open("./data/output-2023-06-22T03:37:39.362Z.csv", "r") as file:
#         csvreader = csv.reader(file)
#         next(csvreader, None)
#         for row in csvreader:
#             print(row[0])
#             chat_bot.add("web_page", row[0])


# etl()


# from embedchain import App

# chat_bot = App()


# chat_bot.add("youtube_video", "https://www.youtube.com/watch?v=3qHkcs3kG44")


# chat_bot.add_local(
#     "qna_pair",
#     (
#         "Who is Naval Ravikant?",
#         "Naval Ravikant is an Indian-American entrepreneur and investor.",
#     ),
# )


# print(
#     chat_bot.query(
#         "What unique capacity does Naval argue humans possess when it comes to understanding explanations or concepts?"
#     )
# )
