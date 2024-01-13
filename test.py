import os
import embedchain
import openai

from embedchain import App
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]

print(embedchain.__version__)

chat_bot = App.from_config(config_path="config.yaml")
chat_bot.add("./data/output-2023-06-22T03:37:39.362Z.csv", data_type="csv")
