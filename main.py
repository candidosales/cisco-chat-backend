from dotenv import load_dotenv
from fastapi import FastAPI
from models import Conversation

load_dotenv()

from embedchain import App

chat_bot = App()


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/conversation")
def conversation(conversation: Conversation):
    # return {"conversation": conversation.messages[0].content.message}
    return chat_bot.query("What's CVE-2023-20028")
