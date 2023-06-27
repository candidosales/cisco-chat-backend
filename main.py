from dotenv import load_dotenv

load_dotenv()

from embedchain import App

chat_bot = App()

chat_bot.add("youtube_video", "https://www.youtube.com/watch?v=3qHkcs3kG44")


chat_bot.add_local(
    "qna_pair",
    (
        "Who is Naval Ravikant?",
        "Naval Ravikant is an Indian-American entrepreneur and investor.",
    ),
)


print(
    chat_bot.query(
        "What unique capacity does Naval argue humans possess when it comes to understanding explanations or concepts?"
    )
)
