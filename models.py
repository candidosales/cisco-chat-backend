from pydantic import BaseModel


class Message(BaseModel):
    content: str
    role: str


class Conversation(BaseModel):
    conversation_id: str
    messages: list[Message]
    streaming: bool = False


class Document(BaseModel):
    page_content: str
    metadata: dict
