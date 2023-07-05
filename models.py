from pydantic import BaseModel


class Author(BaseModel):
    role: str


class Content(BaseModel):
    content_type: str
    message: str


class Message(BaseModel):
    content: Content


class Conversation(BaseModel):
    conversation_id: str
    messages: list[Message]
    author: Author | None = None
    model: str | None = None
    parent_message_id: str | None = None
    streaming: bool = False


class Document(BaseModel):
    page_content: str
    metadata: dict
