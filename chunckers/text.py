from langchain.text_splitter import RecursiveCharacterTextSplitter
from advisory.base_advisory_chunker import BaseAdvisoryChunker

TEXT_SPLITTER_CHUNK_PARAMS = {
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "length_function": len,
}


class TextChunker(BaseAdvisoryChunker):
    def __init__(self):
        text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CHUNK_PARAMS)
        super().__init__(text_splitter)
