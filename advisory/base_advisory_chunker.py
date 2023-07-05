import hashlib

from loaders.text import TextLoader
from advisory.models import Advisory
from advisory.utils import mount_advisory_content


class BaseAdvisoryChunker:
    def __init__(self, text_splitter):
        self.text_splitter = text_splitter

    def create_chunks(self, advisory: Advisory):
        documents = []
        ids = []
        loader = TextLoader()
        content = mount_advisory_content(advisory)
        datas = loader.load_data(advisory.url, advisory.id, content)
        metadatas = []
        for data in datas:
            content = data["content"]
            meta_data = data["meta_data"]
            chunks = self.text_splitter.split_text(content)
            url = meta_data["url"]
            for chunk in chunks:
                chunk_id = hashlib.sha256((chunk + url).encode()).hexdigest()
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(meta_data)
        return {
            "documents": documents,
            "ids": ids,
            "metadatas": metadatas,
        }
