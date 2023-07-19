from bs4 import BeautifulSoup

from embedchain.utils import clean_string

from models import Document


class TextLoader:
    def load_data(self, id: str, url: str, content: str):
        soup = BeautifulSoup(content, "html.parser")
        output = []
        content = soup.get_text()
        content = clean_string(content)
        metadata = {
            "url": url,
            "id": id,
        }
        # TODO - Add more metadata to use self-querying retriever -
        # severity, firstPublished, cvsScore
        output.append(
            {
                "page_content": content,
                "metadata": metadata,
            }
        )
        return Document(page_content=content, metadata=metadata)
