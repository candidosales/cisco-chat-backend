from bs4 import BeautifulSoup

from embedchain.utils import clean_string


class TextLoader:
    def load_data(self, url: str, id: str, content: str):
        soup = BeautifulSoup(content, "html.parser")
        output = []
        content = soup.get_text()
        content = clean_string(content)
        meta_data = {"url": url, "id": id}
        output.append(
            {
                "content": content,
                "meta_data": meta_data,
            }
        )
        return output
