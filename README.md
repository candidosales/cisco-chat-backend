# Cisco ChatGPT Security Advisories - Backend

## Project

- [Crawler to scrap data](https://github.com/candidosales/cisco-crawler-security-advisories)
- [FastAPI](https://fastapi.tiangolo.com/)
  - [pydantic](https://docs.pydantic.dev/latest/)
- [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
- [OpenAI](https://openai.com/)
- Database vector: [ChromaDB](https://www.trychroma.com/)
- [Embedchain](https://github.com/embedchain/embedchain)
- Deploy: [Modal](https://modal.com/)

[Check out how the fronted was made](https://github.com/candidosales/cisco-chat-frontend)

## Diagram architecture

![Diagram architecture](./docs/diagram-architecture.png)

## Set up

### Create and activate the virtual environment

Environments isolate libraries used in one context from those used in another context.

For example, we can use them to isolate the libraries used in this project from those used in other projects.

Done naively, this would result in an explosion of space taken up by duplicated libraries.

Virtual environments allow the sharing of Python libraries across environments if they happen to be using the same version.

We create one for this project with:

```bash
pyenv virtualenv 3.10 chat-backend
```

To start using it, we need to "activate" it:

```bash
pyenv activate chat-backend
```

We've set it as the default environment for this directory with:

```bash
pyenv local chat-backend
```

which generates a `.python-version` file in the current directory.main.py

## Install dependencies

```bash
poetry install
```

## Activate virtual env

To activate the virtual env, remember to paste the code below in your terminal

```bash
poetry shell
```

## Run the project in dev environment

Using modal to run the container

```bash
modal serve main.py
```

## Deploy

```bash
modal deploy main.py
```

## Permission bash files

```bash
sudo chmod -R 755 tasks/pretty_log.sh
```

## Secrets

Update secret in modal account

```bash
modal secret create openai-api-key OPENAI_API_KEY="$OPENAI_API_KEY"
```

## 👍 Contribute

If you want to say thank you and/or support the active development this project:

1. Add a [GitHub Star](https://github.com/candidosales/cisco-chat-backend/stargazers) to the project.
2. Write a review or tutorial on [Medium](https://medium.com/), [Dev.to](https://dev.to/) or personal blog.
3. Support the project by donating a [cup of coffee](https://buymeacoff.ee/candidosales).

## ☕ Supporters

If you want to support Personal Portfolio, you can ☕ [**buy a coffee here**](https://buymeacoff.ee/candidosales)

## ⚠️ Copyright and license

Code and documentation copyright 2023-2030 the [Authors](https://github.com/candidosales/cisco-chat-backend/graphs/contributors) and Code released under the [MIT License](https://github.com/candidosales/cisco-chat-backend/blob/master/LICENSE). Docs released under [Creative Commons](https://creativecommons.org/licenses/by/3.0/).

## References

- [Deeplearning.ai short courses](https://www.deeplearning.ai/short-courses/);
- [LLM Bootcamp - Spring 2023](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/);
- [Modal examples](https://modal.com/examples);
