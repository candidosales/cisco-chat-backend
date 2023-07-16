# Cisco ChatGPT Security Advisories - Backend

## Project

- [FastAPI](https://fastapi.tiangolo.com/)
  - [pydantic](https://docs.pydantic.dev/latest/)
- [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
- [OpenAI](https://openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Embedchain](https://github.com/embedchain/embedchain)
- [Frontend with SvelteKit](https://github.com/candidosales/cisco-chat-frontend)

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

## Parallel Programming

https://www.sitepoint.com/python-multiprocessing-parallel-programming

## Secrets

Update secret in modal account

```bash
modal secret create openai-api-key OPENAI_API_KEY="$OPENAI_API_KEY"
```
