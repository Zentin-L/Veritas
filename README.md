# Veritas

Veritas is a small command-line tool that lets you ask questions about your PDF files.

The main goal is to keep answers tied to the documents you provide. If the documents do not contain enough information, Veritas can say that it cannot support an answer instead of guessing.

## How it works

Veritas uses two steps:

1. **Build an index.** It reads the text from each PDF, breaks the text into small sections, and saves a local search index.
2. **Ask a question.** It searches the index for the most useful sections, sends those sections to the selected language model, and prints the answer.

The search uses two simple methods together:

- **Keyword search** finds exact words from the question.
- **Meaning-based search** finds sections that are similar in meaning, even when they use different words.

Combining these methods makes the search useful for both exact names and more natural questions. The answer generation code also checks the returned source references so unsupported claims can be rejected.

## Project structure

- `veritas/ingest.py` reads text from PDF files.
- `veritas/chunking.py` splits pages into searchable sections.
- `veritas/embeddings.py` creates meaning-based vectors locally.
- `veritas/index.py` stores the searchable index.
- `veritas/retrieve.py` finds relevant sections with keyword and meaning-based search.
- `veritas/generate.py` builds the question prompt and checks the answer.
- `veritas/llm.py` connects to the configured language model provider.
- `veritas/__main__.py` provides the command-line interface.

The index and embeddings are created on your computer. They are not part of the source repository and are ignored by Git.

## Installation

Veritas requires Python 3.11 or newer.

```powershell
python -m pip install -r requirements.txt
```

## Use Veritas

First, create an index from one or more PDFs:

```powershell
python -m veritas ingest handbook.pdf --output storage/index
```

Then ask a question about those PDFs:

```powershell
python -m veritas ask "What is the main safety rule?" --index storage/index
```

The default index location is `storage/index`, so the `--output` and `--index` options can be left out when using that location.

## Language model setup

The tests do not need a secret or an internet connection. For a live answer, copy `.env.example` to `.env` and configure a provider.

```powershell
Copy-Item .env.example .env
```

Hugging Face can be used by setting `HF_TOKEN`. If automatic provider selection does not work, set `HF_PROVIDER` to the provider listed in the model's Hugging Face inference settings.

For a local, offline option, install Ollama, download the configured model, and set:

```dotenv
LLM_PROVIDERS=ollama
```

Gemini is also supported with `GEMINI_API_KEY`. The `LLM_PROVIDERS` setting controls the provider order. If one provider fails, Veritas tries the next configured provider.

## Run the tests

```powershell
python -m pytest
```

The tests cover PDF extraction, chunking, search, answer validation, provider fallback, and the command-line interface.
