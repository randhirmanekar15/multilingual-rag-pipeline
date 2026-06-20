# Multi-Language RAG Pipeline

Ask a question in English and retrieve relevant documents written in French or Spanish — with **no translation step**. A multilingual embedding model maps every language into one shared vector space, so the geometry does the cross-lingual work.

Runs fully locally: Chroma + Ollama + HuggingFace embeddings. No data leaves the machine.

## Stack

| Piece | Choice |
|-------|--------|
| Orchestration | LangChain (LCEL) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | ChromaDB |
| LLM | `llama3` via Ollama |

## Setup

```bash
ollama pull llama3
pip install -r requirements.txt
```

## Usage

```bash
python rag.py
```

Configure via env vars: `EMBED_MODEL`, `OLLAMA_MODEL`, `TOP_K`.

## Test

```bash
pip install pytest
pytest        # tests format_docs without loading embeddings
```

## Limitations

- Small multilingual models trade nuance for coverage; idioms and legal phrasing slip through.
- Retrieval can pull a wrong-language near-duplicate; metadata filtering helps.
- Chunking decides everything — split mid-clause and the embedding drifts.

---

Inspired by Aman Kharwal's tutorial, [Build a Multi-Language RAG Pipeline](https://amanxai.com/2026/04/22/build-a-multi-language-rag-pipeline/). Rebuilt and extended (source-labelled context, configurable k/models).

MIT licensed.
