# Multilingual RAG Pipeline

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green) ![Runs 100% Local](https://img.shields.io/badge/runs-100%25%20local-orange)

**Ask in English, retrieve documents written in French or Spanish — with no translation step.**

## Overview

RAG didn't go away in 2026. It got boring, which is the best thing that can happen to a pattern: it became the default. Every team shipping a product that touches private knowledge reaches for retrieval-augmented generation before they reach for fine-tuning. This project is a clean, readable take on that foundation.

The interesting frontier isn't "RAG, but bigger" — it's RAG that crosses languages. Most pipelines quietly assume your query and your documents share a language. The moment you ship to a global audience, that assumption breaks. The usual fix is bolting on a translation step, which adds latency, cost, and a second place for meaning to get lost. This project skips it entirely. A multilingual embedding model maps every language into one shared vector space, so an English query's nearest neighbors can be French or Spanish chunks. The geometry does the cross-lingual work.

And it all runs on your machine. Embeddings come from a local HuggingFace model; generation comes from Ollama. No API keys, no documents leaving the box, no per-token bill. Local isn't a constraint here — it's the feature. Private by default is the right default when the documents are yours.

## Features

- **Cross-lingual retrieval, zero translation** — ask in English, hit relevant French and Spanish chunks directly.
- **One shared vector space** — `paraphrase-multilingual-MiniLM-L12-v2` embeds all languages into the same geometry.
- **Fully local** — HuggingFace embeddings + Ollama (`llama3`). Nothing leaves your machine.
- **Source-labelled answers** — retrieved context is tagged with its source so you can see where claims came from.
- **Configurable by env var** — swap the embedding model, the LLM, or the number of retrieved chunks without touching code.
- **Built on LCEL** — composed with LangChain Expression Language, so the chain is readable and easy to extend.
- **Tested** — `format_docs` has unit coverage so the context-formatting contract doesn't silently drift.

## How it works

A monolingual embedding model places "the cat sat on the mat" and "le chat" in unrelated regions of vector space. A **multilingual** embedding model is trained so that sentences with the *same meaning* land near each other regardless of language. "The contract expires in March" and "le contrat expire en mars" end up as near neighbors.

That's the whole trick. You embed your French and Spanish documents once, store the vectors in Chroma, and when an English query arrives you embed it with the *same* model. Its nearest vectors are semantic matches — in any language. The LLM then reads the multilingual context and answers in the user's language.

```
   English query
        │
        ▼
 ┌─────────────────────┐
 │ multilingual embed  │   paraphrase-multilingual-MiniLM-L12-v2
 └─────────────────────┘
        │  (one shared vector space)
        ▼
 ┌─────────────────────┐
 │   Chroma retrieve   │   nearest vectors may be EN / FR / ES
 └─────────────────────┘
        │  top-k chunks (any language, source-labelled)
        ▼
 ┌─────────────────────┐
 │   llama3 (Ollama)   │   reasons over multilingual context
 └─────────────────────┘
        ▼
   answer in the user's language
```

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Orchestration | LangChain (LCEL) | Readable, composable chains |
| Vector store | ChromaDB | Local, zero-config persistence |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` | Maps 50+ languages into one shared space |
| LLM | Ollama `llama3` | Local generation, no API keys |
| Runtime | Python 3.10+ | — |

## Project structure

```
multilingual-rag-pipeline/
├── rag.py          # sample docs, build_chain(), format_docs() — run for the demo
├── test_rag.py     # unit tests for format_docs()
├── ARTICLE.md
├── requirements.txt
├── LICENSE         # MIT
└── README.md
```

## Installation

```bash
git clone https://github.com/randhirmanekar15/multilingual-rag-pipeline.git
cd multilingual-rag-pipeline
pip install -r requirements.txt
ollama pull llama3
```

The first run downloads the embedding model from HuggingFace (~120 MB). After that, everything is local.

## Usage

```bash
python rag.py
```

The demo seeds a small set of multilingual documents and asks two questions — one in English, one in French — to show retrieval crossing language boundaries in both directions.

```python
from rag import build_chain

chain = build_chain()
print(chain.invoke("How do I handle urgent IT issues?"))
# retrieves the relevant Spanish/French chunk and answers in English
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace embedding model |
| `OLLAMA_MODEL` | `llama3` | Ollama model used for generation |
| `TOP_K` | `3` | Number of chunks retrieved per query |

## Testing

```bash
pip install pytest
pytest
```

`test_rag.py` covers `format_docs` — the function that turns retrieved documents into source-labelled context, the contract between retrieval and the prompt.

## Limitations

- **Small models trade nuance for coverage.** MiniLM-class multilingual models speak many languages but flatten fine distinctions.
- **Wrong-language near-duplicates.** Retrieval can surface a near-identical chunk in the "wrong" language when several translations exist.
- **Chunking decides everything.** Retrieval quality is downstream of how you split documents.
- **Demo uses in-memory sample docs.** Loading from disk is on the roadmap.

## Roadmap

- [ ] Load documents from disk (PDF / txt / markdown) instead of hardcoded samples
- [ ] Add more languages (German, Hindi, Mandarin)
- [ ] Inline source citations in the generated answer
- [ ] Persist the Chroma index between runs
- [ ] Add a retrieval-quality eval harness

## Credits

📖 Full write-up: [ARTICLE.md](ARTICLE.md).

Based on Aman Kharwal's tutorial, ["Build a Multi-Language RAG Pipeline"](https://amanxai.com/2026/04/22/build-a-multi-language-rag-pipeline/).

**What I changed vs the source tutorial:**

- Source-labelled context, so answers carry provenance.
- Configurable retrieval depth and swappable models via `EMBED_MODEL`, `OLLAMA_MODEL`, and `TOP_K`.

## Author

Built by **Randhir Manekar** — [randhirmanekar.com](https://randhirmanekar.com) · [github.com/randhirmanekar15](https://github.com/randhirmanekar15)

## License

MIT — see [LICENSE](LICENSE).
