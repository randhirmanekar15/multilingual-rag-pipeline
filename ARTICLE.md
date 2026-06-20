# I Built a RAG Pipeline That Reads French and Spanish Docs From English Questions — No Translation Step

*Ask in English, retrieve from documents written in any language, answer back in English. The trick isn't translation. It's a shared vector space — and it runs entirely on my laptop.*

## Why this, why now

RAG isn't dead. It's the boring, load-bearing pattern underneath almost every serious LLM product in 2026. Fine-tuning gets the headlines; retrieval pays the bills.

But the interesting frontier moved. If you're building anything for a global user base, the question isn't "can I do RAG" — it's "can I do RAG across languages." Your support docs are in French. Your compliance team writes in Spanish. Your user asks in English. The naive fix is a translation API in the middle, which means extra latency, extra cost, and one more place to mangle meaning.

I wanted to skip translation entirely. And I wanted it fully local — Chroma for the vector store, Ollama for the model, HuggingFace embeddings on-device — so no document, query, or embedding ever leaves the machine. For anyone handling internal HR or IT policy, that last part isn't a nice-to-have. It's the whole reason it ships.

## What it does

You ask a question in English. The pipeline searches a mixed-language document store, pulls the most relevant chunks — even if they're written in Spanish or French — and answers you in the language you asked.

No `if language == "fr"` branching. No Google Translate call. The retrieval works because of how the embeddings are built, not because anything got converted.

Example: I ask *"What's the policy on remote work equipment?"* in English. The top hit is a Spanish IT-policy document. The model reads the Spanish context and answers me in English. Ask the same kind of thing in French, and it surfaces the French HR doc.

## The stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangChain (LCEL) | Clean, composable chains |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Maps 50+ languages into one shared space |
| Vector store | ChromaDB | Local, zero-config, persistent |
| LLM | llama3 via Ollama | Runs offline, decent multilingual comprehension |
| Output | `StrOutputParser` | Plain text out |

## How it works

Here's the only idea that matters: a multilingual embedding model places sentences with the same *meaning* near each other in vector space, regardless of language.

"Remote work equipment policy," "política de equipos para trabajo remoto," and "politique d'équipement pour le télétravail" all land in roughly the same neighborhood. So when I embed an English question and search, the nearest vectors can absolutely be Spanish or French chunks. The geometry does the cross-lingual work that a translation step used to do.

Set up the embeddings and the store:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

Then the LCEL chain. The prompt is doing real work here — it warns the model the context may be in a different language than the question, and tells it to answer in the user's language:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

That's the entire engine. Everything else is data and polish.

## What I changed

Aman's original is inline-doc, single-file, k=2. I took it further toward something I'd actually run:

1. **Source-labelled context.** `format_docs` prepends each retrieved chunk with its `source` (e.g. `[IT_Policy_ES]`), so the context — and any answer grounded in it — carries provenance instead of being an unsourced wall of text.
2. **Configurable retrieval and models.** The embedding model, the LLM, and the retrieval depth `k` are all environment variables (`EMBED_MODEL`, `OLLAMA_MODEL`, `TOP_K`), so you can tune the pipeline without editing code. With mixed languages, bumping `k` past 2 helps when near-duplicate translations crowd the top results.
3. **Validated `TOP_K`.** The `k` value is parsed and checked to be a positive integer, so a bad env var fails fast with a clear message instead of a confusing downstream error.

Documents still ship as an inline sample set (English/French/Spanish). Loading from disk and adding more languages are on the roadmap, not in the box yet.

## Where it breaks

This is not magic, and I'd be lying if I pretended otherwise.

Small multilingual models trade coverage for nuance. MiniLM-L12 knows 50+ languages but doesn't deeply *get* any of them — idioms and legal phrasing slip through. For high-stakes retrieval, a bigger embedding model earns its keep.

Retrieval can pull the wrong-language near-duplicate. If your store has the same policy in three languages, the top-k can fill up with translations of one doc and miss the doc you needed. Per-language dedup or metadata filtering helps.

And chunking decides everything. Split a French doc mid-clause and its embedding drifts off-meaning, so the English query never finds it. I spent more time tuning chunk size than tuning the model.

## Takeaway

Cross-lingual RAG without a translation step isn't a trick — it's just choosing an embedding model that already speaks every language into one geometry. Add a local stack and you get retrieval that's private by default.

*Built on the foundation of Aman Kharwal's walkthrough, ["Build a Multi-Language RAG Pipeline"](https://amanxai.com/2026/04/22/build-a-multi-language-rag-pipeline/) — I adapted the architecture and added source-labelled context, configurable models/retrieval depth, and `TOP_K` validation.*

### Sources
- [Aman Kharwal — Build a Multi-Language RAG Pipeline](https://amanxai.com/2026/04/22/build-a-multi-language-rag-pipeline/)
- [LangChain — AI Agent Frameworks](https://www.langchain.com/resources/ai-agent-frameworks)
- [Codersera — Open-Source LLMs Landscape 2026](https://codersera.com/blog/open-source-llms-landscape-2026/)
