"""Multi-language RAG pipeline.

Ask in one language, retrieve documents written in another — no translation step.
A multilingual embedding model maps every language into one shared vector space,
so an English query can surface French or Spanish chunks directly.

Inspired by Aman Kharwal's tutorial:
https://amanxai.com/2026/04/22/build-a-multi-language-rag-pipeline/
"""

from __future__ import annotations

import os
from collections.abc import Iterable

EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
TOP_K = int(os.environ.get("TOP_K", "3"))

PROMPT_TEMPLATE = """
You are a helpful assistant.
Use the retrieved context to answer the user's question.
The context may be in a different language than the question.
Answer in the same language as the user's question.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def sample_documents():
    """A small multilingual knowledge base used for the demo."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content=(
                "The company policy allows up to 20 days of paid time off per year. "
                "Employees must request time off at least two weeks in advance."
            ),
            metadata={"language": "English", "source": "HR_Policy_EN"},
        ),
        Document(
            page_content=(
                "La politique de l'entreprise accorde jusqu'a 20 jours de conges payes "
                "par an. Les employes doivent en faire la demande au moins deux semaines "
                "a l'avance."
            ),
            metadata={"language": "French", "source": "HR_Policy_FR"},
        ),
        Document(
            page_content=(
                "El soporte tecnico esta disponible 24/7. Para problemas urgentes, llame "
                "al numero de emergencia en lugar de enviar un correo electronico."
            ),
            metadata={"language": "Spanish", "source": "IT_Policy_ES"},
        ),
    ]


def format_docs(docs: Iterable) -> str:
    """Join retrieved documents into a single context string, with sources."""
    return "\n\n".join(
        f"[{getattr(d, 'metadata', {}).get('source', '?')}] {d.page_content}"
        for d in docs
    )


def build_chain():
    """Assemble the LCEL RAG chain. Heavy imports stay inside the function."""
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=sample_documents(),
        embedding=embeddings,
        collection_name="multilingual_docs",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main() -> None:
    chain = build_chain()
    for query in (
        "How do I handle urgent IT issues?",
        "Combien de jours de conges payes puis-je prendre ?",
    ):
        print(f"User: {query}")
        print(f"AI: {chain.invoke(query)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
