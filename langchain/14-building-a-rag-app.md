> [← Retrievers](13-retrievers.md) | [↑ Index](README.md) | [Tools →](15-tools.md)

# Building a RAG App with LangChain

Complete notes for building a **Retrieval-Augmented Generation (RAG)** application — from theory to a full working PDF Q&A example.

---

## What is RAG?

**RAG** = **R**etrieval-**A**ugmented **G**eneration.

It combines two ideas:

1. **Retrieval** — search an external knowledge base for relevant text.
2. **Generation** — let the LLM answer using that retrieved text as context.

> RAG retrieves relevant information from an external knowledge base and gives it to the LLM as context before the LLM generates an answer.

### Why RAG?

| LLM limitation | How RAG helps |
|---|---|
| No access to private company data | Index your own documents |
| Training data may be outdated | Re-index when data changes |
| Tendency to hallucinate | Ground answers in retrieved context |
| Fixed context window | Retrieve only the most relevant chunks |
| Cannot store all docs in weights | Store docs in a vector store |

### The running example (used throughout these notes)

From the basics notes — a **PDF question-answering app**:

> **User asks:** "Explain page 5 like I am a 5-year-old."

To answer correctly, the app must:

```text
1. Load the PDF
2. Split it into pages or chunks
3. Convert chunks into embeddings
4. Store embeddings in a vector store
5. Convert the user query into an embedding
6. Search for the most relevant chunks
7. Put those chunks into the prompt
8. Send the prompt to the LLM
9. Parse and display the final answer
```

LangChain gives you building blocks for every step above.

---

## RAG architecture overview

RAG has two phases: **indexing** (offline, run once) and **querying** (online, run per question).

```text
INDEXING TIME (build the knowledge base)
Documents -> Loader -> Splitter -> Embeddings -> Vector Store

QUERY TIME (answer a user question)
Question -> Retriever -> Context -> Prompt -> LLM -> Answer
```

### Four stages of RAG

| Stage | What happens | When |
|---|---|---|
| **1. Indexing** | Load docs, chunk, embed, store | Once (or when data changes) |
| **2. Retrieval** | Find top-k relevant chunks for the query | Every question |
| **3. Augmentation** | Insert chunks + question into a prompt | Every question |
| **4. Generation** | LLM produces a grounded answer | Every question |

```text
User Query
   ↓
Retriever searches external knowledge base
   ↓
Relevant chunks are returned
   ↓
Chunks + query are inserted into prompt
   ↓
LLM generates grounded answer
```

---

## Prerequisites

Install the packages you need for the examples below:

```bash
pip install langchain langchain-core langchain-community langchain-openai chromadb pypdf
```

Set your API key (OpenAI used in examples; swap for Anthropic, Google, etc. if needed):

```bash
export OPENAI_API_KEY="your-key-here"
```

Project layout for the complete example:

```text
my_rag_app/
├── data/
│   └── storybook.pdf      # your PDF file
└── rag_app.py             # the script below
```

---

## Stage 1: Indexing

Indexing prepares your knowledge base so it can be searched at query time.

```text
Document ingestion -> Text chunking -> Embedding generation -> Vector store
```

### Step 1.1 — Document ingestion (Loader)

Loaders convert raw files into LangChain `Document` objects:

```python
Document(
    page_content="Actual text content",
    metadata={"source": "storybook.pdf", "page": 5}
)
```

**Load a PDF** (one `Document` per page):

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/storybook.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")
print(documents[4].metadata)       # {'source': '...', 'page': 4}  (0-indexed)
print(documents[4].page_content[:200])
```

**Load plain text:**

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/notes.txt", encoding="utf-8")
documents = loader.load()
```

**Load an entire folder:**

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    "data/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
documents = loader.load()
```

**Load a web page:**

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/article")
documents = loader.load()
```

| Loader | Best for |
|---|---|
| `TextLoader` | `.txt` files |
| `PyPDFLoader` | Simple PDFs (one doc per page) |
| `DirectoryLoader` | Many files in a folder |
| `WebBaseLoader` | Static HTML pages |
| `CSVLoader` | Tabular data (one row = one doc) |

For scanned or complex PDFs, use OCR or layout-aware loaders (`UnstructuredPDFLoader`, `PDFPlumberLoader`, etc.).

---

### Step 1.2 — Text chunking (Splitter)

LLMs and embedding models have input limits. Smaller chunks also improve semantic search precision.

**Key terms:**

| Term | Meaning |
|---|---|
| **Chunk size** | Max size of each chunk (characters or tokens) |
| **Chunk overlap** | Text repeated between neighboring chunks to preserve context |

```text
Chunk 1: A B C D E
Chunk 2:       D E F G H
               ↑ overlap
```

**Recursive character splitter** (most common default):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # characters per chunk
    chunk_overlap=200,    # overlap between chunks
    separators=["\n\n", "\n", " ", ""],  # try paragraph, then line, then word
)

chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} pages into {len(chunks)} chunks")
```

**Page-aware splitting** (keep page metadata for "Explain page 5" queries):

```python
# PyPDFLoader already sets metadata["page"] on each document.
# Splitting preserves metadata on every chunk.
for chunk in chunks[:3]:
    print(chunk.metadata.get("page"), chunk.page_content[:80])
```

| Content type | Recommended splitter |
|---|---|
| Plain text | `RecursiveCharacterTextSplitter` |
| Markdown | `MarkdownHeaderTextSplitter` |
| HTML | `HTMLHeaderTextSplitter` |
| Code | Language-aware / code splitter |
| Mixed topics | Semantic chunker (uses embeddings) |

---

### Step 1.3 — Embedding generation

Embeddings convert text into numerical vectors. Similar meaning → similar vectors.

```text
"Virat Kohli is a cricketer" -> [0.12, -0.03, 0.88, ...]
```

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single string
vector = embeddings.embed_query("Explain page 5 like I am a 5-year-old.")
print(len(vector))   # e.g. 1536 dimensions
```

Embeddings are used twice in RAG:

1. **Indexing** — embed every chunk before storing.
2. **Retrieval** — embed the user query, then find closest chunk vectors.

---

### Step 1.4 — Store in vector store

A vector store holds embeddings + original text + metadata.

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="storybook",
    persist_directory="./chroma_db",   # optional: save to disk
)

# Later: add more documents without rebuilding everything
# vectorstore.add_documents(new_chunks)
```

**Other common vector stores:**

| Store | Best for |
|---|---|
| **Chroma** | Local dev, small-to-medium projects |
| **FAISS** | Fast in-memory prototyping |
| **Pinecone / Qdrant / Weaviate** | Production, scaling, filtering |

Common vector store methods:

```python
# Direct similarity search (returns Document objects)
results = vectorstore.similarity_search("What happens on page 5?", k=4)

# Search with scores
results_with_scores = vectorstore.similarity_search_with_score("page 5 story", k=4)
for doc, score in results_with_scores:
    print(score, doc.metadata, doc.page_content[:60])
```

---

## Stage 2: Retrieval

At query time, the **retriever** fetches relevant documents for a question.

```text
query -> query embedding -> similarity search -> top-k chunks
```

### Basic vector store retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},   # return top 4 chunks
)

docs = retriever.invoke("Explain page 5 like I am a 5-year-old.")
for doc in docs:
    print(f"Page {doc.metadata.get('page')}: {doc.page_content[:100]}...")
```

### MMR retriever (reduce duplicate chunks)

**MMR** = Maximal Marginal Relevance. Returns relevant but *diverse* chunks.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)
```

Use MMR when overlapping chunks waste your context window.

### Multi-query retriever (vague questions)

Generates multiple query variations, searches each, merges results:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm,
)
```

### Hybrid retrieval (keyword + semantic)

Combine BM25 (exact keywords) with vector search:

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)
```

| Problem | Retriever to try |
|---|---|
| Basic semantic search | Vector store retriever |
| Duplicate/overlapping chunks | MMR |
| Vague user query | Multi-query retriever |
| Long docs with mixed content | Contextual compression |
| Exact names, IDs, codes | BM25 or ensemble |

---

## Stage 3: Augmentation

Augmentation = building the prompt with retrieved context + user question.

### Format retrieved documents

Retrievers return a list of `Document` objects. The prompt needs a single string:

```python
def format_docs(docs):
    """Turn retrieved documents into one context string."""
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)
```

### Grounded RAG prompt template

Tell the model to answer **only** from context and refuse when context is insufficient:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. "
        "Answer the question only from the provided context. "
        "If the context is insufficient, say you do not know. "
        "When possible, mention the page number you used.",
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}",
    ),
])
```

Plain-text version of the same idea:

```text
You are a helpful assistant.
Answer the question only from the provided context.
If the context is insufficient, say you do not know.

Context:
{context}

Question:
{question}
```

---

## Stage 4: Generation

Pass the augmented prompt to a chat model and parse the output.

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# For creative explanations (like "explain like I'm 5"), slightly higher temperature is fine.
# For factual Q&A, keep temperature low (0–0.3).
```

```text
context + question -> LLM -> grounded response
```

---

## The RAG chain (LCEL)

LangChain Expression Language (LCEL) connects indexing output to query-time logic with the `|` pipe operator.

### Mental model

```text
question --------------------------┐
                                   ↓
question -> retriever -> context -> prompt -> LLM -> parser -> answer
```

In LCEL this becomes a **parallel dict** — fetch context and pass the question through at the same time:

```text
{
  "context": question -> retriever -> format_docs,
  "question": RunnablePassthrough()
}
-> prompt
-> model
-> parser
```

### Chain code

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("Explain page 5 like I am a 5-year-old.")
print(answer)
```

### What happens on `invoke`

```text
1. User question enters the chain
2. Same question goes to retriever -> top-k chunks -> format_docs -> context string
3. Same question passes through as {question}
4. Prompt template fills {context} and {question}
5. LLM generates answer grounded in retrieved pages
6. StrOutputParser returns plain text
```

### Stream the answer (optional)

```python
for chunk in rag_chain.stream("What is the main character's name?"):
    print(chunk, end="", flush=True)
```

---

## Complete example: PDF Q&A app

Full end-to-end script for the **"Explain page 5 like I am a 5-year-old"** use case.

Save as `rag_app.py`:

```python
"""
Complete RAG example: PDF question-answering with LangChain.

User question: "Explain page 5 like I am a 5-year-old."

Flow:
  PDF -> Loader -> Splitter -> Embeddings -> Chroma -> Retriever -> Prompt -> LLM -> Answer
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# --- Config ---
PDF_PATH = "data/storybook.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "storybook"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def format_docs(docs):
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        parts.append(f"[{source} | page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_vectorstore(force_reindex: bool = False):
    """Stage 1: Indexing — load, split, embed, store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not force_reindex and os.path.exists(CHROMA_DIR):
        print("Loading existing vector store...")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    print("Indexing PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages from {PDF_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"  Created {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print("  Vector store saved.")
    return vectorstore


def build_rag_chain(vectorstore):
    """Stages 2–4: Retriever + augmentation + generation."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a friendly teacher explaining stories to a 5-year-old. "
            "Use only the context below. Use simple words and short sentences. "
            "If the context does not contain the answer, say 'I don't know from this book.' "
            "Mention which page you used when helpful.",
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY before running.")

    vectorstore = build_vectorstore(force_reindex=False)
    rag_chain = build_rag_chain(vectorstore)

    # The example question from the LangChain basics notes
    question = "Explain page 5 like I am a 5-year-old."
    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print("-" * 60)
    answer = rag_chain.invoke(question)
    print(answer)
    print("-" * 60)

    # Bonus: inspect what was retrieved (debugging)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved = retriever.invoke(question)
    print("\nRetrieved chunks:")
    for i, doc in enumerate(retrieved, 1):
        print(f"  [{i}] page {doc.metadata.get('page')} — {doc.page_content[:120]}...")


if __name__ == "__main__":
    main()
```

### Run it

```bash
# First run: indexes the PDF (takes longer)
python rag_app.py

# Re-index from scratch after PDF changes
# set force_reindex=True in build_vectorstore() or delete ./chroma_db
```

### Expected flow for our example question

```text
Question: "Explain page 5 like I am a 5-year-old."
   ↓
Retriever embeds the question and finds chunks from page 5 (and nearby pages)
   ↓
format_docs() builds a context string with page labels
   ↓
Prompt tells the LLM to explain simply, only from context
   ↓
LLM returns a child-friendly summary of page 5 content
```

---

## RAG improvements

Once the basic pipeline works, improve quality at each stage.

### 1. Indexing improvements

- Use the right loader for your PDF type (OCR for scans).
- Tune `chunk_size` / `chunk_overlap` (try 500–1500 chars, 10–20% overlap).
- Store rich metadata: `page`, `source`, `section`, `date`.
- Re-index when documents change; version your collections.

### 2. Retrieval improvements

**Pre-retrieval:**

- Query rewriting
- Multi-query generation
- Route queries to the right document collection

**During retrieval:**

- MMR for diversity
- Hybrid BM25 + vector search
- Reranking with a cross-encoder

**Post-retrieval:**

- Contextual compression (keep only relevant sentences)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
)
```

### 3. Augmentation improvements

- Strong grounding instructions ("answer only from context").
- Tell the model to cite page/source.
- Trim context to fit the model's window.

### 4. Generation improvements

- Answers with citations: `"According to page 5, ..."`
- Guardrails and refusal when context is insufficient
- Lower temperature for factual answers

### 5. Evaluation

Tools: **Ragas**, **LangSmith**

| Metric | What it checks |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer relevancy** | Does the answer address the question? |
| **Context precision** | How much retrieved context was useful? |
| **Context recall** | Did retrieval find the necessary information? |

Simple manual check:

```python
question = "Explain page 5 like I am a 5-year-old."
retrieved = retriever.invoke(question)
answer = rag_chain.invoke(question)

print("=== Retrieved ===")
print(format_docs(retrieved))
print("\n=== Answer ===")
print(answer)
# Ask: Does the answer only use retrieved text? Does it match page 5?
```

### 6. System design variants

- **Multimodal RAG** — images, tables, charts
- **Agentic RAG** — agent decides when/how to retrieve
- **Memory-based RAG** — combine chat history with retrieval

---

## Common RAG issues and fixes

| Issue | Possible fix |
|---|---|
| Answer is hallucinated | Stronger grounding prompt, better retrieval, citations |
| Wrong chunks retrieved | Improve chunking, embeddings, retriever settings |
| Too much irrelevant context | Contextual compression, reranking, lower `k` |
| Missing important info | Increase `k`, multi-query retriever, better indexing |
| Duplicate chunks | Use MMR |
| Slow search | Better vector DB, metadata filters, smaller collections |
| Bad PDF extraction | OCR loader, `PDFPlumberLoader`, layout parser |
| "Page 5" queries miss | Preserve `page` metadata; filter by metadata if supported |

---

## RAG checklist

Use this before calling your app production-ready:

- [ ] Load documents correctly (right loader for file type)
- [ ] Split text into useful chunks with overlap
- [ ] Choose good embeddings for your domain
- [ ] Store vectors with metadata (`page`, `source`, etc.)
- [ ] Use the right retriever (`similarity`, MMR, hybrid)
- [ ] Use a grounded prompt with refusal instruction
- [ ] Format context clearly for the LLM
- [ ] Test with the exact questions users will ask
- [ ] Evaluate faithfulness and answer relevancy
- [ ] Add citations when users need to verify answers

---

## Quick reference

```text
INDEX:  Loader -> Splitter -> Embeddings -> VectorStore
QUERY:  Question -> Retriever -> format_docs -> Prompt -> LLM -> Parser

Minimal chain:
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
  | prompt | llm | StrOutputParser()
```

Start with the complete PDF example above. Add MMR, multi-query, or compression only when you hit a real quality or performance problem.
