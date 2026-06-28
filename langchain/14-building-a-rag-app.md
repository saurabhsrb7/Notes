> [← Retrievers](13-retrievers.md) | [Tools →](15-tools.md)

# Building a RAG App with LangChain

Complete notes for building a **Retrieval-Augmented Generation (RAG)** application — from theory to two full working projects:

| Project | Source | Knowledge base | Vector store |
|---|---|---|---|
| **Project 1** | PDF Q&A app | PDF pages | Chroma (persistent) |
| **Project 2** | [YouTube Transcript Chatbot](youtube-chatbot/rag_using_langchain.ipynb) | YouTube captions | FAISS (in-memory) |

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

### Example use cases in this guide

**Project 1 — PDF question-answering app** (from the basics notes):

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

**Project 2 — YouTube transcript chatbot** (from [`youtube-chatbot/rag_using_langchain.ipynb`](youtube-chatbot/rag_using_langchain.ipynb)):

> **User asks:** "Can you summarize the video?"

Same 9-step pipeline, but step 1 loads a YouTube transcript instead of a PDF.

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

## Project 1: PDF Q&A app

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

## Project 2: YouTube Transcript Chatbot

Full walkthrough of the notebook [`youtube-chatbot/rag_using_langchain.ipynb`](youtube-chatbot/rag_using_langchain.ipynb).

This project builds a **chatbot that answers questions about a YouTube video** using only the video's transcript. No need to watch the full video — ask questions like *"Who is Demis?"* or *"Can you summarize the video?"* and get answers grounded in the captions.

### What video does the notebook use?

| Field | Value |
|---|---|
| **Video ID** | `Gfr50f6ZBvo` |
| **Video** | 3Blue1Brown — *But what is a GPT?* (intro to transformers / LLMs) |
| **Transcript source** | YouTube auto-generated or manual captions via `youtube-transcript-api` |
| **Chunks created** | ~168 chunks (with `chunk_size=1000`, `chunk_overlap=200`) |

### Project structure

```text
LangChain/
├── 14-building-a-rag-app.md          ← this guide
└── youtube-chatbot/
    └── rag_using_langchain.ipynb     ← step-by-step notebook
```

### Architecture

Same four RAG stages as Project 1, but the **document loader** is replaced by a YouTube transcript fetcher:

```text
INDEXING TIME
YouTube video ID -> Transcript API -> plain text -> Splitter -> Embeddings -> FAISS

QUERY TIME
Question -> Retriever (top 4 chunks) -> Prompt -> GPT-4o-mini -> Answer
```

```text
video_id "Gfr50f6ZBvo"
   ↓
YouTubeTranscriptApi.fetch()  →  transcript_list (timed caption segments)
   ↓
Flatten to one string  →  "Imagine you happen across a short movie script..."
   ↓
RecursiveCharacterTextSplitter  →  168 Document chunks
   ↓
OpenAIEmbeddings  →  vectors
   ↓
FAISS.from_documents  →  in-memory vector store
   ↓
retriever.invoke("What is deepmind?")  →  4 most similar chunks
   ↓
PromptTemplate + ChatOpenAI  →  grounded answer
```

### Project 1 vs Project 2

| Aspect | Project 1 (PDF) | Project 2 (YouTube) |
|---|---|---|
| Data source | PDF file on disk | YouTube captions API |
| Loader | `PyPDFLoader` | `youtube-transcript-api` |
| Metadata | `page`, `source` | Plain text only (no timestamps in chunks) |
| Vector store | Chroma (persisted to disk) | FAISS (in-memory) |
| Prompt style | `ChatPromptTemplate` | `PromptTemplate` (string template) |
| Teaching style | One complete script | Notebook: manual steps first, then LCEL chain |
| Example question | "Explain page 5 like I am a 5-year-old." | "Can you summarize the video?" |

---

### Setup

**Install dependencies** (from the notebook):

```bash
pip install youtube-transcript-api langchain-community langchain-openai \
            faiss-cpu tiktoken python-dotenv langchain-text-splitters
```

**Set API key** — never hardcode keys in source code; use an environment variable:

```python
import os
from dotenv import load_dotenv

load_dotenv()
# Requires OPENAI_API_KEY in .env or shell environment
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY first"
```

**Imports used in the notebook:**

```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
```

> **Note:** The notebook imports `RecursiveCharacterTextSplitter` from `langchain.text_splitter`. Prefer `langchain_text_splitters` in new code — same class, updated package path.

---

### Step 1a — Indexing: Document ingestion (fetch transcript)

Instead of loading a file, we fetch captions from YouTube using the **video ID** (the part after `v=` in the URL):

```text
https://www.youtube.com/watch?v=Gfr50f6ZBvo
                              ↑ video_id
```

Each caption segment is a dict with `text`, `start` (seconds), and `duration`:

```python
{'text': 'Imagine you happen across a short movie script that',
 'start': 1.14,
 'duration': 2.836}
```

**Fetch and flatten the transcript:**

```python
video_id = "Gfr50f6ZBvo"

try:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=["en"])
    transcript_list = fetched.to_raw_data()

    # Join all caption segments into one plain-text string
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript[:500], "...")

except TranscriptsDisabled:
    print("No captions available for this video.")
```

**What this step does:**

1. Calls YouTube's caption endpoint (no YouTube API key needed).
2. Requests English captions (`languages=["en"]`).
3. Converts timed segments into one continuous string for chunking.

**Inspect raw segments** (optional — notebook cell 6):

```python
transcript_list[:3]
# [{'text': 'Imagine you happen across...', 'start': 1.14, 'duration': 2.836}, ...]
```

#### API migration note

The notebook originally used the **old static API**:

```python
# Old (removed in youtube-transcript-api v1.2.0)
transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
```

If you see `AttributeError: ... has no attribute 'get_transcript'`, use the **new instance API** shown above:

```python
ytt_api = YouTubeTranscriptApi()
transcript_list = ytt_api.fetch(video_id, languages=["en"]).to_raw_data()
```

---

### Step 1b — Indexing: Text splitting

The full transcript is too long for a single embedding. Split it into overlapping chunks.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# create_documents() wraps a raw string in Document objects
chunks = splitter.create_documents([transcript])

print(len(chunks))   # ~168 for this video
print(chunks[0].page_content[:200])
```

**Why these settings?**

| Parameter | Value | Reason |
|---|---|---|
| `chunk_size` | 1000 | Fits embedding model limits; small enough for precise retrieval |
| `chunk_overlap` | 200 | Keeps context across chunk boundaries (20% overlap) |

**Difference from Project 1:** The PDF app uses `split_documents(pages)` on loader output (keeps `page` metadata). Here `create_documents([transcript])` starts from a single string — chunks have no timestamp metadata unless you add it yourself.

**Optional improvement — preserve timestamps:**

```python
from langchain_core.documents import Document

docs = [
    Document(
        page_content=chunk["text"],
        metadata={"start": chunk["start"], "video_id": video_id},
    )
    for chunk in transcript_list
]
chunks = splitter.split_documents(docs)  # metadata carries into chunks
```

---

### Step 1c & 1d — Indexing: Embeddings + FAISS vector store

Convert each chunk to a vector and store in **FAISS** (Facebook AI Similarity Search).

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(chunks, embeddings)
```

**What happens internally:**

```text
For each of 168 chunks:
  chunk text  →  OpenAIEmbeddings  →  1536-dim vector  →  stored in FAISS index
```

**FAISS vs Chroma (Project 1):**

| | FAISS | Chroma |
|---|---|---|
| Persistence | In-memory by default (lost on restart) | Saves to `./chroma_db` |
| Best for | Notebooks, quick experiments | Apps that re-use the same index |
| Speed | Very fast similarity search | Fast; adds metadata filtering |

**Inspect the vector store** (exploratory cells from the notebook):

```python
# Map FAISS index positions to document IDs
vector_store.index_to_docstore_id
# {0: 'uuid-1', 1: 'uuid-2', ...}

# Fetch a document by its ID
vector_store.get_by_ids(["2436bdb8-3f5f-49c6-8915-0c654c888700"])
```

**Save FAISS to disk** (so you don't re-embed every run):

```python
vector_store.save_local("faiss_youtube_index")

# Load later
vector_store = FAISS.load_local(
    "faiss_youtube_index",
    embeddings,
    allow_dangerous_deserialization=True,
)
```

---

### Step 2 — Retrieval

Create a retriever from the vector store. The retriever embeds the user question and returns the **top-k most similar chunks**.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# Test retrieval
docs = retriever.invoke("What is deepmind")
for i, doc in enumerate(docs, 1):
    print(f"[{i}] {doc.page_content[:120]}...")
```

**Example queries from the notebook:**

| Question | What retrieval should find |
|---|---|
| `"What is deepmind"` | Chunks mentioning DeepMind (if present in transcript) |
| `"who is Demis"` | Chunks about Demis Hassabis |
| `"is the topic of nuclear fusion discussed..."` | Chunks about nuclear fusion, or nothing if not in video |

The 3Blue1Brown video (`Gfr50f6ZBvo`) covers LLMs, transformers, and attention — **not** DeepMind or nuclear fusion. For those questions a well-grounded model should say *"I don't know from the transcript"* after retrieval returns weak or irrelevant chunks.

---

### Step 3 — Augmentation (manual walkthrough)

Before building the chain, the notebook walks through augmentation **step by step** so you can see each piece.

**3.1 — Create the LLM:**

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
```

Low temperature (0.2) keeps answers factual — good for transcript Q&A.

**3.2 — Create the prompt template:**

```python
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"],
)
```

**3.3 — Retrieve documents for a specific question:**

```python
question = (
    "is the topic of nuclear fusion discussed in this video? "
    "if yes then what was discussed"
)
retrieved_docs = retriever.invoke(question)
```

**3.4 — Format retrieved chunks into one context string:**

```python
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(context_text[:500])
```

**3.5 — Fill the prompt template:**

```python
final_prompt = prompt.invoke({"context": context_text, "question": question})
print(final_prompt.to_string())
```

At this point you have a complete prompt ready to send to the LLM — context and question are both filled in.

---

### Step 4 — Generation

Pass the filled prompt to the LLM and read the response:

```python
answer = llm.invoke(final_prompt)
print(answer.content)
```

**Expected behavior for the nuclear fusion question** on this video:

Since the 3Blue1Brown LLM video does not discuss nuclear fusion, a grounded answer should be:

> *"I don't know"* or *"The transcript does not mention nuclear fusion."*

This is correct RAG behavior — the model refuses instead of hallucinating.

---

### Building the LCEL chain

After understanding each step manually, the notebook combines everything into one **LCEL chain** — the same pattern as Project 1.

**5.1 — Helper to format retrieved docs:**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)
```

**5.2 — Parallel chain** (fetch context + pass question at the same time):

```python
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

# Test: see what the parallel chain produces
parallel_chain.invoke("who is Demis")
# {'context': '...retrieved transcript text...', 'question': 'who is Demis'}
```

**5.3 — Full chain** (parallel → prompt → LLM → parser):

```python
parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

answer = main_chain.invoke("Can you summarize the video")
print(answer)
```

**Flow on `invoke("Can you summarize the video")`:**

```text
1. Question enters parallel_chain
2. Retriever finds top-4 transcript chunks about LLMs, transformers, training, etc.
3. format_docs() joins chunks into context string
4. Question passes through unchanged
5. PromptTemplate fills {context} and {question}
6. GPT-4o-mini generates a summary using only those chunks
7. StrOutputParser returns plain text
```

---

### Complete script: YouTube Transcript RAG

Consolidated version of the entire notebook — save as `youtube-chatbot/youtube_rag.py`:

```python
"""
YouTube Transcript RAG — answers questions about a video using its captions.

Based on: youtube-chatbot/rag_using_langchain.ipynb
Video: 3Blue1Brown "But what is a GPT?" (Gfr50f6ZBvo)
"""

import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# --- Config ---
VIDEO_ID = "Gfr50f6ZBvo"
FAISS_DIR = "faiss_youtube_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def fetch_transcript(video_id: str) -> str:
    """Step 1a: Fetch YouTube captions and return plain text."""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en"])
        segments = fetched.to_raw_data()
        return " ".join(seg["text"] for seg in segments)
    except TranscriptsDisabled:
        raise ValueError(f"No captions available for video {video_id}")


def build_vectorstore(video_id: str, force_reindex: bool = False):
    """Steps 1b–1d: Split, embed, and store in FAISS."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not force_reindex and os.path.exists(FAISS_DIR):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )

    print(f"Fetching transcript for {video_id}...")
    transcript = fetch_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([transcript])
    print(f"  Created {len(chunks)} chunks")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_DIR)
    print(f"  FAISS index saved to {FAISS_DIR}/")
    return vector_store


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store):
    """Steps 2–4 + LCEL chain."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY in .env or environment.")

    vector_store = build_vectorstore(VIDEO_ID, force_reindex=False)
    rag_chain, retriever = build_rag_chain(vector_store)

    # Example questions from the notebook
    questions = [
        "What is deepmind",
        "who is Demis",
        "is the topic of nuclear fusion discussed in this video? if yes then what was discussed",
        "Can you summarize the video",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)

        # Show retrieved context (debugging)
        retrieved = retriever.invoke(question)
        print(f"Retrieved {len(retrieved)} chunks")
        for i, doc in enumerate(retrieved, 1):
            print(f"  [{i}] {doc.page_content[:100]}...")

        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
```

**Run it:**

```bash
cd LangChain/youtube-chatbot
pip install youtube-transcript-api langchain-community langchain-openai \
            faiss-cpu tiktoken python-dotenv langchain-text-splitters
export OPENAI_API_KEY="your-key-here"
python youtube_rag.py
```

---

### Notebook cell map

Quick reference — which notebook cell maps to which RAG stage:

| Notebook section | Cells | RAG stage | Key code |
|---|---|---|---|
| Install libraries | 1–3 | Setup | `pip install ...`, imports |
| Step 1a — Document ingestion | 4–6 | Indexing | `YouTubeTranscriptApi().fetch()` |
| Step 1b — Text splitting | 7–10 | Indexing | `RecursiveCharacterTextSplitter` |
| Step 1c & 1d — Embeddings + store | 11–14 | Indexing | `FAISS.from_documents()` |
| Step 2 — Retrieval | 15–18 | Retrieval | `as_retriever(k=4)` |
| Step 3 — Augmentation | 19–26 | Augmentation | `PromptTemplate`, manual context |
| Step 4 — Generation | 27–28 | Generation | `llm.invoke(final_prompt)` |
| Building a Chain | 29–36 | All stages | `RunnableParallel` + LCEL pipe |

---

### Troubleshooting (Project 2)

| Problem | Cause | Fix |
|---|---|---|
| `get_transcript` AttributeError | Old `youtube-transcript-api` API | Use `YouTubeTranscriptApi().fetch(id).to_raw_data()` |
| `TranscriptsDisabled` | Video has no captions | Pick a video with subtitles enabled |
| Wrong or empty answers | Weak retrieval (`k` too low) | Increase `k`, improve chunk size, add metadata |
| Answers about topics not in video | Model hallucinating | Strengthen prompt: "say you don't know" |
| Re-indexing every run | FAISS in-memory only | Call `save_local()` / `load_local()` |
| Hardcoded API key in notebook | Security risk | Use `.env` + `load_dotenv()` instead |

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

| Project | Index pipeline | Query example |
|---|---|---|
| PDF Q&A | `PyPDFLoader` → Chroma | "Explain page 5 like I am a 5-year-old." |
| YouTube chatbot | `YouTubeTranscriptApi` → FAISS | "Can you summarize the video" |

Start with Project 1 or Project 2. Add MMR, multi-query, or compression only when you hit a real quality or performance problem.
