> [← Runnables and LCEL](08-runnables-and-lcel.md) | [↑ Index](../README.md) | [Document Loaders →](10-document-loaders.md)

# RAG: Retrieval-Augmented Generation

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

It combines:

1. Information retrieval
2. Language generation

Simple definition:

> RAG retrieves relevant information from an external knowledge base and gives it to the LLM as context before the LLM generates an answer.

## Why RAG?

LLMs have limitations:

- They may not know private company data.
- They may not know recent information.
- They can hallucinate.
- Their context window is limited.
- They cannot store every document inside their weights.

RAG helps solve these problems by injecting relevant external context into the prompt.

## RAG benefits

1. Uses up-to-date information.
2. Works with private data.
3. Reduces hallucinations.
4. Supports large document collections.
5. Does not require fine-tuning for every knowledge update.

## RAG high-level flow

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

## Four stages of RAG

1. Indexing
2. Retrieval
3. Augmentation
4. Generation

---

## 1. Indexing

Indexing prepares the knowledge base so it can be searched efficiently later.

Indexing includes:

1. Document ingestion
2. Text chunking
3. Embedding generation
4. Storage in vector store

### Step 1: Document ingestion

Load source data into LangChain documents.

Sources can include:

- PDFs
- Word files
- Text files
- Websites
- YouTube transcripts
- GitHub repositories
- SQL records
- Internal wikis

### Step 2: Text chunking

Large documents are split into smaller chunks.

Why?

- LLMs have context limits.
- Embedding models have input limits.
- Smaller chunks improve semantic search.
- Smaller chunks reduce irrelevant text.

### Step 3: Embedding generation

Each chunk is converted into a vector.

```text
chunk text -> embedding model -> vector
```

### Step 4: Store in vector store

Store:

- Vector embedding
- Original chunk text
- Metadata, such as source, page number, author, date

---

## 2. Retrieval

Retrieval happens at query time.

The user asks a question. The retriever finds the most relevant chunks from the vector store.

```text
query -> query embedding -> similarity search -> top-k chunks
```

---

## 3. Augmentation

Augmentation combines retrieved chunks with the user question.

Example prompt:

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

## 4. Generation

The LLM uses the augmented prompt to generate the final answer.

```text
context + question -> LLM -> grounded response
```
