> [← Vector Stores](12-vector-stores.md) | [Building a RAG App with LangChain →](14-building-a-rag-app.md)

# Retrievers

## What is a retriever?

A retriever fetches relevant documents from a data source in response to a user query.

```text
Query -> Retriever -> Relevant documents
```

In LangChain, retrievers are runnables, so you can use them inside LCEL chains.

## Retriever vs vector store

A vector store stores vectors and performs similarity search.

A retriever is the interface used by the app to fetch relevant documents.

Often you create a retriever from a vector store:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

## Types of retrievers

The notes mention several retriever types:

- Wikipedia Retriever
- Vector Store Retriever
- MMR Retriever
- Multi-Query Retriever
- Contextual Compression Retriever
- BM25 Retriever
- ParentDocumentRetriever
- SelfQueryRetriever
- MultiVectorRetriever
- TimeWeightedVectorRetriever
- EnsembleRetriever
- ArxivRetriever

---

## Wikipedia Retriever

Queries Wikipedia and returns relevant articles as LangChain documents.

Flow:

```text
User query -> Wikipedia API -> relevant article text -> Document objects
```

Good for general knowledge lookups.

## Vector Store Retriever

The most common retriever in RAG.

How it works:

1. Store documents in a vector store.
2. Convert each document chunk into an embedding.
3. Convert user query into an embedding.
4. Compare query vector with stored vectors.
5. Return top-k most similar chunks.

## MMR Retriever

MMR stands for **Maximal Marginal Relevance**.

Goal:

> Retrieve results that are relevant to the query but also different from each other.

Problem with normal similarity search:

- Top results may all say almost the same thing.
- Context window gets wasted with duplicate information.
- The LLM gets less diverse context.

MMR solves this by:

1. Picking the most relevant document first.
2. Then picking documents that are relevant but not too similar to already selected documents.

Good for RAG when many chunks overlap.

## Multi-Query Retriever

Sometimes one query does not capture all possible phrasings.

Example query:

```text
How can I stay healthy?
```

This could mean:

- What should I eat?
- How often should I exercise?
- How can I manage stress?
- What habits improve health?

Multi-query retriever uses an LLM to generate multiple versions of the query, searches for each, then combines and deduplicates results.

Flow:

```text
Original query
   ↓
LLM generates q1, q2, q3, q4
   ↓
Retriever searches each query
   ↓
Combine and deduplicate results
```

## Contextual Compression Retriever

Retrieves documents, then compresses them to keep only the relevant parts.

Problem:

A retrieved paragraph may contain one relevant sentence and many irrelevant sentences.

Contextual compression keeps only the relevant part.

Flow:

```text
Query -> Base retriever -> Documents -> Compressor -> Relevant snippets only
```

Use when:

- Documents are long.
- Retrieved chunks contain mixed information.
- You want to reduce context length.
- You want better answer accuracy.

## BM25 Retriever

Keyword-based retriever.

Good for:

- Exact keyword matches
- Names, IDs, codes, technical terms

## Ensemble Retriever

Combines multiple retrievers.

Example:

```text
BM25 keyword search + vector semantic search
```

This is useful for hybrid retrieval.

## Choosing a retrieval strategy

| Problem | Good retriever/strategy |
|---|---|
| Basic semantic search | Vector Store Retriever |
| Duplicate/overlapping chunks | MMR |
| User query is vague | Multi-Query Retriever |
| Long mixed documents | Contextual Compression |
| Exact keyword lookup | BM25 |
| Need keyword + semantic | Ensemble / hybrid retrieval |
| Need metadata-aware search | SelfQueryRetriever |
