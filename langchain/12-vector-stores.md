> [← Text Splitters](11-text-splitters.md) | [Retrievers →](13-retrievers.md)

# Vector Stores

## Why vector stores?

For semantic search, text is converted into vectors.

A vector store stores these vectors and allows fast similarity search.

```text
Text chunk -> embedding vector -> vector store
```

At query time:

```text
User query -> query vector -> compare with stored vectors -> most similar chunks
```

## What is a vector store?

A vector store is a system that stores and retrieves data represented as numerical vectors.

It usually stores:

- Vector
- Original text
- Metadata

## Key features

### 1. Storage

Keeps vectors and metadata in memory or on disk.

### 2. Similarity search

Finds vectors most similar to a query vector.

### 3. Indexing

Uses special data structures for faster search over high-dimensional vectors.

Example idea:

Instead of comparing against every vector one by one, the system organizes vectors into indexes/clusters.

### 4. CRUD operations

CRUD means:

- Create/add vectors
- Read/search vectors
- Update vectors
- Delete vectors

## Use cases

- Semantic search
- RAG
- Recommender systems
- Image search
- Multimedia search

## Vector store vs vector database

### Vector store

Usually lightweight.

Best for:

- Prototyping
- Local development
- Smaller apps

Example:

- FAISS

May not include:

- Authentication
- Distributed scaling
- ACID transactions
- Rich metadata filtering
- Backup/restore

### Vector database

A more complete production database for vectors.

Often includes:

- Persistence
- Scaling
- Metadata filtering
- Authentication
- Replication
- Backup/restore
- Production deployment features

Examples:

- Chroma
- Pinecone
- Qdrant
- Weaviate
- Milvus

## Vector stores in LangChain

LangChain supports many vector stores through a common interface.

Common operations:

```python
vectorstore = SomeVectorStore.from_documents(docs, embedding_model)
vectorstore.add_documents(new_docs)
results = vectorstore.similarity_search("your query", k=4)
```

Common methods:

- `from_documents(...)`
- `from_texts(...)`
- `add_documents(...)`
- `add_texts(...)`
- `similarity_search(...)`
- metadata-based filtering, depending on backend

## Chroma

Chroma is a lightweight open-source vector database friendly for local development and small-to-medium projects.

Conceptually, Chroma organizes data like:

```text
Tenant
└── Database
    └── Collection
        └── Documents
            ├── Embeddings
            ├── Text
            └── Metadata
```
