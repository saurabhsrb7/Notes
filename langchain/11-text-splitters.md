> [← Document Loaders](10-document-loaders.md) | [Vector Stores →](12-vector-stores.md)

# Text Splitters

## What is text splitting?

Text splitting is the process of breaking large text into smaller chunks.

```text
Large document -> chunks
```

## Why split text?

1. LLMs have context length limits.
2. Embedding models have input size limits.
3. Smaller chunks improve semantic search.
4. Smaller chunks reduce noise in retrieved context.
5. Smaller chunks are easier to process in parallel.

## Important terms

### Chunk size

Maximum size of each chunk.

It can be measured in:

- Characters
- Tokens
- Words

### Chunk overlap

Amount of text repeated between neighboring chunks.

Overlap helps preserve context across boundaries.

Example:

```text
Chunk 1: A B C D E
Chunk 2: D E F G H
```

Here `D E` is overlap.

## Types of text splitting

The rough notes mention four styles:

1. Length-based splitting
2. Text-structure-based splitting
3. Document-structure-based splitting
4. Semantic-meaning-based splitting

---

## 1. Length-based splitting

Splits text after a fixed length.

Example:

```text
Every 100 characters -> new chunk
```

Pros:

- Simple
- Fast
- Predictable

Cons:

- Can break sentences in the middle.
- May lose meaning at chunk boundaries.

## 2. Text-structure-based splitting

Uses text separators like:

- Paragraph break: `\n\n`
- Line break: `\n`
- Space: ` `
- Character fallback

A common approach is recursive splitting:

1. Try to split by paragraph.
2. If chunk is still too large, split by line.
3. If still too large, split by words.
4. If still too large, split by characters.

This is better than pure length splitting because it tries to preserve natural text structure.

## 3. Document-structure-based splitting

Uses document-specific structure.

Examples:

- Markdown headings
- HTML tags
- Python class/function definitions
- Code blocks

Useful for:

- Markdown files
- Code files
- HTML documents
- Technical docs

Example:

```text
Split Markdown by headings:
# Title
## Section
### Subsection
```

For code:

```text
Split Python by class and function definitions.
```

## 4. Semantic-meaning-based splitting

Splits text based on meaning, not just length or separators.

The system checks sentence embeddings and detects where the topic changes.

Example:

- Paragraph about farming
- Paragraph about IPL cricket
- Paragraph about terrorism

A semantic splitter tries to separate these into meaningful chunks.

Pros:

- Better meaning preservation
- Good for mixed-topic documents

Cons:

- More expensive
- Requires embeddings
- More complex

## Choosing a splitter

| Content type | Good splitter |
|---|---|
| Plain text | RecursiveCharacterTextSplitter |
| Markdown | MarkdownHeaderTextSplitter |
| HTML | HTMLHeaderTextSplitter or similar |
| Code | Language-aware/code splitter |
| Mixed topics | Semantic splitter |
| Very simple use case | Character/length splitter |
