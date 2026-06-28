> [← RAG: Retrieval-Augmented Generation](09-rag-retrieval-augmented-generation.md) | [↑ Index](../README.md) | [Text Splitters →](11-text-splitters.md)

# Document Loaders

## What are document loaders?

Document loaders load data from different sources into a standard LangChain `Document` format.

A document usually contains:

```python
Document(
    page_content="Actual text content",
    metadata={"source": "filename.pdf", "page": 1}
)
```

This standard format allows the data to be used later for:

- Splitting
- Embedding
- Retrieval
- RAG

## Common document loaders

| Loader | Use case |
|---|---|
| TextLoader | Load `.txt` files |
| PyPDFLoader | Load PDF pages |
| DirectoryLoader | Load many files from a folder |
| WebBaseLoader | Load text from web pages |
| CSVLoader | Load CSV rows as documents |

## TextLoader

Loads plain text files.

Best for:

- Chat logs
- Transcripts
- Scraped text
- Code snippets
- Any plain `.txt` data

Limitation:

- Works only with text files.

## PyPDFLoader

Loads PDF files and converts each page into a `Document` object.

Example:

```text
PDF with 25 pages -> 25 Document objects
```

Limitation:

- Not ideal for scanned PDFs.
- Not ideal for complex layouts.
- Uses PyPDF internally.

For complex PDF cases:

| PDF type | Better loader option |
|---|---|
| Simple clean PDF | PyPDFLoader |
| Tables/columns | PDFPlumberLoader |
| Scanned/image PDF | UnstructuredPDFLoader or OCR-based loader |
| Need layout/image data | PyMuPDFLoader |
| Best structure extraction | UnstructuredPDFLoader |

## DirectoryLoader

Loads multiple documents from a folder.

Useful for loading many `.txt`, `.pdf`, `.csv`, or other files.

Glob examples:

| Glob pattern | Meaning |
|---|---|
| `*.pdf` | PDF files in current folder |
| `**/*.txt` | Text files in all subfolders |
| `data/*.csv` | CSV files in `data` folder |
| `**/*` | All files recursively |

## Load vs Lazy Load

### `load()`

Eager loading.

- Loads all documents immediately.
- Returns a list of `Document` objects.
- Good for small datasets.

### `lazy_load()`

Lazy loading.

- Loads documents only when needed.
- Returns a generator.
- Good for large files or many files.
- Saves memory.

## WebBaseLoader

Loads text from web pages.

It uses BeautifulSoup to parse HTML.

Best for:

- Blogs
- News articles
- Public static websites

Limitations:

- Does not handle JavaScript-heavy pages well.
- Only loads static HTML content.

For JavaScript-heavy pages, use browser/Selenium-style loaders.

## CSVLoader

Loads CSV rows as LangChain documents.

By default:

```text
1 CSV row -> 1 Document
```

Useful for:

- Product catalogs
- Tabular datasets
- Logs
- Records
