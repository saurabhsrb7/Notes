# LangChain Notes - Clean and Easy Version

These notes convert the rough LangChain PDF notes into a structured study guide. The goal is to explain each concept in simple language, show how the pieces connect, and give practical mental models for building LLM applications.

---

## Table of Contents

1. [LangChain Basics](#1-langchain-basics)
2. [LangChain Components](#2-langchain-components)
3. [Models](#3-models)
4. [Prompts](#4-prompts)
5. [Structured Output](#5-structured-output)
6. [Output Parsers](#6-output-parsers)
7. [Chains](#7-chains)
8. [Runnables and LCEL](#8-runnables-and-lcel)
9. [RAG: Retrieval-Augmented Generation](#9-rag-retrieval-augmented-generation)
10. [Document Loaders](#10-document-loaders)
11. [Text Splitters](#11-text-splitters)
12. [Vector Stores](#12-vector-stores)
13. [Retrievers](#13-retrievers)
14. [Building a RAG App with LangChain](#14-building-a-rag-app-with-langchain)
15. [Tools](#15-tools)
16. [Tool Calling](#16-tool-calling)
17. [AI Agents](#17-ai-agents)
18. [Quick Revision](#18-quick-revision)

---

# 1. LangChain Basics

## What is LangChain?

LangChain is an open-source framework for building applications powered by Large Language Models, or LLMs.

A simple way to think about it:

> LangChain helps you connect LLMs with prompts, tools, memory, documents, databases, APIs, and workflows.

Without LangChain, you can still call an LLM API directly. But when your application becomes bigger, you need many extra pieces:

- Prompt templates
- Model switching
- Chat memory
- Document loading
- Text splitting
- Embeddings
- Vector stores
- Retrieval
- Output parsing
- Tool calling
- Agents
- Monitoring and evaluation

LangChain gives you reusable building blocks for all of these.

## Why do we need LangChain?

LLM apps are not just one API call.

For example, imagine a PDF question-answering app. The user asks:

> Explain page 5 like I am a 5-year-old.

To answer this properly, the app must:

1. Load the PDF.
2. Split it into pages or chunks.
3. Convert chunks into embeddings.
4. Store embeddings in a vector store.
5. Convert the user query into an embedding.
6. Search for the most relevant chunks.
7. Put those chunks into the prompt.
8. Send the prompt to the LLM.
9. Parse and display the final answer.

LangChain makes this kind of pipeline easier to build.

## Main benefits of LangChain

### 1. Concept of chains

You can connect multiple steps together:

```text
User input -> Prompt -> LLM -> Parser -> Final output
```

Or more complex flows:

```text
User query -> Retriever -> Context -> Prompt -> LLM -> Answer
```

### 2. Model-agnostic development

You can switch between different providers with less code change.

Examples:

- OpenAI
- Anthropic
- Google Gemini
- Hugging Face
- Ollama/local models

### 3. Complete ecosystem

LangChain includes support for:

- Models
- Prompts
- Chains
- Memory
- Document loaders
- Text splitters
- Vector stores
- Retrievers
- Tools
- Agents
- LangSmith tracing/evaluation

### 4. Memory and state handling

LLM API calls are usually stateless. That means the model does not automatically remember the previous conversation unless you pass history again.

LangChain helps manage conversation history and application state.

## What can you build with LangChain?

You can build:

- Conversational chatbots
- AI knowledge assistants
- PDF/document Q&A apps
- Research assistants
- Summarization tools
- Workflow automation systems
- AI agents
- RAG applications
- Database question-answering systems

## Alternatives to LangChain

Some alternatives are:

- **LlamaIndex** - especially strong for data/document indexing and RAG.
- **Haystack** - useful for search, retrieval, and production NLP/RAG pipelines.

---

# 2. LangChain Components

The rough notes divide LangChain into these major components:

1. Models
2. Prompts
3. Chains
4. Memory
5. Indexes / RAG components
6. Agents

## High-level view

```text
LangChain
├── Models
├── Prompts
├── Chains
├── Memory
├── Indexes / RAG
│   ├── Document Loaders
│   ├── Text Splitters
│   ├── Embeddings
│   ├── Vector Stores
│   └── Retrievers
├── Tools
└── Agents
```

## Simple mental model

| Component | Simple meaning | Example |
|---|---|---|
| Model | The AI model you call | GPT, Claude, Gemini, Llama |
| Prompt | Instructions given to the model | "Summarize this in 5 lines" |
| Chain | Multiple steps connected together | Prompt -> LLM -> Parser |
| Memory | Conversation or state history | Last 10 chat messages |
| Index / RAG | External knowledge system | PDF search, vector DB |
| Tool | Function/API the model can call | Calculator, search API |
| Agent | LLM that decides what to do next | Travel planner, research agent |

---

# 3. Models

## What are models in LangChain?

In LangChain, **models** are interfaces for interacting with AI models.

The Model component hides provider-specific API details and gives you a common interface to work with:

- Language models
- Chat models
- Embedding models

## Types of models

```text
Models
├── Language Models
│   ├── LLMs
│   └── Chat Models
└── Embedding Models
```

## Language Models

Language models process, understand, and generate natural language text.

There are two common types:

### 1. LLMs

Traditional LLM interface:

```text
string input -> model -> string output
```

Example:

```text
Input: "Write a poem about cricket"
Output: "Cricket is a game of..."
```

LLMs are good for raw text generation, but modern applications usually prefer chat models.

### 2. Chat Models

Chat models are optimized for conversations.

They work with messages rather than plain strings:

```text
list of messages -> chat model -> AI message
```

A message can have a role:

- `system` - tells the model how to behave
- `human` / `user` - user input
- `ai` / `assistant` - model response

Example:

```python
messages = [
    ("system", "You are a helpful assistant."),
    ("user", "Explain LangChain in simple words."),
]
```

## LLMs vs Chat Models

| Feature | LLMs | Chat Models |
|---|---|---|
| Input | Plain text string | Sequence of messages |
| Output | Plain text string | Chat message |
| Best for | Text generation, summarization | Chatbots, assistants, multi-turn conversations |
| Role awareness | Usually no explicit roles | Understands system/user/assistant roles |
| Modern usage | Less common | More common |

## Embedding Models

Embedding models convert text into numbers.

```text
text -> embedding model -> vector
```

Example:

```text
"Virat Kohli is a cricketer" -> [0.12, -0.03, 0.88, ...]
```

These vectors capture meaning. Similar text will have similar vectors.

Embeddings are used in:

- Semantic search
- RAG
- Recommendation systems
- Clustering
- Similarity matching

## Semantic search mental model

Suppose you have three paragraphs:

1. Paragraph about Virat Kohli
2. Paragraph about Jasprit Bumrah
3. Paragraph about Rohit Sharma

User query:

```text
How many runs has Virat scored?
```

The query is converted into a vector. Each paragraph is also converted into a vector. The system finds the paragraph vector closest to the query vector.

That is semantic search.

## Temperature

Temperature controls randomness in model responses.

| Temperature | Behavior | Use case |
|---|---|---|
| 0.0 - 0.3 | Deterministic, predictable | Math, code, facts |
| 0.5 - 0.7 | Balanced | General Q&A, explanations |
| 0.9 - 1.2 | Creative | Stories, jokes, brainstorming |
| 1.5+ | Highly random | Wild idea generation |

Use low temperature when correctness matters.
Use higher temperature when creativity matters.

## Closed-source models vs open-source models

### Closed-source models

Examples:

- OpenAI GPT models
- Anthropic Claude models
- Google Gemini models

Pros:

- Easy API access
- Strong performance
- Less setup
- Often better instruction-following

Cons:

- Paid API usage
- Data goes to provider servers unless enterprise/private setup is used
- Less control over internals

### Open-source models

Examples from the rough notes:

- Llama models
- Mistral models
- Mixtral models
- Falcon models
- BLOOM
- GPT-Neo / GPT-J style models
- StableLM

Pros:

- More control
- Can run locally
- Can fine-tune
- Better privacy if self-hosted
- No per-token API cost when running locally

Cons:

- Requires hardware, often GPU
- Setup can be complex
- May need CUDA, PyTorch, Transformers, etc.
- Some models may be weaker at instruction-following
- Multimodal support may be limited depending on the model

## Ways to use open-source models

1. Use hosted inference APIs, such as Hugging Face Inference API.
2. Run locally using tools like Ollama or local inference servers.
3. Deploy on your own cloud or on-premise infrastructure.

---

# 4. Prompts

## What is a prompt?

A prompt is the instruction or query given to a model to guide its output.

Example:

```text
Write a 5-line poem about cricket.
```

Prompts can be:

- Text prompts
- Chat prompts
- Multimodal prompts with text + images/audio/video, depending on model support

## Static vs dynamic prompts

### Static prompt

A static prompt is hardcoded.

```python
prompt = "Summarize Attention Is All You Need in simple language."
```

Problem: You must edit the string every time the paper, style, or length changes.

### Dynamic prompt

A dynamic prompt uses variables.

```text
Summarize the research paper titled {paper_name}.
Explanation style: {style}
Explanation length: {length}
```

At runtime, values are inserted into the placeholders.

## PromptTemplate

A `PromptTemplate` is a structured way to create reusable prompts with variables.

Example:

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Summarize {topic} in a {tone} tone."
)

formatted_prompt = prompt.format(topic="cricket", tone="fun")
```

## Why use PromptTemplate instead of f-strings?

You can use f-strings for simple cases, but `PromptTemplate` is better for LangChain workflows because it provides:

1. Validation of required variables
2. Reusability
3. Integration with LangChain chains/runnables
4. Cleaner code for larger applications

## Role-based prompts

Chat models understand roles.

Example:

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced {profession}."),
    ("user", "Tell me about {topic}."),
])

messages = chat_prompt.format_messages(
    profession="doctor",
    topic="viral fever"
)
```

## Main message types

| Message type | Purpose |
|---|---|
| System message | Sets behavior, role, rules, style |
| Human/User message | User input |
| AI/Assistant message | Previous model response |

## ChatPromptTemplate

A `ChatPromptTemplate` creates a list of structured messages.

Example:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("user", "Explain {topic} in simple language."),
])
```

This is useful for chat models because chat models are trained to respond to message roles.

## MessagesPlaceholder

`MessagesPlaceholder` is used when you want to insert chat history dynamically.

Example use case:

```text
System: You are a helpful assistant.
Chat history: inserted here dynamically
User: latest user question
```

In LangChain, this lets you include previous conversation messages at runtime.

Conceptually:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("user", "{question}"),
])
```

## Few-shot prompting

Few-shot prompting means giving examples inside the prompt so the model understands the expected pattern.

Example:

```text
Classify support tickets into Billing Issue, Technical Problem, or General Inquiry.

Ticket: I was charged twice this month.
Category: Billing Issue

Ticket: The app crashes whenever I log in.
Category: Technical Problem

Ticket: Can you explain how to upgrade my plan?
Category: General Inquiry

Ticket: I cannot connect to the internet using your service.
Category:
```

The model learns the pattern from the examples.

## Prompting best practices

- Be clear about the task.
- Specify the output format.
- Provide examples for tricky tasks.
- Tell the model what to do when information is missing.
- Use system messages for role and behavior.
- Use dynamic templates when inputs change.

---

# 5. Structured Output

## What is structured output?

Structured output means the LLM returns data in a fixed format, such as JSON, instead of free text.

Free-form response:

```text
Morning: Visit Eiffel Tower.
Afternoon: Walk through Louvre Museum.
Evening: Dinner near the Seine.
```

Structured response:

```json
[
  {"time": "Morning", "activity": "Visit Eiffel Tower"},
  {"time": "Afternoon", "activity": "Walk through Louvre Museum"},
  {"time": "Evening", "activity": "Dinner near the Seine"}
]
```

## Why do we need structured output?

Structured output is important when the model response must be used by another system.

Common use cases:

1. Data extraction
2. API responses
3. Database insertion
4. Agents and tool calling
5. Classification
6. Sentiment analysis
7. Resume parsing
8. Product review analysis

Example:

```json
{
  "summary": "The product is good but delivery was late.",
  "sentiment": "mixed",
  "pros": ["good product quality"],
  "cons": ["late delivery"]
}
```

## Ways to get structured output

There are two broad ways:

1. Use models that support structured output directly.
2. Use output parsers to parse raw text into structured data.

## `with_structured_output`

Some LangChain chat models support:

```python
structured_model = model.with_structured_output(schema)
```

The schema can be defined using:

- `TypedDict`
- Pydantic model
- JSON Schema

## TypedDict

`TypedDict` defines the expected keys and value types of a dictionary.

Example:

```python
from typing import TypedDict

class Review(TypedDict):
    summary: str
    sentiment: str
```

Important point:

> `TypedDict` helps with type hints, but it does not validate data at runtime.

Use `TypedDict` when:

- You only need basic structure.
- You trust the model output.
- You do not need runtime validation.

## Pydantic

Pydantic validates and parses data at runtime.

Example:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Review(BaseModel):
    summary: str = Field(description="Short summary of the review")
    sentiment: Literal["positive", "neutral", "negative"]
```

Use Pydantic when:

- You need data validation.
- You want default values.
- You want automatic type conversion.
- You want safer production code.

## JSON Schema

JSON Schema is a standard language-independent way to define structure.

Use JSON Schema when:

- You need a standard JSON format.
- You do not want Python-specific objects.
- You want cross-language compatibility.

## When to use what?

| Need | Best choice |
|---|---|
| Only type hints | TypedDict |
| Runtime validation | Pydantic |
| Default values | Pydantic |
| Automatic type conversion | Pydantic |
| Cross-language schema | JSON Schema |
| Simple JSON output | JSON Schema or JSON parser |

---

# 6. Output Parsers

## What are output parsers?

Output parsers convert raw LLM responses into useful formats.

They can convert output into:

- Plain string
- JSON
- CSV
- Python dictionary
- Pydantic object
- Custom structured format

## Why output parsers matter

LLM output may include extra text, explanations, or formatting issues.

Parsers help make output:

- Consistent
- Easier to use in code
- Easier to validate
- Easier to pass to databases/APIs

## StrOutputParser

The simplest parser.

It returns the model output as a plain string.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
```

Typical chain:

```python
chain = prompt | model | StrOutputParser()
```

Use it when you only need text.

## JsonOutputParser

Parses the model output into JSON/dictionary-like data.

Use it when you want the model to return JSON.

Example output:

```json
{
  "fact_1": "...",
  "fact_2": "...",
  "fact_3": "..."
}
```

## StructuredOutputParser

This parser extracts structured JSON based on predefined response fields.

It is useful when you want fixed field names.

Example fields:

```text
name: string
age: integer
city: string
```

## PydanticOutputParser

Uses a Pydantic model to validate output.

Benefits:

- Strict schema enforcement
- Type safety
- Validation
- Converts output into Python objects

## Parser comparison

| Parser | Output type | Validation | Best for |
|---|---|---|---|
| StrOutputParser | String | No | Plain text answers |
| JsonOutputParser | JSON/dict | Basic JSON parsing | JSON responses |
| StructuredOutputParser | Structured JSON | Field-level structure | Simple schemas |
| PydanticOutputParser | Pydantic object | Strong validation | Production structured data |

---

# 7. Chains

## What is a chain?

A chain is a pipeline of steps.

Simple chain:

```text
Input -> Prompt -> LLM -> Output Parser -> Final output
```

Chains are useful because real applications usually require multiple steps.

## Why use chains?

Without chains, you manually pass output from one step to the next.

With chains, you define the workflow once and invoke it as a unit.

## Simple chain

A simple chain has one main flow.

Example:

```text
Topic -> Prompt -> LLM -> Summary
```

## Sequential chain

A sequential chain runs multiple steps one after another.

Example:

```text
English text -> Translate to Hindi -> Summarize -> Final summary
```

Each step depends on the previous step.

## Parallel chain

A parallel chain runs multiple branches at the same time.

Example:

```text
Input document
├── Generate notes
└── Generate quiz

Then combine both outputs.
```

Useful when tasks are independent.

## Conditional chain

A conditional chain routes input based on some condition.

Example:

```text
Feedback -> Sentiment analysis
├── Positive -> Thank-you response
└── Negative -> Escalation email
```

Useful when different inputs require different workflows.

## Legacy chains vs modern LangChain

Older LangChain used many special chain classes:

- `LLMChain`
- `SequentialChain`
- `SimpleSequentialChain`
- `ConversationalRetrievalChain`
- `RetrievalQA`
- `RouterChain`
- `SQLDatabaseChain`

Modern LangChain favors **Runnables** and **LCEL**, which give a more flexible way to build pipelines.

---

# 8. Runnables and LCEL

## Why Runnables?

LangChain has many components:

- Prompt templates
- Models
- Output parsers
- Retrievers
- Tools
- Custom functions

Runnables give all these components a common interface.

A runnable is a unit of work with:

```text
input -> process -> output
```

## Common Runnable methods

| Method | Meaning |
|---|---|
| `invoke()` | Run once on one input |
| `batch()` | Run on multiple inputs |
| `stream()` | Stream output chunks |

## LCEL

LCEL stands for **LangChain Expression Language**.

It lets you connect runnables using the pipe operator `|`.

Instead of writing:

```python
RunnableSequence(prompt, model, parser)
```

You can write:

```python
chain = prompt | model | parser
```

This is cleaner and easier to read.

## RunnableSequence

Runs steps one after another.

```text
R1 -> R2 -> R3
```

Example:

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "AI"})
```

## RunnableParallel

Runs multiple runnables in parallel on the same input.

Example:

```text
Input topic: AI
├── Generate tweet
└── Generate LinkedIn post

Output:
{
  "tweet": "...",
  "linkedin": "..."
}
```

Conceptual code:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel({
    "tweet": tweet_chain,
    "linkedin": linkedin_chain,
})
```

## RunnablePassthrough

Returns the input unchanged.

Useful when you want to keep the original input while also running another branch.

Example:

```text
Input topic -> generate joke
            -> also pass original topic forward
```

## RunnableLambda

Wraps a normal Python function as a runnable.

Use it for:

- Preprocessing
- Cleaning text
- Counting words
- Formatting output
- API calls
- Filtering
- Custom transformations

Example:

```python
from langchain_core.runnables import RunnableLambda

count_words = RunnableLambda(lambda text: len(text.split()))
```

## RunnableBranch

Works like an `if/elif/else` block.

Example:

```text
Input email -> classify intent
├── complaint -> complaint chain
├── refund -> refund chain
└── general -> general support chain
```

Useful for routing workflows.

## LCEL mental model

```python
chain = prompt | model | parser
```

Means:

```text
1. Format the prompt.
2. Send it to the model.
3. Parse the model output.
```

---

# 9. RAG: Retrieval-Augmented Generation

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

---

# 10. Document Loaders

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

---

# 11. Text Splitters

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

---

# 12. Vector Stores

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

---

# 13. Retrievers

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

---

# 14. Building a RAG App with LangChain

## Basic RAG architecture

```text
Indexing time:
Documents -> Loader -> Splitter -> Embeddings -> Vector Store

Query time:
Question -> Retriever -> Context -> Prompt -> LLM -> Answer
```

## Simple RAG chain mental model

```text
question --------------------------┐
                                   ↓
question -> retriever -> context -> prompt -> LLM -> parser -> answer
```

In LCEL, this often becomes a parallel chain:

```text
{
  "context": retriever,
  "question": RunnablePassthrough()
}
-> prompt
-> model
-> parser
```

Conceptual code:

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What is this document about?")
```

## Prompt for grounded RAG answers

```text
You are a helpful assistant.
Answer the question only from the provided context.
If the context is insufficient, say you do not know.

Context:
{context}

Question:
{question}
```

## RAG improvements

The rough notes list improvements across the whole RAG system.

### 1. UI enhancements

Example:

- Chrome extension
- Better side panel
- More user-friendly upload/search interface

### 2. Evaluation

Tools:

- Ragas
- LangSmith

Common RAG metrics:

| Metric | What it checks |
|---|---|
| Faithfulness | Is the answer grounded in the retrieved context? |
| Answer relevancy | Does the answer address the user question? |
| Context precision | How much retrieved context was actually useful? |
| Context recall | Did retrieval find the necessary information? |

### 3. Indexing improvements

- Better document ingestion
- Better text splitting
- Better vector store choice
- Store useful metadata

### 4. Retrieval improvements

Pre-retrieval:

- Query rewriting
- Multi-query generation
- Domain-aware routing

During retrieval:

- MMR
- Hybrid retrieval
- Reranking

Post-retrieval:

- Contextual compression

### 5. Augmentation improvements

- Better prompt templates
- Answer grounding instructions
- Context window optimization

### 6. Generation improvements

- Answer with citations
- Guardrails
- Refusal when context is insufficient

### 7. System design improvements

- Multimodal RAG
- Agentic RAG
- Memory-based RAG

## Common RAG issues and fixes

| Issue | Possible fix |
|---|---|
| Answer is hallucinated | Stronger grounding prompt, better retrieval, citations |
| Wrong chunks retrieved | Improve chunking, embeddings, retriever settings |
| Too much irrelevant context | Contextual compression, reranking |
| Missing important info | Increase `k`, use multi-query, improve indexing |
| Duplicate chunks | Use MMR |
| Slow search | Better vector DB/indexing, smaller dataset filters |
| Bad PDF extraction | Use better loader/OCR/layout parser |

---

# 15. Tools

## What is a tool?

A tool is a Python function or API packaged in a way the LLM can understand and request when needed.

Simple definition:

> A tool lets an LLM use external capabilities.

LLMs are good at:

- Reasoning
- Understanding language
- Generating language

But they cannot reliably do some things alone:

- Access live data
- Perform guaranteed math
- Call APIs
- Run code
- Query databases
- Send emails

Tools solve this.

## Tool examples

- Search tool
- Calculator
- Weather API
- Currency conversion API
- SQL database query tool
- Gmail send tool
- Slack message tool
- Python REPL tool
- Shell command tool

## Built-in tools

Built-in tools are tools LangChain already provides.

Examples from the rough notes:

| Tool | Purpose |
|---|---|
| DuckDuckGoSearchRun | Web search |
| WikipediaQueryRun | Wikipedia summary |
| PythonREPLTool | Run Python code |
| ShellTool | Run shell commands |
| RequestsGetTool | Make HTTP GET requests |
| GmailSendMessageTool | Send Gmail messages |
| SlackSendMessageTool | Post to Slack |
| SQLDatabaseQueryTool | Run SQL queries |

## Custom tools

A custom tool is a tool you define yourself.

Use custom tools when:

- You want to call your own APIs.
- You want to wrap business logic.
- You want the LLM to interact with your app, database, or product.

## Ways to create custom tools

### 1. `@tool` decorator

Simple and fast way to convert a Python function into a tool.

Conceptual example:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

### 2. StructuredTool + Pydantic

Use this when the tool input needs a structured schema.

Good for tools with multiple arguments or validation.

### 3. BaseTool class

`BaseTool` is the abstract base class for tools.

Use it when you need full customization.

## Toolkits

A toolkit is a bundle of related tools.

Example:

```text
GoogleDriveToolkit
├── GoogleDriveCreateFileTool
├── GoogleDriveSearchTool
└── GoogleDriveReadFileTool
```

Toolkits make tools reusable and organized.

---

# 16. Tool Calling

Tool calling has three different steps:

1. Tool binding
2. Tool calling
3. Tool execution

These are related but not the same.

---

## 1. Tool binding

Tool binding means registering tools with the LLM.

After binding, the LLM knows:

1. What tools are available
2. What each tool does
3. What input schema each tool expects

A tool has:

- Name
- Description
- Input schema

Conceptually:

```text
LLM + tool descriptions + input schemas -> tool-aware LLM
```

---

## 2. Tool calling

Tool calling is when the LLM decides it needs a tool and returns a tool call request.

Important:

> The LLM does not actually run the tool. It only suggests the tool name and arguments.

Example user question:

```text
What is 8 multiplied by 7?
```

LLM may output something like:

```json
{
  "tool": "multiply",
  "args": {"a": 8, "b": 7}
}
```

---

## 3. Tool execution

Tool execution is when LangChain or your code actually runs the Python function/API.

Example:

```python
multiply(a=8, b=7)
# returns 56
```

Then the tool result is passed back to the LLM or returned to the user.

## Tool calling flow

```text
User asks question
   ↓
LLM decides tool is needed
   ↓
LLM returns tool name + arguments
   ↓
Runtime executes actual tool
   ↓
Tool returns observation/result
   ↓
LLM uses result to produce final answer
```

## Currency conversion example

User asks:

```text
Convert 10 USD to INR.
```

A good tool setup may separate this into two tools:

1. `get_conversion_factor("USD", "INR")`
2. `convert(amount=10, rate=85.3415)`

Flow:

```text
User: Convert 10 USD to INR
LLM: I need the rate first.
Tool call: get_conversion_factor(USD, INR)
Tool result: 85.3415
LLM: Now I can multiply.
Tool call: convert(10, 85.3415)
Tool result: 853.415 INR
LLM final answer: 10 USD is 853.415 INR at the current rate.
```

Important design point:

> Do not ask the LLM to fill arguments that should come from the runtime. Let your system inject those values after previous tool calls.

---

# 17. AI Agents

## What is an AI agent?

An AI agent is an LLM-powered system that can autonomously reason, decide, and take actions using tools/APIs to achieve a goal.

Simple definition:

> An agent is a chatbot with decision-making and tool-use ability.

## Agent example: travel planner

User asks:

```text
Can you create a budget travel itinerary from Delhi to Goa from May 1 to May 7?
```

A normal chatbot may only generate a generic answer.

An agent can:

1. Understand the intent.
2. Search train/flight options.
3. Search hotels.
4. Compare prices.
5. Plan local travel.
6. Create activity plan.
7. Estimate budget.
8. Ask the user to confirm.
9. Make bookings if tools are connected.
10. Send confirmation email.

## Characteristics of agents

| Characteristic | Meaning |
|---|---|
| Goal-driven | User gives the goal, not every step |
| Autonomous planning | Agent breaks goal into subtasks |
| Tool-using | Agent can call APIs/tools |
| Context-aware | Agent maintains history/state |
| Reasoning-capable | Agent decides what to do next |
| Adaptive | Agent changes plan when tool results change |

## ReAct pattern

ReAct means:

```text
Reasoning + Acting
```

The model alternates between thinking and taking actions.

Typical ReAct format:

```text
Question: What is the capital of France and its population?
Thought: I need to find the capital of France first.
Action: search_tool
Action Input: capital of France
Observation: Paris is the capital of France.
Thought: Now I need the population of Paris.
Action: search_tool
Action Input: population of Paris
Observation: Paris has a population of approximately 2.1 million.
Thought: I now know the final answer.
Final Answer: Paris is the capital of France and has a population of approximately 2.1 million.
```

## Why ReAct is useful

ReAct helps the agent:

- Solve multi-step problems
- Use tools at the right time
- Keep reasoning transparent
- Handle tasks where one LLM response is not enough

## Agent vs AgentExecutor

### Agent

The agent decides what to do next.

It may output:

- An action/tool call
- A final answer

### AgentExecutor

The AgentExecutor runs the full loop.

It:

1. Sends input and previous messages to the agent.
2. Gets the next action from the agent.
3. Executes the selected tool.
4. Adds the tool observation to the scratchpad/history.
5. Repeats until the agent returns a final answer.

## Agent scratchpad

The scratchpad stores previous reasoning/action/observation steps.

Example:

```text
Thought: I need to find the capital of France.
Action: search_tool
Action Input: capital of France
Observation: Paris is the capital of France.
```

The scratchpad is passed back into the next LLM call so the agent knows what has already happened.

## Creating an agent in LangChain

Conceptual code:

```python
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt,
)
```

This creates the decision-making part.

## Creating an AgentExecutor

Conceptual code:

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
)
```

This runs the full loop.

## Agent execution flow

```text
AgentExecutor
   ↓
Receive user query
   ↓
Pass query + scratchpad to agent
   ↓
Agent response
   ├── AgentAction -> execute tool -> collect observation -> update scratchpad -> loop again
   └── AgentFinish -> return final output
```

## When should you use agents?

Use agents when:

- The task needs multiple steps.
- The model must decide which tool to use.
- The workflow is not fixed in advance.
- External APIs/tools are needed.
- The user gives a goal, not exact instructions.

Avoid agents when:

- The workflow is fixed and predictable.
- A simple chain is enough.
- Tool use can be hardcoded safely.
- You need strict control and low latency.

---

# 18. Quick Revision

## LangChain in one line

LangChain helps build LLM apps by connecting models, prompts, chains, memory, documents, retrievers, tools, and agents.

## Core pipeline patterns

### Basic LLM app

```text
Input -> Prompt -> LLM -> Output
```

### Parsed LLM app

```text
Input -> Prompt -> LLM -> Output Parser -> Structured/Text Output
```

### RAG app

```text
Question -> Retriever -> Context -> Prompt -> LLM -> Answer
```

### Agent app

```text
Goal -> Agent thinks -> Tool call -> Observation -> Agent thinks again -> Final answer
```

## Most important differences

| Concept | Meaning |
|---|---|
| Prompt | Instruction to model |
| PromptTemplate | Reusable dynamic prompt |
| Model | LLM/chat/embedding interface |
| Parser | Converts model output into desired format |
| Chain | Connects multiple steps |
| Runnable | Standard unit with invoke/batch/stream |
| LCEL | Pipe syntax for runnables |
| Document Loader | Loads data into Document objects |
| Text Splitter | Breaks documents into chunks |
| Embedding Model | Converts text into vectors |
| Vector Store | Stores vectors and supports similarity search |
| Retriever | Fetches relevant documents |
| RAG | Adds retrieved context before generation |
| Tool | External function/API callable by LLM |
| Agent | LLM system that decides actions using tools |

## Choosing the right component

| Task | Use |
|---|---|
| Ask model a simple question | Model + prompt |
| Reuse prompt with variables | PromptTemplate |
| Chat with roles/history | ChatPromptTemplate + MessagesPlaceholder |
| Need JSON output | Structured output / parser |
| Need validation | Pydantic |
| Multi-step fixed workflow | Chain / LCEL |
| Need external documents | RAG |
| Need semantic search | Embeddings + vector store + retriever |
| Need live API/database access | Tools |
| Need autonomous decisions | Agent |

## RAG checklist

- [ ] Load documents correctly.
- [ ] Split text into useful chunks.
- [ ] Choose good embeddings.
- [ ] Store vectors with metadata.
- [ ] Use the right retriever.
- [ ] Use a grounded prompt.
- [ ] Tell the model to say "I don't know" when context is insufficient.
- [ ] Evaluate faithfulness and answer relevance.
- [ ] Add citations if needed.

## Tool calling checklist

- [ ] Define tool function.
- [ ] Give tool a clear name.
- [ ] Add a clear description.
- [ ] Define input schema.
- [ ] Bind tool to model.
- [ ] Let model request tool call.
- [ ] Execute tool in your runtime.
- [ ] Pass result back to model.
- [ ] Return final answer.

## Agent checklist

- [ ] Define tools.
- [ ] Create agent prompt.
- [ ] Create agent.
- [ ] Create AgentExecutor.
- [ ] Enable verbose logs while debugging.
- [ ] Track scratchpad/history.
- [ ] Add guardrails for tool use.
- [ ] Prefer simpler chains when autonomy is not needed.

## Final mental model

LangChain applications usually move through this maturity path:

```text
Basic LLM call
   ↓
Prompt templates
   ↓
Chains / LCEL
   ↓
Structured outputs
   ↓
RAG with documents
   ↓
Tools
   ↓
Agents
```

Start simple. Add RAG, tools, or agents only when the application truly needs them.
