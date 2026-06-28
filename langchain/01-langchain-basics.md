> [LangChain Components →](02-langchain-components.md)

# LangChain Basics

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
