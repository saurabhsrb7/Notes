> [← LangChain Basics](01-langchain-basics.md) | [↑ Index](../README.md) | [Models →](03-models.md)

# LangChain Components

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
