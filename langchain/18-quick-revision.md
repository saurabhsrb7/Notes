> [← AI Agents](17-ai-agents.md) | [↑ Index](../README.md)

# Quick Revision

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
