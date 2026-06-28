> [← LangChain Components](02-langchain-components.md) | [Prompts →](04-prompts.md)

# Models

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
