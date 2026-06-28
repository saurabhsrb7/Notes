> [← Chains](07-chains.md) | [↑ Index](../README.md) | [RAG: Retrieval-Augmented Generation →](09-rag-retrieval-augmented-generation.md)

# Runnables and LCEL

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
