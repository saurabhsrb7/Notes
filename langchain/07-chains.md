> [← Output Parsers](06-output-parsers.md) | [Runnables and LCEL →](08-runnables-and-lcel.md)

# Chains

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
