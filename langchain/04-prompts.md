> [← Models](03-models.md) | [Structured Output →](05-structured-output.md)

# Prompts

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
