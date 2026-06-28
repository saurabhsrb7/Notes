> [← Prompts](04-prompts.md) | [Output Parsers →](06-output-parsers.md)

# Structured Output

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
