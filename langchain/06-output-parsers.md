> [← Structured Output](05-structured-output.md) | [↑ Index](../README.md) | [Chains →](07-chains.md)

# Output Parsers

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
