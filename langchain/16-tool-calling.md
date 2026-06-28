> [← Tools](15-tools.md) | [↑ Index](../README.md) | [AI Agents →](17-ai-agents.md)

# Tool Calling

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
