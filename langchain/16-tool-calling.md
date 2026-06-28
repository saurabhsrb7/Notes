> [← Tools](15-tools.md) | [AI Agents →](17-ai-agents.md)

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

> [!IMPORTANT]
> **The LLM does NOT execute the tool.** The LLM only decides *which* tool to use and generates the arguments for it. The actual execution of the code/API must be done manually by your client-side application.

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

---

## End-to-End Code Example

Here is a complete, working example of tool calling showing binding, calling, execution, and returning the final answer back to the LLM.

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------
# Step 1: Define the Tool
# ---------------------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# Map tool names to the actual functions for dynamic lookup/execution
tools_map = {"multiply": multiply}

# ---------------------------------------------------------------------
# Step 2: Tool Binding
# ---------------------------------------------------------------------
# Bind the tool to the LLM to make it tool-aware
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([multiply])

# ---------------------------------------------------------------------
# Step 3: Tool Calling (LLM Decides and Requests)
# ---------------------------------------------------------------------
messages = [HumanMessage(content="What is 8 multiplied by 7?")]
ai_message = llm_with_tools.invoke(messages)

print("LLM Response (Tool Call Request):")
print(ai_message.tool_calls)
# Output: [{'name': 'multiply', 'args': {'a': 8, 'b': 7}, 'id': 'call_123xyz'}]

# Append the LLM's response (containing tool call requests) to conversation history
messages.append(ai_message)

# ---------------------------------------------------------------------
# Step 4: Tool Execution (Runtime executes the function)
# ---------------------------------------------------------------------
for tool_call in ai_message.tool_calls:
    # 1. Fetch the tool by name from the map
    selected_tool = tools_map[tool_call["name"]]
    
    # 2. Invoke the tool with the LLM-provided arguments
    tool_output = selected_tool.invoke(tool_call["args"])
    
    # 3. Create a ToolMessage with the result and the matching tool_call_id
    tool_message = ToolMessage(
        content=str(tool_output),
        tool_call_id=tool_call["id"]
    )
    
    # 4. Append the tool output to conversation history
    messages.append(tool_message)

# ---------------------------------------------------------------------
# Step 5: Final Response Generation (LLM formulates final response)
# ---------------------------------------------------------------------
# Pass the original question, LLM's tool call request, and tool output back to the LLM
final_response = llm_with_tools.invoke(messages)

print("\nFinal LLM Response:")
print(final_response.content)
# Output: "8 multiplied by 7 is 56."
```
