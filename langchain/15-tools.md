> [← Building a RAG App with LangChain](14-building-a-rag-app.md) | [Tool Calling →](16-tool-calling.md)

# Tools

## What is a tool?

A tool is a Python function or API packaged in a way the LLM can understand and request when needed.

Simple definition:

> A tool lets an LLM use external capabilities.

LLMs are good at:

- Reasoning
- Understanding language
- Generating language

But they cannot reliably do some things alone:

- Access live data
- Perform guaranteed math
- Call APIs
- Run code
- Query databases
- Send emails

Tools solve this.

## Tool examples

- Search tool
- Calculator
- Weather API
- Currency conversion API
- SQL database query tool
- Gmail send tool
- Slack message tool
- Python REPL tool
- Shell command tool

## Built-in tools

Built-in tools are tools LangChain already provides.

Examples of built-in tools:

| Tool | Purpose |
|---|---|
| DuckDuckGoSearchRun | Web search |
| WikipediaQueryRun | Wikipedia summary |
| PythonREPLTool | Run Python code |
| ShellTool | Run shell commands |
| RequestsGetTool | Make HTTP GET requests |
| GmailSendMessageTool | Send Gmail messages |
| SlackSendMessageTool | Post to Slack |
| SQLDatabaseQueryTool | Run SQL queries |

### Examples of using Built-in Tools

#### 1. Web Search (`DuckDuckGoSearchRun`)
Requires installing `langchain-community`.

```python
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the search tool
search = DuckDuckGoSearchRun()

# Run search query
result = search.invoke("What is the latest version of LangChain?")
print(result)
```

#### 2. Wikipedia Summary (`WikipediaQueryRun`)
Requires installing `wikipedia` and `langchain-community`.

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Initialize Wikipedia API Wrapper and Tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)

# Query Wikipedia
result = wikipedia.invoke("LangChain")
print(result)
```

#### 3. Python REPL (`PythonREPLTool`)
Requires installing `langchain-experimental`. Used by LLMs to write and execute code.

```python
from langchain_experimental.tools import PythonREPLTool

# Initialize Python REPL
python_repl = PythonREPLTool()

# Execute code snippet
result = python_repl.invoke("print(sum([1, 2, 3, 4]))")
print(result)  # Output: 10\n
```

#### 4. Loading tools with `load_tools`
LangChain provides a helper function to quickly load certain tools by name.

```python
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Load multiple tools (some tools require an LLM for calculation or planning)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
```

## Custom tools

A custom tool is a tool you define yourself.

Use custom tools when:

- You want to call your own APIs.
- You want to wrap business logic.
- You want the LLM to interact with your app, database, or product.

## Ways to create custom tools

### 1. `@tool` decorator

Simple and fast way to convert a Python function into a tool.

Conceptual example:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

### 2. StructuredTool + Pydantic

Use `StructuredTool` when:
- The tool needs multiple inputs.
- You want to define a strict validation schema for inputs using Pydantic.
- You want to specify parameter names, types, and descriptions explicitly so the LLM knows exactly what values to generate.

#### Example:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 1. Define the input schema using Pydantic
class MultiplyInput(BaseModel):
    a: int = Field(description="First number to multiply")
    b: int = Field(description="Second number to multiply")

# 2. Define the Python function containing the logic
def multiply(a: int, b: int) -> int:
    return a * b

# 3. Create the StructuredTool from the function and schema
multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers together.",
    args_schema=MultiplyInput
)

# 4. Invoke the tool
result = multiply_tool.invoke({"a": 8, "b": 7})
print(result)  # Output: 56
```

### 3. BaseTool class

Subclass `BaseTool` when:
- You need absolute control over the tool's behavior, setup, state, or initialization.
- You need to manage internal state or pass complex configuration objects (e.g. API clients, database connections) during initialization.
- You want to implement both synchronous (`_run`) and asynchronous (`_arun`) executions.

When subclassing `BaseTool`, you must:
1. Define the `name` and `description` attributes.
2. Define the `args_schema` attribute referencing a Pydantic model.
3. Implement the `_run` method for synchronous execution.
4. Implement the `_arun` method for asynchronous execution (or raise `NotImplementedError` if async is not supported).

#### Example:

```python
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# 1. Define the input schema using Pydantic
class MultiplyInput(BaseModel):
    a: int = Field(description="First number to multiply")
    b: int = Field(description="Second number to multiply")

# 2. Define the custom tool class inheriting from BaseTool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers together."
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        """Synchronous execution logic."""
        return a * b

    async def _arun(self, a: int, b: int) -> int:
        """Asynchronous execution logic."""
        return self._run(a, b)

# 3. Instantiate the tool
multiply_tool = MultiplyTool()

# 4. Invoke the tool
result = multiply_tool.invoke({"a": 8, "b": 7})
print(result)  # Output: 56
```

## Toolkits

A toolkit is a bundle of related tools.

Example:

```text
GoogleDriveToolkit
├── GoogleDriveCreateFileTool
├── GoogleDriveSearchTool
└── GoogleDriveReadFileTool
```

Toolkits make tools reusable and organized.
