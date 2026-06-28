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

Use this when the tool input needs a structured schema.

Good for tools with multiple arguments or validation.

### 3. BaseTool class

`BaseTool` is the abstract base class for tools.

Use it when you need full customization.

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
