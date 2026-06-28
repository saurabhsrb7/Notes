> [← Tool Calling](16-tool-calling.md) | [Quick Revision →](18-quick-revision.md)

# AI Agents

## What is an AI agent?

An AI agent is an LLM-powered system that can autonomously reason, decide, and take actions using tools/APIs to achieve a goal.

Simple definition:

> An agent is a chatbot with decision-making and tool-use ability.

## Agent example: travel planner

User asks:

```text
Can you create a budget travel itinerary from Delhi to Goa from May 1 to May 7?
```

A normal chatbot may only generate a generic answer.

An agent can:

1. Understand the intent.
2. Search train/flight options.
3. Search hotels.
4. Compare prices.
5. Plan local travel.
6. Create activity plan.
7. Estimate budget.
8. Ask the user to confirm.
9. Make bookings if tools are connected.
10. Send confirmation email.

### Code Implementation

Here is how you can implement this travel planner agent in LangChain using mock tools, a tool-calling LLM, and `AgentExecutor` to orchestrate the decision-making loop:

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------
# Step 1: Define Travel Planner Tools
# ---------------------------------------------------------------------

@tool
def search_flights(source: str, destination: str, date: str) -> str:
    """Search for flights from a source city to a destination city on a specific date."""
    # Simulated database/API lookup
    return (
        f"Flights from {source} to {destination} on {date}:\n"
        f"- IndiGo 6E-501: ₹5,500 (Departs: 06:00, Direct)\n"
        f"- Air India AI-803: ₹7,200 (Departs: 09:30, Direct)\n"
        f"Recommendation: IndiGo (₹5,500) is the cheapest option."
    )

@tool
def search_hotels(location: str, budget_level: str) -> str:
    """Search for hotels in a destination within a budget category ('budget', 'mid-range', 'luxury')."""
    # Simulated search logic
    if budget_level.lower() == "budget":
        return (
            f"Budget hotels in {location}:\n"
            f"- Zostel Hostel: ₹800/night\n"
            f"- CoCo Heritage Resort: ₹2,200/night\n"
            f"Recommendation: CoCo Heritage Resort (₹2,200/night)."
        )
    elif budget_level.lower() == "luxury":
        return f"Luxury hotels in {location}:\n- Taj Exotica: ₹18,000/night."
    else:
        return f"Mid-range hotels in {location}:\n- Radisson Resort: ₹6,500/night."

@tool
def generate_itinerary(destination: str, duration_days: int) -> str:
    """Generate a daily activity itinerary for a destination for a specific duration in days."""
    # Simulated itinerary generator
    return (
        f"{duration_days}-Day Itinerary for {destination}:\n"
        f"- Day 1: Arrive, check-in, visit local beaches (Calangute/Baga).\n"
        f"- Day 2: Historic tour of Old Goa Churches (Basilica of Bom Jesus).\n"
        f"- Day 3: Day trip to Dudhsagar Waterfalls.\n"
        f"- Day 4: Explore South Goa beaches & enjoy sunset cruise.\n"
        f"- Day 5: Visit spice plantations & local temples.\n"
        f"- Day 6: Visit Anjuna Flea Market / local shopping.\n"
        f"- Day 7: Check-out and head to airport."
    )

# List of tools available for the agent
tools = [search_flights, search_hotels, generate_itinerary]

# ---------------------------------------------------------------------
# Step 2: Initialize LLM and Prompt
# ---------------------------------------------------------------------
# We use a tool-calling LLM model (gpt-4o) with temperature 0 for predictable planning.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# The prompt must contain agent_scratchpad to allow the agent to track its thoughts/tool results.
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert AI Travel Planner agent. Your job is to help users design itineraries "
        "by finding flights, hotels, and planning daily activities using your tools. "
        "Be detailed and summarize the total budget."
    )),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ---------------------------------------------------------------------
# Step 3: Create Agent & AgentExecutor
# ---------------------------------------------------------------------
# The agent determines the steps; the executor carries out the steps and executes the tools.
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------------------------------------------------------
# Step 4: Run the Agent
# ---------------------------------------------------------------------
query = "Can you create a budget travel itinerary from Delhi to Goa from May 1 to May 7?"
response = agent_executor.invoke({"input": query})

print("\n--- Final Generated Itinerary ---")
print(response["output"])
```

## Characteristics of agents

| Characteristic | Meaning |
|---|---|
| Goal-driven | User gives the goal, not every step |
| Autonomous planning | Agent breaks goal into subtasks |
| Tool-using | Agent can call APIs/tools |
| Context-aware | Agent maintains history/state |
| Reasoning-capable | Agent decides what to do next |
| Adaptive | Agent changes plan when tool results change |

## ReAct pattern

ReAct means:

```text
Reasoning + Acting
```

The model alternates between thinking and taking actions.

Typical ReAct format:

```text
Question: What is the capital of France and its population?
Thought: I need to find the capital of France first.
Action: search_tool
Action Input: capital of France
Observation: Paris is the capital of France.
Thought: Now I need the population of Paris.
Action: search_tool
Action Input: population of Paris
Observation: Paris has a population of approximately 2.1 million.
Thought: I now know the final answer.
Final Answer: Paris is the capital of France and has a population of approximately 2.1 million.
```

## Why ReAct is useful

ReAct helps the agent:

- Solve multi-step problems
- Use tools at the right time
- Keep reasoning transparent
- Handle tasks where one LLM response is not enough

## Agent vs AgentExecutor

### Agent

The agent decides what to do next.

It may output:

- An action/tool call
- A final answer

### AgentExecutor

The AgentExecutor runs the full loop.

It:

1. Sends input and previous messages to the agent.
2. Gets the next action from the agent.
3. Executes the selected tool.
4. Adds the tool observation to the scratchpad/history.
5. Repeats until the agent returns a final answer.

## Agent scratchpad

The scratchpad stores previous reasoning/action/observation steps.

Example:

```text
Thought: I need to find the capital of France.
Action: search_tool
Action Input: capital of France
Observation: Paris is the capital of France.
```

The scratchpad is passed back into the next LLM call so the agent knows what has already happened.

## Creating an Agent and Executor in LangChain

Here are the two primary ways to implement an agent and its executor loop in LangChain depending on whether the LLM supports native tool calling.

### 1. End-to-End ReAct Agent Example (General LLMs)

The ReAct pattern uses text-based reasoning (`Thought -> Action -> Observation`). We pull the standard ReAct prompt from LangChain Hub.

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------
# Step 1: Define the Tools
# ---------------------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Put the tools in a list
tools = [multiply, add]

# ---------------------------------------------------------------------
# Step 2: Initialize LLM
# ---------------------------------------------------------------------
# ReAct agents usually need temperature=0 for consistent reasoning steps
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------------------
# Step 3: Set up ReAct Prompt
# ---------------------------------------------------------------------
# Pull the pre-defined ReAct prompt template from LangChain Hub.
# This prompt structure instructs the LLM to write:
# Thought -> Action -> Action Input -> Observation -> Thought ... -> Final Answer.
prompt = hub.pull("hwchase17/react")

# ---------------------------------------------------------------------
# Step 4: Create the ReAct Agent (Reasoning Engine)
# ---------------------------------------------------------------------
# The agent is the "brain" that receives user input, decides which tool
# to call, and generates the arguments based on the ReAct loop.
agent = create_react_agent(llm, tools, prompt)

# ---------------------------------------------------------------------
# Step 5: Create the AgentExecutor (Execution Runtime)
# ---------------------------------------------------------------------
# The AgentExecutor handles the execution loop: it sends prompt to LLM,
# receives the tool call action, runs the actual Python function,
# records the output (Observation), and passes it back to the LLM.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Prints the agent's thoughts and tool calls in the console
    handle_parsing_errors=True  # Safely handles LLM formatting errors
)

# ---------------------------------------------------------------------
# Step 6: Execute the Agent
# ---------------------------------------------------------------------
# Run the agent with a complex query requiring multiple tools
response = agent_executor.invoke({"input": "What is 8 multiplied by 7 and then add 10 to it?"})

print("\nFinal Output:")
print(response["output"])
```

### 2. Modern Tool-Calling Agent Example (Recommended for native Tool-Calling LLMs)

For modern LLMs (e.g. `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`) that natively support tool calling, `create_tool_calling_agent` is faster, more reliable, and does not require a custom ReAct text format.

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. Define the tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

tools = [multiply]

# 2. Define the Chat Prompt
# Tool-calling agent prompt needs specific message placeholders for history and the agent scratchpad.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. Initialize LLM & Create Tool-Calling Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)

# 4. Create the Executor Loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# 5. Execute the Agent
response = agent_executor.invoke({"input": "What is 8 multiplied by 7?"})
print(response["output"])
```

## Agent execution flow

```text
AgentExecutor
   ↓
Receive user query
   ↓
Pass query + scratchpad to agent
   ↓
Agent response
   ├── AgentAction -> execute tool -> collect observation -> update scratchpad -> loop again
   └── AgentFinish -> return final output
```

## When should you use agents?

Use agents when:

- The task needs multiple steps.
- The model must decide which tool to use.
- The workflow is not fixed in advance.
- External APIs/tools are needed.
- The user gives a goal, not exact instructions.

Avoid agents when:

- The workflow is fixed and predictable.
- A simple chain is enough.
- Tool use can be hardcoded safely.
- You need strict control and low latency.
