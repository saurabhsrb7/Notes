> [← Tool Calling](16-tool-calling.md) | [↑ Index](../README.md) | [Quick Revision →](18-quick-revision.md)

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

## Creating an agent in LangChain

Conceptual code:

```python
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt,
)
```

This creates the decision-making part.

## Creating an AgentExecutor

Conceptual code:

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
)
```

This runs the full loop.

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
