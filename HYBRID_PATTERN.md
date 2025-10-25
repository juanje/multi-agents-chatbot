# Hybrid Pattern: LangGraph Orchestration + create_agent Specialists

This document explains the **hybrid pattern** demonstrated in `hybrid-chatbot.py`: using LangGraph for workflow orchestration while leveraging `create_agent` for specialized sub-agents.

## The Pattern

```
┌────────────────────────────────────────────────────┐
│           LangGraph Workflow (Orchestration)       │
│                                                    │
│  ┌──────────┐    ┌──────────┐     ┌──────────┐     │
│  │ Chatbot  │───▶│ Analysis │──-─▶│ Answer   │     │
│  │  Node    │    │   Node   │     │   Node   │     │
│  │ (simple) │    │          │     │          │     │
│  └──────────┘    └────┬─────┘     └────┬─────┘     │
│                       │                │           │
│                       ▼                ▼           │
│              ┌─────────────────┐ ┌─────────────┐   │
│              │ create_agent    │ │create_agent │   │
│              │ (classifier)    │ │ (responder) │   │
│              │ - Middleware    │ │ - Middleware│   │
│              │ - temp 0.0      │ │ - temp 0.7  │   │
│              └─────────────────┘ └─────────────┘   │
└────────────────────────────────────────────────────┘
```

## Why This Pattern?

### ✅ Combines Best of Both Worlds

**From LangGraph:**
- Complex workflow orchestration
- Conditional routing
- State management across agents
- Circular flows
- Full visibility and control

**From create_agent:**
- Simplified agent creation
- Middleware support
- Less boilerplate per agent
- Production-ready patterns
- Easy to add features (memory, tools, etc.)

### ✅ Scales Well

For complex applications with:
- Multiple specialized agents
- Complex routing logic
- Need for both control and simplicity
- Team collaboration (different people work on different agents)

## Code Comparison

### Creating Specialized Agents

```python
# Define Pydantic schema for structured output
from pydantic import BaseModel, Field
from typing import Literal

class MessageClassification(BaseModel):
    message_type: Literal["greeting", "question", "other"] = Field(
        description="The type of message"
    )

# Create classification agent with structured output
@dynamic_prompt
def classification_prompt(request: ModelRequest) -> str:
    return "Classify the user's message as..."

classification_agent = create_agent(
    model=ChatOllama(model="llama3.2:3b", temperature=0.0),
    tools=[],
    middleware=[classification_prompt],
    response_format=MessageClassification,  # ✨ Structured output
)

# Create response agent
response_agent = create_agent(
    model=ChatOllama(model="llama3.2:3b", temperature=0.7),
    tools=[],
    middleware=[],
)
```

### Structured Output with create_agent

**LangChain v1.0** integrates structured output directly into `create_agent`:

```python
# ✅ With create_agent (LangChain v1.0)
classification_agent = create_agent(
    model=model,
    response_format=MessageClassification,  # Pass Pydantic model
)

result = classification_agent.invoke({"messages": [...]})
classification = result["structured_response"]  # Get Pydantic object
message_type = classification.message_type  # Type-safe access
```

**Benefits:**
- ✨ Integrated in main loop (no extra LLM call)
- ✅ Returns validated Pydantic object in `result["structured_response"]`
- 🎯 More deterministic than parsing text responses
- 🛡️ Type-safe with automatic validation

### Using Agents in LangGraph Nodes

```python
def analysis_agent(state: State) -> State:
    """LangGraph node that wraps create_agent."""
    user_question = state.get("user_question", "")
    
    # Invoke the specialized agent
    result = classification_agent.invoke({
        "messages": [HumanMessage(content=user_question)]
    })
    
    # Extract structured response (Pydantic object)
    classification = result["structured_response"]
    message_type = classification.message_type
    
    state["message_type"] = message_type
    return state
```

### LangGraph Orchestration

```python
# Same as before - full control over workflow
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot_agent)
graph_builder.add_node("analysis", analysis_agent)  # ← Wraps create_agent
graph_builder.add_node("answer", answer_agent)      # ← Wraps create_agent

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", should_exit, {...})
graph_builder.add_edge("analysis", "answer")
graph_builder.add_edge("answer", "chatbot")

graph = graph_builder.compile()
```

## When to Use This Pattern

### ✅ Use Hybrid Pattern When:

1. **Complex Orchestration Needed**
   - Multiple decision points
   - Conditional routing
   - Parallel execution
   - Human-in-the-loop at workflow level

2. **Specialized Sub-Agents**
   - Each agent has its own tools
   - Different prompting strategies
   - Different models/temperatures
   - Agent-level middleware needed

3. **Team Collaboration**
   - Different teams own different agents
   - Agents developed independently
   - Clear interfaces between agents

4. **Gradual Migration**
   - Moving from manual agents to create_agent
   - Want to refactor piece by piece
   - Need to maintain workflow structure

### ❌ Don't Use Hybrid When:

1. **Simple Linear Flow**
   - Just use `create_agent` directly
   - No need for LangGraph overhead

2. **Single Agent**
   - Overkill for one agent
   - `create_agent` alone is sufficient

3. **Extremely Complex Agents**
   - If individual agents need custom loops
   - Better to use LangGraph for each agent too

## Comparison with Other Approaches

### vs Pure LangGraph (`llm-chatbot.py`)

| Aspect | Pure LangGraph | Hybrid |
|--------|----------------|--------|
| Node Definition | Manual LLM calls | create_agent + wrappers |
| Structured Output | with_structured_output | response_format |
| Per-Agent Middleware | ❌ | ✅ |
| Boilerplate per Agent | Medium | Low |
| Workflow Control | Full | Full |
| Agent Complexity | Any | Best for standard patterns |
| Modularity | Medium | High |

## Real-World Example: Customer Support

```python
# Define specialized agents
email_analyzer = create_agent(
    model="...",
    tools=[extract_intent, extract_priority],
    system_prompt="Analyze customer emails..."
)

knowledge_searcher = create_agent(
    model="...",
    tools=[vector_search, keyword_search],
    system_prompt="Search knowledge base..."
)

response_generator = create_agent(
    model="...",
    tools=[get_template, personalize],
    system_prompt="Generate customer responses..."
)

# Orchestrate with LangGraph
def build_support_workflow():
    graph = StateGraph(SupportState)
    
    graph.add_node("analyze_email", wrap_agent(email_analyzer))
    graph.add_node("search_kb", wrap_agent(knowledge_searcher))
    graph.add_node("generate_response", wrap_agent(response_generator))
    graph.add_node("human_review", human_review_node)
    
    # Complex routing
    graph.add_conditional_edges("analyze_email", route_by_priority, {
        "high": "human_review",
        "medium": "search_kb",
        "low": "search_kb"
    })
    
    graph.add_conditional_edges("generate_response", quality_check, {
        "pass": END,
        "review": "human_review"
    })
    
    return graph.compile()
```

## Benefits at Scale

### 1. **Modularity**
- Agents are self-contained
- Easy to test individually
- Can be developed in parallel

### 2. **Reusability**
- Same agent can be used in multiple workflows
- Agents can be packaged and shared

### 3. **Maintainability**
- Changes to agent logic don't affect workflow
- Changes to workflow don't affect agent internals
- Clear separation of concerns

### 4. **Observability**
- LangGraph gives workflow visibility
- create_agent gives agent-level tracing
- Best of both monitoring approaches

## Migration Strategy

### Step 1: Start with Pure LangGraph
```python
def my_agent(state):
    # Manual implementation
    result = model.invoke(...)
    return state
```

### Step 2: Identify Reusable Agents
- Which agents could benefit from middleware?
- Which agents follow standard patterns?
- Which agents need tools?

### Step 3: Replace with create_agent
```python
my_specialized_agent = create_agent(...)

def my_agent(state):
    result = my_specialized_agent.invoke(...)
    # Parse and update state
    return state
```

### Step 4: Keep LangGraph Orchestration
- Workflow stays the same
- Routing logic unchanged
- State management unchanged

## Best Practices

### 1. **Agent Factory Pattern**
```python
def create_analyzer(temperature: float = 0.0):
    """Factory for creating analyzer agents."""
    return create_agent(
        model=ChatOllama(model="llama3.2:3b", temperature=temperature),
        tools=[...],
        middleware=[...],
    )
```

### 2. **Wrapper Functions**
```python
def wrap_agent(agent, state_in_key, state_out_key, use_structured=False):
    """Generic wrapper to adapt create_agent to LangGraph node."""
    def node_fn(state):
        input_data = state[state_in_key]
        result = agent.invoke({
            "messages": [HumanMessage(content=input_data)]
        })
        
        # Handle structured or text output
        if use_structured:
            state[state_out_key] = result["structured_response"]
        else:
            state[state_out_key] = result["messages"][-1].content
        return state
    return node_fn
```

### 3. **Clear Interfaces**
```python
class AgentInput(TypedDict):
    """Standard input for all agents."""
    messages: list
    context: dict

class AgentOutput(TypedDict):
    """Standard output from all agents."""
    result: str
    metadata: dict
```

### 4. **Structured Output for Determinism**

For agents that need deterministic, validated output, use `response_format`:

```python
from pydantic import BaseModel, Field
from typing import Literal

# Define schema
class Classification(BaseModel):
    category: Literal["support", "sales", "billing"]
    priority: Literal["low", "medium", "high"]
    sentiment: float = Field(ge=0.0, le=1.0)

# Create agent with structured output
classifier_agent = create_agent(
    model=ChatOllama(model="llama3.2:3b", temperature=0.0),
    response_format=Classification,  # ✨ Returns Pydantic object
    middleware=[...],
)

# Use in LangGraph node
def classify_node(state: State) -> State:
    result = classifier_agent.invoke({
        "messages": [HumanMessage(content=state["query"])]
    })
    
    # Access validated Pydantic object
    classification = result["structured_response"]
    state["category"] = classification.category
    state["priority"] = classification.priority
    state["sentiment"] = classification.sentiment
    return state
```

**Why this matters:**
- 🎯 **Deterministic**: Temperature 0.0 + structured output = consistent results
- ✅ **Validated**: Pydantic validates types, ranges, and constraints
- 🛡️ **Type-safe**: IDE autocomplete and type checking
- 🚀 **Efficient**: No extra LLM call (integrated in main loop)

## Conclusion

The hybrid pattern is powerful for **production applications with complex workflows**:

- Use **LangGraph** for orchestration, routing, and state management
- Use **create_agent** for individual specialized agents
- Get the benefits of both: control + simplicity

**When in doubt:**
- Simple app → Pure `create_agent`
- Complex workflow → Hybrid
- Learning/experimentation → Pure LangGraph

