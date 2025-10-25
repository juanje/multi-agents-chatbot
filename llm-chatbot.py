from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from helpers import print_state


## LLM Configuration ##

# Model for classification (deterministic)
classifier_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0,
)

# Model for answer generation (more creative)
answer_model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.7,
)


## Structured Output Schema ##

class MessageClassification(BaseModel):
    message_type: Literal["greeting", "question", "other"] = Field(
        description="The type of message: 'greeting' for greetings, 'question' for questions, 'other' for everything else"
    )


## Graph State ##

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str
    message_type: str
    answer: str


def create_initial_state() -> State:
    return {
        "messages": [{"role": "user", "content": ""}],
        "user_question": "",
        "message_type": "other",
        "answer": "",
    }


## Nodes ##

# Chatbot Agent
def chatbot_agent(state: State) -> State:
    print_state(state)
    chatbot_answer = state.get("answer", "")
    # If the chatbot has no answer, it's the beginning of the conversation
    if not chatbot_answer:
        user_input = input("Ask me anything: ")
    else:
        # If the chatbot has an answer, it's the continuation of the conversation
        user_input = input(f"Chatbot: {chatbot_answer}: \n\nYou: ")

    state.update(
        {
            "messages": [{"role": "user", "content": user_input}],
            "user_question": user_input,
        }
    )
    return state


# Analysis Agent
def analysis_agent(state: State) -> State:
    user_question = state.get("user_question", "")
    
    # Use LLM with structured output to classify the message type
    classifier = classifier_model.with_structured_output(MessageClassification)
    classification_prompt = [
        SystemMessage(
            content="""Classify the user's message into one of these categories:
- 'greeting': if the message is a greeting like hello, hi, hola, hey, good morning, etc.
- 'question': if the message is asking for information or help (contains ?, starts with who/what/where/when/why/how, or is clearly requesting information)
- 'other': for any other type of message (statements, commands, comments, etc.)

Focus on the intent: if they're asking something, it's a question."""
        ),
        HumanMessage(content=user_question),
    ]
    
    result = classifier.invoke(classification_prompt)
    state["message_type"] = result.message_type
    return state


# Answer Agent
def answer_agent(state: State) -> State:
    message_type = state.get("message_type", "other")
    messages = state.get("messages", [])
    
    # Customize system message based on message type
    system_prompts = {
        "greeting": "You are a friendly assistant. Respond warmly to greetings and offer help.",
        "question": "You are a helpful assistant. Answer the user's question clearly and concisely based on the conversation history.",
        "other": "You are a helpful assistant. Respond appropriately to the user's message.",
    }
    system_message = system_prompts.get(message_type, system_prompts["other"])
    
    # Build conversation context
    conversation = [SystemMessage(content=system_message)]
    conversation.extend(messages)
    
    response = answer_model.invoke(conversation)
    
    # Update state with answer AND add it to messages history
    state.update({
        "answer": response.content,
        "messages": [AIMessage(content=response.content)],
    })
    return state


# Conditional routing function
def should_exit(state: State) -> Literal["yes", "no"]:
    user_question = state.get("user_question", "")
    if user_question.lower() == "exit":
        return "yes"
    else:
        return "no"


## Graph ##

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot_agent)
graph_builder.add_node("analysis", analysis_agent)
graph_builder.add_node("answer", answer_agent)

# (START -> chatbot)
graph_builder.add_edge(START, "chatbot")
# if should_exit == yes ==> (chatbot -> END)
# if should_exit == no ==> (chatbot -> analysis)
graph_builder.add_conditional_edges(
    "chatbot",
    should_exit,
    {
        "yes": END,
        "no": "analysis",
    },
)
# (analysis -> answer)
graph_builder.add_edge("analysis", "answer")
# answer -> chatbot
graph_builder.add_edge("answer", "chatbot")

graph = graph_builder.compile()


def run_chatbot():
    print("Multi-Agents Chatbot")
    print("Type 'exit' to quit\n")

    initial_state = create_initial_state()

    try:
        graph.invoke(initial_state)
        print("Bye")
    except Exception as e:
        print(f"Error: {e}")
        print("Bye")


if __name__ == "__main__":
    run_chatbot()

