from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from helpers import get_message_type, print_state


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
    state["message_type"] = get_message_type(user_question)
    return state


# Answer Agent
def answer_agent(state: State) -> State:
    message_type = state.get("message_type", "other")
    responses = {
        "greeting": "Hello! How can I help you today?",
        "question": "I'm sorry, I don't know the answer to that question.",
        "other": "I'm sorry, I don't understand. Please try again.",
    }
    state["answer"] = responses[message_type]
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
