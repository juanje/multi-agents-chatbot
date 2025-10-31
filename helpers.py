import re
from typing import Literal
from pprint import pprint


def _is_greeting(text: str) -> bool:
    """Check if the text contains a greeting."""
    greetings = ["hi", "hello", "hola"]
    pattern = r"\b(" + "|".join(greetings) + r")\b"
    if re.search(pattern, text, re.IGNORECASE) is not None:
        return True
    else:
        return False


def _is_question(text: str) -> bool:
    """Check if the text ends with a question mark."""
    if text.endswith("?"):
        return True
    else:
        return False


def get_message_type(user_question: str) -> Literal["greeting", "question", "other"]:
    """Classify the message type based on content."""
    if _is_greeting(user_question):
        return "greeting"
    elif _is_question(user_question):
        return "question"
    else:
        return "other"


def print_state(state):
    """Print the current state in a formatted way."""
    print("\n>>> Chatbot Agent State:")
    pprint(state)
    print("\n")

def print_history_state(state):
    """Print the current state in a formatted way."""
    print("\n>>> Chatbot Agent history State:\n")
    for s in state:
        values = s.values
        if 'user_question' not in values:
            continue
        else:
            print(f">>> {values['user_question']=}")
        print(f">>> {values['message_type']=}")
        print(f">>> {values['answer']=}")
        print("\n")