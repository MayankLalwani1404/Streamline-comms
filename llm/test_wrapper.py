# test_wrapper.py
from llm_client import LLMClient, LLMError

client = LLMClient()

messages = [
    {"role": "system", "content": "You are a concise helpful customer support assistant for a small restaurant."},
    {"role": "user", "content": "Hii, aapke opening hours kya hai?"}
]

try:
    reply = client.ask(messages)
    print("LLM reply:\n", reply)
except LLMError as e:
    print("LLMError:", e)
