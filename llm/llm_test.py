# llm_test.py (debug-friendly)
import os
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER")

def debug_print_response(res):
    print("HTTP status:", res.status_code)
    # Try to print JSON, otherwise print text
    try:
        j = res.json()
        print("Response JSON:")
        print(j)
    except Exception:
        print("Response text:")
        print(res.text)

def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": os.getenv("GROQ_MODEL"),
        "messages": [
            {"role": "system", "content": "You are a helpful business chatbot. Keep responses short and friendly."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    res = requests.post(url, json=payload, headers=headers, timeout=30)
    # Debug output
    if res.status_code != 200:
        print("Groq API returned non-200. See details below.")
        debug_print_response(res)
        # Raise a helpful exception
        raise RuntimeError(f"Groq API error ({res.status_code}). See output above.")
    # safe access
    data = res.json()
    # Guard against missing keys
    if "choices" not in data or not data["choices"]:
        print("Groq response has no 'choices' key or it is empty. Full response below:")
        debug_print_response(res)
        raise RuntimeError("Groq response missing choices.")
    return data["choices"][0]["message"]["content"]


def call_together(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": os.getenv("TOGETHER_MODEL"),
        "messages": [
            {"role": "system", "content": "You are a helpful business chatbot. Keep responses short and friendly."},
            {"role": "user", "content": prompt}
        ]
    }

    res = requests.post(url, json=payload, headers=headers, timeout=30)
    if res.status_code != 200:
        print("Together API returned non-200. See details below.")
        debug_print_response(res)
        raise RuntimeError(f"Together API error ({res.status_code}). See output above.")
    data = res.json()
    if "choices" not in data or not data["choices"]:
        print("Together response has no 'choices' key or it is empty. Full response below:")
        debug_print_response(res)
        raise RuntimeError("Together response missing choices.")
    return data["choices"][0]["message"]["content"]


def ask_llm(prompt):
    if LLM_PROVIDER == "groq":
        return call_groq(prompt)
    elif LLM_PROVIDER == "together":
        return call_together(prompt)
    else:
        raise RuntimeError("LLM_PROVIDER not set in .env (expected 'groq' or 'together').")

if __name__ == "__main__":
    tests = [
        "Hii, mera naam Mayank hai. Aap kaise ho?",
        "Kya aap log delivery karte ho?",
        "Kitne baje tak shop open rehti hai?",
        "Have vegan options?",
        "Aaj ka special kya hai?"
    ]

    print("Using LLM_PROVIDER =", LLM_PROVIDER)
    print("Running mixed-language test prompts...\n")

    for prompt in tests:
        print("──────────────────────────────────────────────")
        print("User:", prompt)
        try:
            response = ask_llm(prompt)
            print("Bot :", response, "\n")
        except Exception as e:
            print("ERROR:", str(e))
            print("(See details printed above.)\n")

    print("──────────────────────────────────────────────")
    print("Test complete.")
