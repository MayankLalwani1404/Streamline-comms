# llm_client.py
import os
import time
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TIMEOUT = 30
RETRY_COUNT = 2
RETRY_BACKOFF = 1.0  # seconds

class LLMError(Exception):
    pass

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER")
        if self.provider not in ("groq", "together"):
            raise RuntimeError("LLM_PROVIDER not configured. Set LLM_PROVIDER=groq|together in .env")

    def ask(self, messages: List[Dict[str,str]]) -> str:
        """messages should be a list of {"role": "...", "content": "..."}"""
        if self.provider == "groq":
            return self._ask_groq(messages)
        elif self.provider == "together":
            return self._ask_together(messages)
        raise LLMError("Unsupported provider")

    def _post_with_retries(self, url, headers, payload):
        last_exc = None
        for attempt in range(RETRY_COUNT + 1):
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
            except Exception as e:
                last_exc = e
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            # got response — if server errors, retry a bit
            if res.status_code >= 500 and attempt < RETRY_COUNT:
                last_exc = RuntimeError(f"Server error {res.status_code}")
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            return res
        # exhausted retries
        if last_exc:
            raise LLMError(f"Request failed after retries: {last_exc}")
        raise LLMError("Unknown request failure")

    def _parse_response(self, res):
        # try safe parse JSON
        try:
            data = res.json()
        except Exception:
            raise LLMError(f"Invalid JSON response (status {res.status_code}): {res.text[:200]}")

        # provider shapes should have choices array
        if not isinstance(data, dict):
            raise LLMError(f"Unexpected response shape: {data}")
        if "choices" not in data or not data["choices"]:
            # try to surface error info if available
            err = data.get("error") or data.get("message") or data
            raise LLMError(f"LLM returned no choices. Info: {err}")
        # defensive extraction
        choice = data["choices"][0]
        # some providers place message differently
        if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
            return choice["message"]["content"]
        # fallback to text field (older shapes)
        if "text" in choice:
            return choice["text"]
        raise LLMError("Unable to extract content from response")

    def _ask_groq(self, messages):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": os.getenv("GROQ_MODEL"),
            "messages": messages,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7"))
        }
        res = self._post_with_retries(url, headers, payload)
        if res.status_code != 200:
            # surface JSON error if possible
            try:
                err = res.json()
            except Exception:
                err = res.text
            raise LLMError(f"Groq API error {res.status_code}: {err}")
        return self._parse_response(res)

    def _ask_together(self, messages):
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": os.getenv("TOGETHER_MODEL"),
            "messages": messages
        }
        res = self._post_with_retries(url, headers, payload)
        if res.status_code != 200:
            try:
                err = res.json()
            except Exception:
                err = res.text
            raise LLMError(f"Together API error {res.status_code}: {err}")
        return self._parse_response(res)
