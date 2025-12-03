"""
rag_service.py

Simple RAG service:
- embed user message
- retrieve top-k docs from Qdrant
- build prompt with context
- call LLM via llm_client.LLMClient
- detect/save leads (phone/email/intent) to Supabase (optional)

Run locally:
python rag_service.py
"""

import os
import re
import time
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests

# import your production-ready llm client (from Day 2)
from llm.llm_client import LLMClient, LLMError

load_dotenv()

# Config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "business_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "2"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "leads")

# Initialize models/clients
print("Loading embedding model:", EMBED_MODEL)
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print("Initializing LLM client...")
llm = LLMClient()

# ---- helper functions ----

PHONE_RE = re.compile(r"(\+?\d{2,4}[-\s]?)?(\d{3,5}[-\s]?\d{3,5}[-\s]?\d{3,5})")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

LEAD_INTENT_KEYWORDS = [
    "book", "booking", "reserve", "reservation", "order", "buy", "purchase", "lead", "interested",
    "appointment", "visit", "visit karna", "order karna", "book karna", "order karenge"
]

def detect_lead(user_text):
    """Detect phone/email/intent - returns dict or None"""
    phones = PHONE_RE.findall(user_text)
    emails = EMAIL_RE.findall(user_text)
    phone = None
    if phones:
        # join matched groups (Regex returns tuples sometimes). pick longest digit-looking substring
        for p in phones:
            s = "".join(p) if isinstance(p, tuple) else str(p)
            # extract digits
            digits = re.sub(r"\D", "", s)
            if len(digits) >= 7:
                phone = digits
                break

    email = emails[0] if emails else None

    # intent detection simple keyword match (Hinglish-friendly by lowercasing)
    lower = user_text.lower()
    intent = any(k in lower for k in LEAD_INTENT_KEYWORDS)

    lead = {}
    if phone: lead["phone"] = phone
    if email: lead["email"] = email
    if intent: lead["intent"] = True

    return lead if lead else None

def save_lead_supabase(lead_obj, customer_id=None):
    """Save lead to Supabase via REST post. Returns True/False."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Supabase not configured; skipping lead save.")
        return False

    url = SUPABASE_URL.rstrip("/") + f"/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "Content-Type": "application/json",
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Prefer": "return=minimal"
    }

    payload = {
        "phone": lead_obj.get("phone"),
        "email": lead_obj.get("email"),
        "intent": lead_obj.get("intent", False),
        "source": lead_obj.get("source", "whatsapp"),
        "raw_text": lead_obj.get("raw_text"),
        "customer_id": customer_id,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code in (201, 200):
            print("Lead saved to Supabase.")
            return True
        else:
            print("Supabase error:", r.status_code, r.text[:400])
            return False
    except Exception as e:
        print("Supabase save exception:", e)
        return False

def retrieve_context(user_text, top_k=TOP_K):
    """Embed and query Qdrant, return list of text chunks (most relevant first)"""
    vec = embedder.encode([user_text])[0].tolist()
    # Use qdrant-client search - ensure payloads returned
    # Some versions call 'search' or 'search_points'; try both
    try:
        if hasattr(qdrant, "search"):
            hits = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=top_k, with_payload=True)
        elif hasattr(qdrant, "search_points"):
            hits = qdrant.search_points(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=top_k, with_payload=True)
        else:
            # fallback REST
            url = QDRANT_URL.rstrip("/") + f"/collections/{QDRANT_COLLECTION}/points/search"
            headers = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}
            payload = {"vector": vec, "limit": top_k, "with_payload": True}
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            res.raise_for_status()
            body = res.json()
            hits = body.get("result") or body.get("hits") or []
    except Exception as e:
        print("Qdrant search error:", e)
        return []

    contexts = []
    for h in hits:
        # support object-like or dict result
        if isinstance(h, dict):
            payload = h.get("payload", {})
            text = payload.get("text") or payload.get("content") or json.dumps(payload)
            contexts.append(text)
        else:
            payload = getattr(h, "payload", {}) or {}
            text = payload.get("text") or payload.get("content") or json.dumps(payload)
            contexts.append(text)
    return contexts

# Prompt template (simple)
PROMPT_SYSTEM = (
    "You are a concise, polite customer support assistant for a local business. "
    "You MUST only answer factual questions using the CONTEXT provided. "
    "If the answer is not present in CONTEXT, say: \"Sorry — I don't have that info right now. Would you like me to connect you to a human?\". "
    "Keep replies under 40 words. Match the user's language and tone (Hinglish is fine)."
)

def build_messages(user_text, contexts):
    """
    Build chat messages list for LLM: system + context + user.
    contexts: list of short text chunks
    """
    ctx_block = "\n\n".join(contexts) if contexts else ""
    system_msg = PROMPT_SYSTEM
    if ctx_block:
        system_msg += "\n\nCONTEXT:\n" + ctx_block

    messages = [
        {"role":"system", "content": system_msg},
        {"role":"user", "content": user_text}
    ]
    return messages

def handle_message(user_text, user_meta=None, customer_id=None):
    """
    Full pipeline: retrieve -> prompt -> llm -> detect/save leads -> return reply
    user_meta optional dict (channel, phone, user id, etc.)
    """
    t0 = time.time()
    contexts = retrieve_context(user_text, top_k=TOP_K)
    t1 = time.time()
    messages = build_messages(user_text, contexts)
    # call llm
    try:
        reply = llm.ask(messages)
    except LLMError as e:
        print("LLM error:", e)
        return {"error":"LLM failed", "detail": str(e)}
    t2 = time.time()

    # lead detection
    lead = detect_lead(user_text)
    if lead:
        lead_payload = {
            "phone": lead.get("phone"),
            "email": lead.get("email"),
            "intent": lead.get("intent", False),
            "raw_text": user_text,
            "source": user_meta.get("channel") if user_meta else "unknown"
        }
        saved = save_lead_supabase(lead_payload, customer_id=customer_id)
        lead_payload["saved"] = saved
    else:
        lead_payload = None

    timing = {
        "embed_and_retrieval_s": round(t1 - t0, 3),
        "llm_s": round(t2 - t1, 3),
        "total_s": round(t2 - t0, 3)
    }

    return {
        "reply": reply,
        "contexts": contexts,
        "lead": lead_payload,
        "timing": timing
    }

# ---- simple CLI test when run directly ----
if __name__ == "__main__":
    print("RAG service test harness. Type exit to quit.")
    while True:
        text = input("\nUser text: ").strip()
        if not text:
            continue
        if text.lower() in ("exit","quit"):
            break
        res = handle_message(text, user_meta={"channel":"cli_test"}, customer_id=None)
        print("\n----- REPLY -----")
        if "error" in res:
            print("ERROR:", res["detail"])
        else:
            print("Reply:", res["reply"])
            print("\nContexts returned:")
            for c in res["contexts"]:
                print(" -", c)
            if res["lead"]:
                print("\nLead detected:", res["lead"])
            print("\nTiming:", res["timing"])
        print("-----------------\n")
