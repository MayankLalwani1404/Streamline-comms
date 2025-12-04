# runtime/rag_service.py
"""
RAG service (Day 6/7). Lazy-inits heavy clients to avoid import-time failures.
Exports: handle_message(user_text, user_meta=None, customer_id=None)
"""

import os
import sys
import json
import re
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add project root to path when run as module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Config from env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "3"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE_LEADS = os.getenv("SUPABASE_TABLE", "leads")

# Lazy singletons
_embedder = None
_qdrant = None
_llm_client = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("Missing sentence-transformers: pip install sentence-transformers") from e
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        try:
            from qdrant_client import QdrantClient
        except Exception as e:
            raise RuntimeError("Missing qdrant-client: pip install qdrant-client") from e
        # If QDRANT_URL not configured, keep None and retrieval will fallback to no-context
        if not QDRANT_URL:
            _qdrant = None
        else:
            _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant

def get_llm():
    global _llm_client
    if _llm_client is None:
        # import local LLM wrapper
        try:
            from llm.llm_client import LLMClient
        except Exception as e:
            raise RuntimeError("Missing llm.llm_client; ensure llm_client.py exists") from e
        _llm_client = LLMClient()
    return _llm_client

# util regex for lead detection
PHONE_RE = re.compile(r"(\+?\d{2,4}[-\s]?)?(\d{3,5}[-\s]?\d{3,5}[-\s]?\d{3,5})")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Client configs directory
CLIENT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "data", "clients")
DEFAULT_CLIENT_CONFIG = {
    "client_id": "default",
    "display_name": "Default",
    "persona": {"tone":"friendly", "language_preference":"auto", "response_length":"short"},
    "lead_rules": {"lead_definition":"phone_or_email_or_intent", "free_leads_per_month":250, "overage_price_per_lead":10},
    "kb_files": [],
    "system_prompt_overrides": ""
}

# ---------------- helpers ----------------
def load_client_config(customer_id):
    if not customer_id:
        return DEFAULT_CLIENT_CONFIG
    path = os.path.join(CLIENT_CONFIG_DIR, f"{customer_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if not cfg.get("client_id"):
            cfg["client_id"] = customer_id
        return cfg
    cfg = dict(DEFAULT_CLIENT_CONFIG)
    cfg["client_id"] = customer_id
    return cfg

HINGLISH_TOKENS = ["mera","kya","hai","kaun","nam","ka","aap","kab","kaha","please","kripya"]
def detect_language(text):
    txt = (text or "").strip()
    if not txt:
        return "other"
    lowered = txt.lower()
    # quick hinglish heuristic
    if any(tok in lowered for tok in HINGLISH_TOKENS) and re.search(r'^[\x00-\x7F]+$', txt):
        return "hinglish"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        lang = detect(txt)
        if lang.startswith("hi"):
            return "hi"
        if lang.startswith("en"):
            return "en"
        return lang
    except Exception:
        return "other"

BASE_SYSTEM_TEMPLATE = (
    "You are a helpful customer support assistant for {client_name}. "
    "Follow persona instructions: tone={tone}; keep responses {response_length}. "
    "ONLY use facts from CONTEXT. If not found, respond: 'Sorry — I don't have that info right now. Would you like me to connect you to a human?'."
)

def build_system_prompt(client_cfg):
    persona = client_cfg.get("persona", {})
    tone = persona.get("tone", "friendly")
    response_length = persona.get("response_length", "short")
    overrides = client_cfg.get("system_prompt_overrides", "")
    prompt = BASE_SYSTEM_TEMPLATE.format(client_name=client_cfg.get("display_name","this business"), tone=tone, response_length=response_length)
    if overrides:
        prompt += "\n\n" + overrides
    return prompt

# ---------------- lead helpers ----------------
def detect_lead(user_text, lead_def="phone_or_email_or_intent"):
    phones = PHONE_RE.findall(user_text or "")
    emails = EMAIL_RE.findall(user_text or "")
    phone = None
    if phones:
        for p in phones:
            s = "".join(p) if isinstance(p, tuple) else str(p)
            digits = re.sub(r"\D","",s)
            if len(digits) >= 7:
                phone = digits
                break
    email = emails[0] if emails else None
    lower = (user_text or "").lower()
    intent = any(k in lower for k in ["book","booking","reserve","order","appointment","visit","lead","interested"])
    if lead_def == "phone_or_email_or_intent":
        if phone or email or intent: return {"phone":phone,"email":email,"intent":intent}
    if lead_def == "phone_and_intent":
        if phone and intent: return {"phone":phone,"email":email,"intent":intent}
    if lead_def == "phone_or_intent":
        if phone or intent: return {"phone":phone,"email":email,"intent":intent}
    return None

def save_lead_supabase(lead_obj, customer_id=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    url = SUPABASE_URL.rstrip("/") + f"/rest/v1/{SUPABASE_TABLE_LEADS}"
    headers = {"Content-Type":"application/json", "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    payload = {
        "phone": lead_obj.get("phone"),
        "email": lead_obj.get("email"),
        "intent": lead_obj.get("intent", False),
        "raw_text": lead_obj.get("raw_text"),
        "source": lead_obj.get("source","unknown"),
        "customer_id": customer_id
    }
    try:
        import requests
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        return r.status_code in (200,201)
    except Exception:
        return False

def get_monthly_lead_count(customer_id):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return 0
    try:
        import requests
        url = SUPABASE_URL.rstrip("/") + f"/rest/v1/{SUPABASE_TABLE_LEADS}"
        start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        query = f"?select=id&customer_id=eq.{customer_id}&created_at=gte.{start}"
        full = url + query
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        r = requests.get(full, headers=headers, timeout=10)
        if r.status_code == 200:
            return len(r.json())
    except Exception:
        pass
    return 0

# ---------------- retrieval ----------------
def retrieve_context(client_cfg, user_text, top_k=TOP_K):
    """Return list of context strings from vector DB for this client."""
    collection = f"{client_cfg.get('client_id')}_kb"
    embedder = get_embedder()
    qdrant = get_qdrant()
    if not qdrant:
        return []
    vec = embedder.encode([user_text])[0].tolist()
    try:
        # prefer modern API
        if hasattr(qdrant, "search"):
            hits = qdrant.search(collection_name=collection, query_vector=vec, limit=top_k, with_payload=True)
            # hits may be objects or dicts
        elif hasattr(qdrant, "search_points"):
            hits = qdrant.search_points(collection_name=collection, query_vector=vec, limit=top_k, with_payload=True)
        else:
            # fallback REST
            import requests
            url = QDRANT_URL.rstrip("/") + f"/collections/{collection}/points/search"
            headers = {"Content-Type":"application/json", "api-key": QDRANT_API_KEY}
            payload = {"vector": vec, "limit": top_k, "with_payload": True}
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            r.raise_for_status()
            hits = r.json().get("result", [])
    except Exception as e:
        # on any qdrant error, return empty to avoid hallucination
        print("qdrant search error:", e)
        return []

    contexts = []
    for h in hits:
        if isinstance(h, dict):
            payload = h.get("payload", {}) or {}
            contexts.append(payload.get("text") or json.dumps(payload))
        else:
            payload = getattr(h, "payload", {}) or {}
            contexts.append(payload.get("text") or json.dumps(payload))
    return contexts

def build_messages(client_cfg, user_text, contexts, detected_lang):
    system = build_system_prompt(client_cfg)
    if detected_lang in ("hi", "hinglish"):
        system += "\n\nReply in the user's language (Hindi or Hinglish) and prefer simple Hindi words or Hinglish, keep tone same as persona."
    messages = [{"role":"system", "content": system}, {"role":"user", "content": user_text}]
    # Optionally we could include contexts as separate system message or as part of prompt; keep simple:
    if contexts:
        # append contexts as extra system message (short)
        ctx_text = "\n\n--- CONTEXT ---\n" + "\n\n".join(contexts[:TOP_K])
        messages.insert(1, {"role":"system", "content": "CONTEXT:" + ctx_text})
    return messages

# ----------------- main API -----------------
def handle_message(user_text, user_meta=None, customer_id=None):
    """
    Main entry. Returns dict:
      { "reply": str, "contexts": [...], "lead": {...}|None, "timing": {"total_s":float} }
    """
    start = time.time()
    client_cfg = load_client_config(customer_id)
    detected = detect_language(user_text)
    user_text_norm = user_text if detected != "hinglish" else re.sub(r'([aeiou])\1{2,}', r'\1', user_text).strip()

    contexts = retrieve_context(client_cfg, user_text_norm)
    # if no context, return safe fallback (do not hallucinate)
    if not contexts:
        reply = (f"Sorry — I don't have information about that for {client_cfg.get('display_name','this business')}. "
                 "Would you like me to connect you to a specialist or look up a nearby business?")
        return {"reply": reply, "contexts": contexts, "lead": None, "timing": {"total_s": time.time() - start}}

    messages = build_messages(client_cfg, user_text_norm, contexts, detected)

    # call LLM
    try:
        llm = get_llm()
        # LLMClient.ask expects messages in whatever shape your llm_client supports
        # we'll call llm.ask(messages)
        reply = llm.ask(messages)
    except Exception as e:
        reply = f"ERROR: LLM failed: {e}"

    # lead detection + saving
    lead = detect_lead(user_text_norm, lead_def=client_cfg.get("lead_rules", {}).get("lead_definition", "phone_or_email_or_intent"))
    lead_payload = None
    if lead:
        current = get_monthly_lead_count(customer_id) if customer_id else 0
        free = client_cfg.get("lead_rules", {}).get("free_leads_per_month", 250)
        overage = current >= free
        lead_payload = {"phone": lead.get("phone"), "email": lead.get("email"), "intent": lead.get("intent"), "raw_text": user_text_norm, "source": user_meta.get("channel") if user_meta else "unknown", "overage": overage}
        saved = save_lead_supabase(lead_payload, customer_id=customer_id)
        lead_payload["saved"] = saved
        lead_payload["current_count"] = current
        lead_payload["free_limit"] = free

    return {"reply": reply, "contexts": contexts, "lead": lead_payload, "timing": {"total_s": time.time() - start}}
