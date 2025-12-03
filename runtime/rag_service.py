# rag_service.py  (Day 6 - persona + language detection + lead quota)
import os, sys, json, time, re
from datetime import datetime, timezone
from dotenv import load_dotenv

# Add project root to path (if running as script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

# Reuse previous imports
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests

from llm.llm_client import LLMClient, LLMError

# New deps
from langdetect import detect, DetectorFactory
from rapidfuzz import fuzz

DetectorFactory.seed = 0  # deterministic

# Config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "2"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE_LEADS = os.getenv("SUPABASE_TABLE", "leads")
SUPABASE_TABLE_COUNTS = os.getenv("SUPABASE_LEAD_COUNTS_TABLE", "lead_counters")

# Initialize
embedder = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = LLMClient()

# util regex
PHONE_RE = re.compile(r"(\+?\d{2,4}[-\s]?)?(\d{3,5}[-\s]?\d{3,5}[-\s]?\d{3,5})")
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Where client configs live
CLIENT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "data", "clients")
DEFAULT_CLIENT_CONFIG = {
    "client_id": "default",
    "display_name": "Default",
    "persona": {
        "tone":"friendly",
        "language_preference":"auto",
        "response_length":"short"
    },
    "lead_rules": {
        "lead_definition":"phone_or_email_or_intent",
        "free_leads_per_month":250,
        "overage_price_per_lead":10
    },
    "kb_files": [],
    "system_prompt_overrides": ""
}

# -----------------------------------------
# Helpers: load client config
# -----------------------------------------
def load_client_config(customer_id):
    if not customer_id:
        return DEFAULT_CLIENT_CONFIG
    path = os.path.join(CLIENT_CONFIG_DIR, f"{customer_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Ensure client_id exists (fallback to filename)
        if not cfg.get("client_id"):
            cfg["client_id"] = customer_id
        return cfg
    # if not found, return default but include requested id for logging
    cfg = dict(DEFAULT_CLIENT_CONFIG)
    cfg["client_id"] = customer_id
    return cfg


# -----------------------------------------
# Helpers: language detection & normalization
# -----------------------------------------
def normalize_romanized(text):
    # simple normalizer: collapse repeated chars, lower, trim spaces
    t = re.sub(r'([aeiou])\1{2,}', r'\1', text)  # collapse long vowels like "hiiiii" -> "hii"
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)  # reduce long repeats
    t = t.strip().lower()
    return t

HINGLISH_INDICATORS = ["mera", "kya", "hai", "kaun", "nam", "ka", "aap", "kab", "kaha", "please", "kripya", "kripya"]
def detect_language(text):
    """
    Return: ('hi'|'en'|'other'|'hinglish')
    - hinglish: Latin-script Hindi words detected (common case)
    - hi/en: language detected by langdetect
    """
    txt = text.strip()
    if not txt:
        return "other"
    # quick Hinglish heuristic: contains common Hindi tokens in Latin text
    lowered = txt.lower()
    latin_letters = bool(re.search(r'^[\x00-\x7F]+$', txt))
    for token in HINGLISH_INDICATORS:
        if token in lowered and latin_letters:
            return "hinglish"
    try:
        lang = detect(txt)
        if lang.startswith("hi"):
            return "hi"
        if lang.startswith("en"):
            return "en"
        return lang
    except Exception:
        return "other"

# -----------------------------------------
# Helpers: build persona/system prompt
# -----------------------------------------
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
    prompt = BASE_SYSTEM_TEMPLATE.format(client_name=client_cfg.get("display_name","this business"),
                                         tone=tone, response_length=response_length)
    if overrides:
        prompt += "\n\n" + overrides
    return prompt

# -----------------------------------------
# Lead detection & quota enforcement
# -----------------------------------------
def detect_lead(user_text, lead_def="phone_or_email_or_intent"):
    phones = PHONE_RE.findall(user_text)
    emails = EMAIL_RE.findall(user_text)
    phone = None
    if phones:
        for p in phones:
            s = "".join(p) if isinstance(p, tuple) else str(p)
            digits = re.sub(r"\D","",s)
            if len(digits) >= 7:
                phone = digits
                break
    email = emails[0] if emails else None
    lower = user_text.lower()
    intent = any(k in lower for k in ["book", "booking", "reserve", "order","appointment","visit","lead","interested"])
    # Evaluate lead definition
    if lead_def == "phone_or_email_or_intent":
        if phone or email or intent: return {"phone":phone,"email":email,"intent":intent}
    if lead_def == "phone_and_intent":
        if phone and intent: return {"phone":phone,"email":email,"intent":intent}
    if lead_def == "phone_or_intent":
        if phone or intent: return {"phone":phone,"email":email,"intent":intent}
    return None

# Supabase helper: increment & check monthly counters
def ensure_lead_counters_table():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    url = SUPABASE_URL.rstrip("/") + "/rest/v1/" + SUPABASE_TABLE_COUNTS
    # Try a simple HEAD or GET; if 404, create table using SQL via the SQL API is not normally allowed via REST.
    # We'll assume admin created the table; otherwise fallback to counting via leads table queries.
    return True

def get_monthly_lead_count(customer_id):
    # fallback to counting leads from leads table if counters not present
    if not SUPABASE_URL or not SUPABASE_KEY:
        return 0
    try:
        url = SUPABASE_URL.rstrip("/") + f"/rest/v1/{SUPABASE_TABLE_LEADS}"
        # filter: customer_id=eq.<customer_id>&created_at=gte.<start_of_month>
        start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        params = {"select":"id", "customer_id": f"eq.{customer_id}", "created_at": f"gte.{start}"}
        # Supabase REST expects query params in URL; build manually
        query = f"?select=id&customer_id=eq.{customer_id}&created_at=gte.{start}"
        full = url + query
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        r = requests.get(full, headers=headers, timeout=10)
        if r.status_code == 200:
            arr = r.json()
            return len(arr)
        return 0
    except Exception:
        return 0

def save_lead_supabase(lead_obj, customer_id=None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    url = SUPABASE_URL.rstrip("/") + f"/rest/v1/{SUPABASE_TABLE_LEADS}"
    headers = {
        "Content-Type":"application/json",
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    payload = {
        "phone": lead_obj.get("phone"),
        "email": lead_obj.get("email"),
        "intent": lead_obj.get("intent", False),
        "raw_text": lead_obj.get("raw_text"),
        "source": lead_obj.get("source","unknown"),
        "customer_id": customer_id
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        return r.status_code in (200,201)
    except Exception:
        return False

# -----------------------------------------
# Retrieval + prompt build (same as Day 4, but with system prompt per-client)
# -----------------------------------------
def retrieve_context(client_cfg, user_text, top_k=TOP_K):
    collection = f"{client_cfg.get('client_id')}_kb"
    vec = embedder.encode([user_text])[0].tolist()
    try:
        if hasattr(qdrant, "search"):
            hits = qdrant.search(collection_name=collection, query_vector=vec, limit=top_k, with_payload=True)
        elif hasattr(qdrant, "search_points"):
            hits = qdrant.search_points(collection_name=collection, query_vector=vec, limit=top_k, with_payload=True)
        else:
            url = QDRANT_URL.rstrip("/") + f"/collections/{collection}/points/search"
            headers = {"Content-Type":"application/json", "api-key": QDRANT_API_KEY}
            payload = {"vector":vec,"limit":top_k,"with_payload":True}
            res = requests.post(url, json=payload, headers=headers, timeout=10); res.raise_for_status()
            hits = res.json().get("result") or []
    except Exception as e:
        print("qdrant search error:", e); return []
    contexts=[]
    for h in hits:
        if isinstance(h, dict):
            payload = h.get("payload",{})
            contexts.append(payload.get("text") or json.dumps(payload))
        else:
            payload = getattr(h,"payload",{}) or {}
            contexts.append(payload.get("text") or json.dumps(payload))
    return contexts

def build_messages(client_cfg, user_text, contexts, detected_lang):
    system = build_system_prompt(client_cfg)
    # If audience language is hinglish/hindi, inject instruction to reply in same language
    if detected_lang in ("hi","hinglish"):
        system += "\n\nReply in the user's language (Hindi or Hinglish) and prefer simple Hindi words or Hinglish, keep tone same as persona."
    messages = [
        {"role":"system","content":system},
        {"role":"user","content": user_text}
    ]
    return messages

# Main pipeline
def handle_message(user_text, user_meta=None, customer_id=None):
    client_cfg = load_client_config(customer_id)
    # language detect + normalize
    detected = detect_language(user_text)
    if detected == "hinglish":
        user_text_norm = normalize_romanized(user_text)
    else:
        user_text_norm = user_text

    # retrieve context
    contexts = retrieve_context(client_cfg, user_text_norm)
    messages = build_messages(client_cfg, user_text_norm, contexts, detected)
    # SAFETY: if no context found for this business, return a short fallback
    if not contexts:
        # If the KB is completely empty OR search returned no hits, don't hallucinate.
        reply = (f"Sorry — I don't have information about that for {client_cfg.get('display_name','this business')}. "
                 "Would you like me to connect you to a specialist or look up a nearby business?")
        return {"reply": reply, "contexts": contexts, "lead": None, "timing": {"total_s": 0}}

    # call llm
    try:
        reply = llm.ask(messages)
    except Exception as e:
        reply = f"ERROR: LLM failed: {e}"

    # lead detection + quota
    lead = detect_lead(user_text_norm, lead_def=client_cfg.get("lead_rules",{}).get("lead_definition","phone_or_email_or_intent"))
    lead_payload=None
    if lead:
        # check current monthly count
        current = get_monthly_lead_count(customer_id) if customer_id else 0
        free = client_cfg.get("lead_rules",{}).get("free_leads_per_month",250)
        overage=False
        if current >= free:
            overage=True
        lead_payload = {"phone":lead.get("phone"),"email":lead.get("email"),"intent":lead.get("intent"),"raw_text":user_text_norm,"source": user_meta.get("channel") if user_meta else "unknown","overage":overage}
        # save only if under free or you still want to save and mark overage
        saved = save_lead_supabase(lead_payload, customer_id=customer_id)
        lead_payload["saved"]=saved
        lead_payload["current_count"]=current
        lead_payload["free_limit"]=free

    timing = {"total_s":0}
    return {"reply": reply, "contexts": contexts, "lead": lead_payload, "timing": timing }
