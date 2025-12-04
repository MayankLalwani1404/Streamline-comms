# serverless/router.py
"""
Lightweight router + adapters for Day 7.

Provides:
- load_mappings()
- identify_customer(canonical_msg)
- adapters: adapter_twilio_form, adapter_meta_webhook, adapter_email_sendgrid
- simple __main__ test harness
"""

import os
import json
import re
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAPPINGS_PATH = os.path.join(PROJECT_ROOT, "data", "mappings.json")

# safe load mappings
def load_mappings(path: str = MAPPINGS_PATH):
    if not os.path.exists(path):
        return {"mappings": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"mappings": []}

MAPPINGS = load_mappings()

def norm_text(t: str) -> str:
    return (t or "").strip().lower()

def map_by_to(channel: str, to_identifier: str):
    if not to_identifier:
        return None
    to_identifier = to_identifier.strip()
    for m in MAPPINGS.get("mappings", []):
        for ch in m.get("channels", []):
            # exact match on 'type' and 'to'
            if ch.get("type") == channel and ch.get("to") == to_identifier:
                return m.get("customer_id")
    return None

def map_by_identifier(identifier: str):
    if not identifier:
        return None
    identifier = identifier.strip()
    for m in MAPPINGS.get("mappings", []):
        for ch in m.get("channels", []):
            if ch.get("to") == identifier or ch.get("page_id") == identifier or ch.get("email") == identifier:
                return m.get("customer_id")
    return None

# simple keyword map (fallback)
KEYWORD_MAP = [
    (["tattoo","ink","tattooing","tattoo studio"], "tattoo_surat"),
    (["pizza","menu","coffee","cafe","special","margherita","paneer"], "cafe_pune"),
    (["hair","salon","cut","styling","appointment","barber"], "salon_mumbai"),
]

def map_by_keyword(text: str):
    t = norm_text(text)
    for tokens, cid in KEYWORD_MAP:
        for tk in tokens:
            if tk in t:
                return cid
    return None

def identify_customer(canonical_msg: Dict[str, Any]) -> str:
    """
    Return customer_id string (or 'default' if none).
    canonical_msg should contain: channel, to, from, text
    """
    if not canonical_msg:
        return "default"
    channel = canonical_msg.get("channel")
    to = canonical_msg.get("to")
    frm = canonical_msg.get("from")
    text = canonical_msg.get("text") or ""

    # 1) exact mapping by to+channel
    cid = map_by_to(channel, to)
    if cid:
        return cid

    # 2) identifier mapping
    cid = map_by_identifier(to) or map_by_identifier(frm)
    if cid:
        return cid

    # 3) keyword fallback on text
    cid = map_by_keyword(text)
    if cid:
        return cid

    # 4) default
    return "default"

# ------------- adapters -------------
def adapter_twilio_form(body: Dict[str, str]) -> Dict[str, Any]:
    """
    Twilio form-encoded POST contains keys 'From', 'To', 'Body'
    """
    return {
        "channel": "whatsapp",
        "from": body.get("From"),
        "to": body.get("To"),
        "text": body.get("Body"),
        "raw": body
    }

def adapter_meta_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort normalize Meta Graph webhook (WhatsApp/Instagram/Facebook).
    This is permissive — you should adapt to the exact webhook shapes you see.
    """
    try:
        # WhatsApp via Meta Cloud common shape
        entry = payload.get("entry", [None])[0]
        if entry:
            changes = entry.get("changes", [None])[0]
            if changes:
                value = changes.get("value", {})
                # WhatsApp messages
                if "messages" in value:
                    msg = value["messages"][0]
                    from_id = msg.get("from")
                    phone_id = value.get("metadata", {}).get("phone_number_id") or value.get("metadata", {}).get("display_phone_number")
                    text = (msg.get("text") or {}).get("body")
                    return {
                        "channel": "whatsapp",
                        "from": f"whatsapp:{from_id}",
                        "to": f"whatsapp:{phone_id}" if phone_id else None,
                        "text": text,
                        "raw": payload
                    }
                # Instagram / Messenger simplified
                if "messages" in value or "direct_message" in value:
                    msgs = value.get("messages", [])
                    if msgs:
                        m = msgs[0]
                        return {
                            "channel": "instagram",
                            "from": f"instagram:{m.get('from') or m.get('sender_id')}",
                            "to": entry.get("id"),
                            "text": m.get("text") or m.get("message"),
                            "raw": payload
                        }
    except Exception:
        pass
    # fallback
    return {"channel": "unknown", "from": None, "to": None, "text": json.dumps(payload), "raw": payload}

def adapter_email_sendgrid(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "channel": "email",
        "from": parsed.get("from"),
        "to": parsed.get("to"),
        "text": (parsed.get("text") or "") + ("\nSubject: " + (parsed.get("subject") or "")),
        "raw": parsed
    }

# ------------- quick test harness -------------
if __name__ == "__main__":
    print("Router quick test - mappings loaded:", bool(MAPPINGS.get("mappings")))
    ex1 = {'channel':'whatsapp','to':'whatsapp:+14155238886','from':'whatsapp:+919876543210','text':'Hii, Aaj ka special kya hai?','raw':{}}
    ex2 = {'channel':'whatsapp','to':'whatsapp:+14155238886','from':'whatsapp:+919876543211','text':'Mujhe haircut chahiye','raw':{}}
    ex3 = {'channel':'email','to':'tattoo@inkvibe.com','from':'user@example.com','text':'I want a tattoo appointment','raw':{}}
    print("ex1 ->", identify_customer(ex1))
    print("ex2 ->", identify_customer(ex2))
    print("ex3 ->", identify_customer(ex3))
    print("Mappings content:\n", json.dumps(MAPPINGS, indent=2))
