# rag_handler_twilio.py  (final - lazy RAG import, safe for Day 7+)
import os
import sys
import json
import logging
from flask import Flask, request, Response
from dotenv import load_dotenv
import importlib

# Ensure project root is available for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from project-root .env (if present)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_handler_twilio")

# Import router/adapters statically (they are lightweight)
try:
    from serverless.router import adapter_twilio_form, identify_customer, adapter_meta_webhook
except Exception as e:
    logger.exception("Failed to import router/adapters: %s", e)
    # router is essential; we re-raise to make the error obvious during startup
    raise

app = Flask(__name__)

def _build_twiml_message(text: str) -> str:
    """Return TwiML XML for Twilio Messaging reply. Keep it simple (plain text)."""
    safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{safe}</Message>
</Response>"""

def _lazy_get_handle_message():
    """
    Lazily import runtime.rag_service and return its handle_message function.
    This avoids circular imports and prevents import-time failures from breaking the webhook.
    """
    try:
        # Use importlib to ensure proper import semantics
        rag_module = importlib.import_module("runtime.rag_service")
        # ensure attribute exists
        if not hasattr(rag_module, "handle_message"):
            raise ImportError("runtime.rag_service has no attribute 'handle_message'")
        return rag_module.handle_message
    except Exception as e:
        # Log full traceback for debugging
        logger.exception("Lazy import of runtime.rag_service failed: %s", e)
        raise

@app.route("/twilio_webhook", methods=["POST", "GET"])
def twilio_webhook():
    """
    Main webhook handler for Twilio (WhatsApp). Accepts form-encoded POSTs.
    We:
      - normalize payload via adapter_twilio_form
      - identify customer via router.identify_customer
      - lazy-import rag_service.handle_message and call it
      - return TwiML XML with the reply
    """
    if request.method == "GET":
        return Response("Expecting POST", status=405)

    # 1) Parse incoming request safely (form-encoded or JSON)
    try:
        if request.content_type and "application/json" in request.content_type:
            incoming = request.get_json(silent=True) or {}
        else:
            # handle form-encoded Twilio POSTs
            incoming = {k: request.form.get(k) for k in request.form.keys()}
        logger.info("Incoming request parsed: keys=%s", list(incoming.keys()))
    except Exception as e:
        logger.exception("Failed to parse incoming request: %s", e)
        twiml = _build_twiml_message("Internal error: could not parse request.")
        return Response(twiml, status=200, mimetype="application/xml")

    # 2) Normalize to canonical message
    try:
        # Prefer Twilio adapter for form data, otherwise best-effort via meta adapter or raw text
        if isinstance(incoming, dict) and ("From" in incoming or "To" in incoming or "Body" in incoming):
            canonical = adapter_twilio_form(incoming)
        else:
            # try meta/graph-adapter (best-effort); fallback to raw
            canonical = adapter_meta_webhook(incoming) if isinstance(incoming, dict) else {"channel": "unknown", "from": None, "to": None, "text": str(incoming), "raw": incoming}
    except Exception as e:
        logger.exception("Adapter normalization failed: %s", e)
        canonical = {"channel": "unknown", "from": incoming.get("From") if isinstance(incoming, dict) else None, "to": incoming.get("To") if isinstance(incoming, dict) else None, "text": incoming.get("Body") if isinstance(incoming, dict) else str(incoming), "raw": incoming}

    logger.info("Canonical message: channel=%s from=%s to=%s text=%s", canonical.get("channel"), canonical.get("from"), canonical.get("to"), (canonical.get("text") or "")[:80])

    # 3) Determine customer via router (safe - router was imported at module load)
    try:
        customer_id = identify_customer(canonical)
    except Exception as e:
        logger.exception("identify_customer failed: %s", e)
        customer_id = None

    logger.info("Routed to customer_id=%s", customer_id)

    # 4) Lazy import and call RAG service
    try:
        handle_message = _lazy_get_handle_message()
    except Exception:
        # If import fails, return an explicit TwiML that helps debugging
        twiml = _build_twiml_message("RAG service not configured or failed to import. Check server logs.")
        return Response(twiml, status=200, mimetype="application/xml")

    # Call the RAG pipeline
    try:
        user_text = canonical.get("text") or ""
        user_meta = {"channel": canonical.get("channel"), "from": canonical.get("from"), "to": canonical.get("to")}
        result = handle_message(user_text, user_meta=user_meta, customer_id=customer_id)
        # Expect result to be dict with 'reply'
        reply_text = (result.get("reply") if isinstance(result, dict) else None) or "Sorry, something went wrong."
    except Exception as e:
        logger.exception("handle_message raised an exception: %s", e)
        reply_text = "Sorry, an internal error occurred while processing your message."

    # 5) Return TwiML response
    twiml = _build_twiml_message(reply_text)
    logger.info("Replying (len=%d) to %s", len(reply_text), canonical.get("from"))
    return Response(twiml, status=200, mimetype="application/xml")


if __name__ == "__main__":
    # Dev server entrypoint; use PORT env if set
    port = int(os.getenv("PORT", 5000))
    logger.info("Starting rag_handler_twilio on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=True)
