# rag_handler_twilio.py
# A lightweight Flask webhook for Twilio WhatsApp sandbox.
# Put this in C:\AI-Agent\serverless\ and run from project root:
#   python -m serverless.rag_handler_twilio
#
# It either imports the handle_message function from runtime.rag_service
# (preferred, local) or calls a RAG HTTP endpoint (RAG_SERVICE_LOCAL_URL).

import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import requests
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# Try to import local handle_message (preferred)
try:
    from runtime.rag_service import handle_message
    LOCAL_HANDLE = True
except Exception:
    LOCAL_HANDLE = False

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # like "whatsapp:+1415..."
RAG_SERVICE_LOCAL_URL = os.getenv("RAG_SERVICE_LOCAL_URL")  # optional HTTP fallback

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = Flask(__name__)

@app.route("/twilio_webhook", methods=["POST"])
def twilio_webhook():
    # Twilio posts form-encoded data
    sender = request.form.get("From")       # e.g. "whatsapp:+91xxxxxxxxxx"
    body = request.form.get("Body", "")
    # You can also handle Media etc here

    print("Incoming from", sender, ":", body)

    # call local handler if available
    if LOCAL_HANDLE:
        res = handle_message(body, user_meta={"channel":"whatsapp", "from":sender}, customer_id=None)
    else:
        # fallback: call HTTP RAG service
        if not RAG_SERVICE_LOCAL_URL:
            return ("RAG service not configured", 500)
        payload = {"text": body}
        r = requests.post(RAG_SERVICE_LOCAL_URL, json=payload, timeout=20)
        if r.status_code != 200:
            return ("RAG service error", 500)
        res = r.json()

    # Build text reply
    if isinstance(res, dict) and res.get("reply"):
        reply_text = res["reply"]
    else:
        reply_text = str(res)

    # Write back via Twilio API (server-to-server) OR use TwiML response
    # Using TwiML is simpler (respond to Twilio request with reply)
    twiml = MessagingResponse()
    twiml.message(reply_text)
    return str(twiml), 200, {"Content-Type": "text/xml"}

if __name__ == "__main__":
    # Run from project root: python -m serverless.rag_handler_twilio
    app.run(host="0.0.0.0", port=5000, debug=True)
