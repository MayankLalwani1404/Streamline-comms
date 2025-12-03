# query_qdrant_demo.py (fixed: request payloads & allow single-best result)
# pip install sentence-transformers qdrant-client python-dotenv requests

import os
import time
import json
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "business_kb"
# change to 1 if you want only the top hit
TOP_K = 3

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in .env")

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# qdrant client
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def qdrant_search_with_client(query_vector, limit=3):
    """Try qdrant-client search variants, explicitly request payload."""
    # Newer qdrant-client: client.search(collection_name=..., query_vector=..., with_payload=True)
    try:
        if hasattr(client, "search"):
            hits = client.search(
                collection_name=COLLECTION,
                query_vector=query_vector,
                limit=limit,
                with_payload=True  # explicitly request payload
            )
            return hits
    except TypeError:
        # some versions expect different args; fallthrough to next
        pass
    except Exception:
        pass

    # Some versions: search_points
    try:
        if hasattr(client, "search_points"):
            hits = client.search_points(
                collection_name=COLLECTION,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            return hits
    except Exception:
        pass

    # REST fallback: include with_payload True in JSON body
    try:
        url = QDRANT_URL.rstrip("/") + f"/collections/{COLLECTION}/points/search"
        headers = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}
        payload = {"vector": query_vector, "limit": limit, "with_payload": True}
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        results = body.get("result") or body.get("hits") or []
        class Hit:
            def __init__(self, score, payload):
                self.score = score
                self.payload = payload
        hits = []
        for r in results:
            score = r.get("score") or r.get("_score") or None
            payload = r.get("payload") or {}
            hits.append(Hit(score=score, payload=payload))
        return hits
    except Exception as e:
        raise RuntimeError(f"All Qdrant search methods failed: {e}")

def interactive_loop():
    print("Qdrant interactive demo. Type 'exit' to quit.")
    while True:
        query = input("\nAsk something (type 'exit' to quit): ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        try:
            t0 = time.time()
            qvec = model.encode([query])[0].tolist()
            t1 = time.time()
        except Exception as e:
            print("Embedding error:", e)
            continue

        try:
            hits = qdrant_search_with_client(qvec, limit=TOP_K)
        except Exception as e:
            print("Search failed:", e)
            continue

        print(f"\nQuery embed time: {t1-t0:.3f}s, search returned {len(hits)} hits (top_k={TOP_K}):")
        if not hits:
            print("No hits returned.")
            continue

        for h in hits:
            # handle both dict-like and object-like hits
            if isinstance(h, dict):
                score = h.get("score")
                payload = h.get("payload", {})
            else:
                score = getattr(h, "score", None)
                payload = getattr(h, "payload", {}) or {}

            # payload should contain the original text under the 'text' key
            text = payload.get("text") or payload.get("content") or payload.get("payload_text") or json.dumps(payload)
            print(f"Score: {score}")
            print("Text:", text)
            print("------")

if __name__ == "__main__":
    interactive_loop()
