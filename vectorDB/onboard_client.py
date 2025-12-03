# onboard_client.py
import os, sys, json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLIENT_DIR = os.path.join(PROJECT_ROOT, "data", "clients")
load_path = lambda p: os.path.join(PROJECT_ROOT, p)

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

model = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def chunk_text(text, max_chars=800):
    parts=[]
    cur=[]
    cur_len=0
    for para in text.split("\n\n"):
        if not para.strip(): continue
        if len(para)+cur_len > max_chars:
            parts.append(" ".join(cur)); cur=[para]; cur_len=len(para)
        else:
            cur.append(para); cur_len += len(para)
    if cur: parts.append(" ".join(cur))
    return parts

def onboard(client_id):
    cfg_path = os.path.join(CLIENT_DIR, f"{client_id}.json")
    cfg = json.load(open(cfg_path,"r",encoding="utf-8"))
    collection = f"{client_id}_kb"
    # gather files
    docs=[]
    for f in cfg.get("kb_files",[]):
        p=load_path(f)
        with open(p,"r",encoding="utf-8") as fh:
            docs.extend(chunk_text(fh.read()))
    if not docs:
        print("No docs to index for", client_id); return
    vectors = model.encode(docs).tolist()
    # recreate collection
    client.recreate_collection(
        collection_name=collection,
        vectors_config=rest.VectorParams(size=len(vectors[0]), distance=rest.Distance.COSINE)
    )
    points=[]
    for i,(d,v) in enumerate(zip(docs,vectors)):
        points.append(rest.PointStruct(id=i, vector=v, payload={"text":d}))
        if len(points)>=256:
            client.upsert(collection_name=collection, points=points)
            points=[]
    if points:
        client.upsert(collection_name=collection, points=points)
    print("Indexed", len(docs), "documents into", collection)

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("Usage: python onboard_client.py <client_id>")
    else:
        onboard(sys.argv[1])
