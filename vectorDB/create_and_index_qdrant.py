import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

# Load Qdrant credentials
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "business_kb"

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in .env")

# Load simple embedding model (CPU friendly)
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

# Load text file
with open("C:\\AI-Agent\\data\\restaurant_faqs.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split into separate chunks (simple split for now)
documents = [d.strip() for d in content.split("\n\n") if d.strip()]

# Embed each chunk
vectors = model.encode(documents).tolist()

# Connect to Qdrant Cloud
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create or recreate collection
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=rest.VectorParams(
        size=len(vectors[0]),
        distance=rest.Distance.COSINE
    ),
)

# Insert data
points = []
for i, (text, vec) in enumerate(zip(documents, vectors)):
    points.append(
        rest.PointStruct(
            id=i,
            vector=vec,
            payload={"text": text}
        )
    )

client.upsert(collection_name=COLLECTION, points=points)

print(f"Indexed {len(points)} documents into '{COLLECTION}'.")
