"""
Populate the Supabase vector store with GAIA benchmark embeddings.

Prerequisites:
  - SUPABASE_URL and SUPABASE_SERVICE_KEY in .env
  - pgvector extension enabled in Supabase
  - gaia_documents table created with columns: content (text), metadata (jsonb), embedding (vector)
"""
import os
import json
from dotenv import load_dotenv
from supabase.client import Client, create_client
from sentence_transformers import SentenceTransformer
from utils import load_config

load_dotenv()

config = load_config()
data_path = config["data"]

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)
embeddings = SentenceTransformer(
    model_name_or_path=config["vector_store"]["embedding_model_name"],
    cache_folder=config["models"]["cache_folder"],
)

with open(data_path, "r") as jsonl_file:
    json_list = list(jsonl_file)

documents = []
for json_str in json_list:
    json_data = json.loads(json_str)
    content = json_data["Question"]
    embedding = embeddings.encode(content, normalize_embeddings=True).tolist()
    documents.append({
        "content": content,
        "metadata": {
            "source": "vector_search",
            "task_id": json_data["task_id"],
        },
        "embedding": embedding,
    })

print(f"Inserting {len(documents)} documents into Supabase...")
try:
    response = supabase.table("gaia_documents").insert(documents).execute()
    print("Done.")
except Exception as e:
    print("Error inserting data into Supabase:", e)
