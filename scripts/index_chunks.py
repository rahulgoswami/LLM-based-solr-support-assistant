#!/usr/bin/env python3
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ——— CONFIGURATION ———
DATA_DIR        = "../data/chunks"        # directory with chunk JSONs
PERSIST_DIR     = "../vector_store"       # ChromaDB persistence folder
COLLECTION_NAME = "solr_support"
BATCH_SIZE      = 64                   # tune between 32–64 on CPU :contentReference[oaicite:4]{index=4}

def main():
    # 1. Initialize PersistentClient (duckdb+parquet under the hood) :contentReference[oaicite:5]{index=5}
    client = PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE
    )

    # 2. Use MPNet for embeddings (supports 384–512 tokens) :contentReference[oaicite:6]{index=6}
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Load chunk files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    ids, texts, metadatas = [], [], []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        chunk = json.load(open(path, "r"))
        ids.append(chunk["chunk_id"])
        texts.append(chunk["text"])
        metadatas.append({
            "issue_number": chunk["issue_number"],
            "source":       chunk["source"]
        })

    # 4. Batch embedding & upsert with progress bar :contentReference[oaicite:7]{index=7}
    model = SentenceTransformer("all-mpnet-base-v2")
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in tqdm(range(0, len(texts), BATCH_SIZE),
                            total=total_batches,
                            desc="Indexing batches",
                            unit="batch"):
        batch_ids       = ids[batch_start:batch_start + BATCH_SIZE]
        batch_texts     = texts[batch_start:batch_start + BATCH_SIZE]
        batch_metadatas = metadatas[batch_start:batch_start + BATCH_SIZE]

        # Generate embeddings in batches on CPU :contentReference[oaicite:8]{index=8}
        embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            device="cpu"
        )

        # Upsert batch into ChromaDB :contentReference[oaicite:9]{index=9}
        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadatas
        )

    print(f"✅ Indexed {len(ids)} chunks in {total_batches} batches.")

if __name__ == "__main__":
    main()

