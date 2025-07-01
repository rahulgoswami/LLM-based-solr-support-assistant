#!/usr/bin/env python3
import os
import sys
import json
from typing import List, Dict

import openai
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from dotenv import load_dotenv
load_dotenv()  # <— load env vars from .env 

# ——— CONFIGURATION ———
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME  = "solr_support"
EMBED_MODEL      = "all-mpnet-base-v2"
OPENAI_MODEL     = "gpt-4"       # or "gpt-3.5-turbo"
TOP_K            = 5             # number of passages to retrieve
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")


class RAGRetriever:
    def __init__(self,
                 persist_dir: str,
                 collection_name: str,
                 embed_model: str):
        # Initialize persistent Chroma client
        self.client = PersistentClient(
            path=persist_dir,
            settings=Settings(),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE
        )
        # Configure embedding function
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        # Load the collection
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        """
        Embed the query and retrieve top_k passages.
        Returns a list of dicts: {'text': ..., 'metadata': {...}}
        """
        # Embed the query
        query_emb = self.collection._embedding_function([query])[0]
        # Perform similarity search
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"text": doc, "metadata": meta}
                for doc, meta in zip(docs, metas)]


class RAGGenerator:
    def __init__(self, llm_model: str, api_key: str):
        openai.api_key = api_key
        self.model = llm_model

    def generate(self, query: str, contexts: List[Dict]) -> str:
        """
        Build a grounded prompt and call the LLM.
        Returns the model's answer text.
        """
        # Build the context block
        context_blocks = []
        for i, ctx in enumerate(contexts, start=1):
            src = ctx["metadata"].get("source", "unknown")
            src_id = ctx["metadata"].get("issue_number", "")
            context_blocks.append(
                f"[{i}] ({src}:{src_id}) {ctx['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        # Prompt template
        prompt = (
            "You are a technical support assistant for Apache Solr and Lucene.\n"
            "Answer the user's question based **only** on the following passages:\n\n"
            f"{context_str}\n\n"
            f"Question: {query}\n"
            "Answer (with citations in [n] format):"
        )

        # Call OpenAI
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()


class RAGPipeline:
    def __init__(self,
                 persist_dir: str,
                 collection_name: str,
                 embed_model: str,
                 llm_model: str,
                 api_key: str,
                 top_k: int = 5):
        self.retriever = RAGRetriever(persist_dir, collection_name, embed_model)
        self.generator = RAGGenerator(llm_model, api_key)
        self.top_k = top_k

    def answer(self, query: str) -> str:
        contexts = self.retriever.retrieve(query, self.top_k)
        return self.generator.generate(query, contexts)


def main():
    if OPENAI_API_KEY is None:
        print("Error: Set the environment variable OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) != 2:
        print("Usage: python rag_pipeline.py \"Your question here\"", file=sys.stderr)
        sys.exit(1)

    user_query = sys.argv[1].strip()
    pipeline = RAGPipeline(
        persist_dir=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME,
        embed_model=EMBED_MODEL,
        llm_model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        top_k=TOP_K
    )
    answer = pipeline.answer(user_query)
    print("\n=== RAG Answer ===")
    print(answer)


if __name__ == "__main__":
    main()

