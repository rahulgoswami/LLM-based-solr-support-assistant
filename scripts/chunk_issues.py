#!/usr/bin/env python3
import os
import json
import argparse
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure NLTK sentence tokenizer data is available
nltk.download('punkt', quiet=True)

def tokenize(text):
    """Split text into word tokens."""
    return word_tokenize(text)

def detokenize(tokens):
    """Join tokens back into a string."""
    return " ".join(tokens)

def chunk_tokens(tokens, chunk_size, overlap):
    """
    Split a list of tokens into overlapping chunks.
    chunk_size: max tokens per chunk
    overlap: tokens to overlap between chunks
    """
    stride = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks

def chunk_text(text, chunk_size, overlap):
    """
    Tokenize text by sentences, then into tokens, 
    and finally chunk into overlapping windows.
    """
    sentences = sent_tokenize(text)
    all_tokens = []
    for sent in sentences:
        all_tokens.extend(tokenize(sent))
    return chunk_tokens(all_tokens, chunk_size, overlap)

def process_issue_file(filepath, output_dir, chunk_size, overlap):
    """
    Read one issue JSON, chunk its title+body and each comment,
    and write chunk files to output_dir.
    """
    with open(filepath, 'r') as f:
        issue = json.load(f)

    issue_num = issue.get("number", "unknown")
    base_id   = f"issue_{issue_num}"

    # Prepare combined title+body
    text_body = f"{issue.get('title','')} {issue.get('body','')}".strip()
    body_chunks = chunk_text(text_body, chunk_size, overlap)

    # Write body chunks
    for idx, tokens in enumerate(body_chunks):
        chunk_id = f"{base_id}_body_{idx}"
        out = {
            "issue_number": issue_num,
            "source": "body",
            "chunk_id": chunk_id,
            "text": detokenize(tokens)
        }
        out_path = os.path.join(output_dir, f"{chunk_id}.json")
        with open(out_path, 'w') as cf:
            json.dump(out, cf, ensure_ascii=False, indent=2)

    # Process comments
    for comment in issue.get("comments", []):
        comment_id = comment.get("id", "x")
        ctext      = comment.get("body", "")
        c_chunks   = chunk_text(ctext, chunk_size, overlap)
        for idx, tokens in enumerate(c_chunks):
            chunk_id = f"{base_id}_comment_{comment_id}_{idx}"
            out = {
                "issue_number": issue_num,
                "source": "comment",
                "comment_id": comment_id,
                "chunk_id": chunk_id,
                "text": detokenize(tokens)
            }
            out_path = os.path.join(output_dir, f"{chunk_id}.json")
            with open(out_path, 'w') as cf:
                json.dump(out, cf, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Chunk GitHub-issue JSON files for embedding"
    )
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory containing issue JSON files"
    )
    parser.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to write chunk JSON files"
    )
    parser.add_argument(
        "--chunk-size", "-c", type=int, default=300,
        help="Maximum number of tokens per chunk (default: 300)"
    )
    parser.add_argument(
        "--overlap", "-l", type=int, default=60,
        help="Number of tokens to overlap between chunks (default: 60)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each JSON file in the input directory
    for fname in os.listdir(args.input_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(args.input_dir, fname)
        process_issue_file(fpath, args.output_dir, args.chunk_size, args.overlap)

    print(f"Chunking complete. Chunks written to: {args.output_dir}")

"""
python chunk_issues.py \
  --input-dir data/github_issues \
  --output-dir data/chunks \
  --chunk-size 300 \
  --overlap 60
"""
if __name__ == "__main__":
    main()

