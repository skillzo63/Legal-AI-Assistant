import os
import json
import hashlib
import numpy as np
from datasets import load_dataset
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors
from turbovec import TurboQuantIndex
from dotenv import load_dotenv
# Load API keys securely
load_dotenv()
embed_client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-2"

MAX_RECORDS = 500  # how many records to embed

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_embedding(text):
    result = embed_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values

def build_index():
    print(f"Loading open-australian-legal-qa (first {MAX_RECORDS} records)")
    dataset = load_dataset("isaacus/open-australian-legal-qa", split="train")

    metadata = []
    vectors = []

    for i, record in enumerate(dataset):
        if i >= MAX_RECORDS:
            break
        text = (
            f"question: {record.get('question', 'Unknown')}\n"
            f"answer: {record.get('answer', 'None')}\n"
            f"source_name: {record.get('source', {})['citation']}\n"
            f"source_url: {record.get('source', {})['url']}"
        )

        vectors.append(get_embedding(record.get('question')))
        metadata.append({
            "id": get_hash(record.get('question')),
            "question": record.get('question'),
            "answer": record.get('answer'),
            "source_name": record.get('source', {})['citation'],
            "source_url": record.get('source', {})['url']
        })

        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{MAX_RECORDS} records...")

    # Build and save TurboVec index
    index = TurboQuantIndex(dim=3072, bit_width=4)
    index.add(np.array(vectors, dtype=np.float32))
    index.write("aus_legal_qa.tv")

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Index saved: aus_legal_qa.tv")

if __name__ == "__main__":
    build_index()

