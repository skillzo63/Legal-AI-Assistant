import json
import numpy as np
from turbovec import TurboQuantIndex
from indexer import get_embedding
import functools

@functools.lru_cache(maxsize=256)
def get_embedding_cached(text: str):
    return tuple(get_embedding(text))

class LegalRetriever:
    def __init__(self, index_path="aus_legal_qa.tv", meta_path="metadata.json"):
        self.index = TurboQuantIndex.load(path=index_path)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

    def search(self, query_text, k=3, threshold=0.65):
        query_vec = get_embedding_cached(query_text)
        queries = np.array([query_vec], dtype=np.float32)
        
        scores, indices = self.index.search(queries, k=k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if round(float(score), 4) >= threshold:
                results.append({
                    "id": self.metadata[idx]["id"],
                    "score": round(float(score), 4),
                    "question": self.metadata[idx]["question"],
                    "answer": self.metadata[idx]["answer"],
                    "source": self.metadata[idx]["source_name"],
                    "url": self.metadata[idx]["source_url"]
                })
        return results if results else None