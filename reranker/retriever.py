from __future__ import annotations
from typing import List
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from .loader import Row


class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.emb: np.ndarray | None = None

    def fit(self, rows: List[Row], show_progress: bool = False) -> None:
        t0 = time.time()
        texts = [r.for_embedding() for r in rows]
        mat = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        self.emb = mat.astype(np.float32)

    def query(self, query_text: str, k: int) -> List[int]:
        q = self.model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
        sims = self.emb @ q
        k = min(k, len(sims))
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx.tolist()
