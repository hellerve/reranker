from __future__ import annotations
from typing import List, Sequence
import numpy as np
from sentence_transformers import CrossEncoder
from .loader import Row


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        max_length: int | None = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self, query_text: str, rows: List[Row], columns_to_rerank: Sequence[str]
    ) -> List[Row]:
        pairs = [[query_text, r.for_rerank(columns_to_rerank)] for r in rows]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        order = np.argsort(-np.asarray(scores))
        return [rows[i] for i in order]
