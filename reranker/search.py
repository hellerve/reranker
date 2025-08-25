from __future__ import annotations
from typing import Any, Dict, List, Sequence
import time
from .loader import Row
from .retriever import Retriever
from .reranker import Reranker


def search_with_rerank(
    rows: List[Row],
    retriever: Retriever,
    reranker: Reranker,
    query_text: str,
    columns_to_rerank: Sequence[str],
    num_candidates: int = 50,
    num_results: int = 10,
) -> Dict[str, Any]:
    t0 = time.time()
    candidate_idx = retriever.query(query_text, num_candidates)
    t1 = time.time()

    reranked = reranker.rerank(
        query_text, [rows[i] for i in candidate_idx], columns_to_rerank
    )
    t2 = time.time()

    results = reranked[:num_results]

    debug = {
        "retrieval_time_sec": t1 - t0,
        "rerank_time_sec": t2 - t1,
        "total_time_sec": t2 - t0,
        "num_candidates": num_candidates,
        "num_results": num_results,
    }
    return {"results": results, "debug_info": debug}
