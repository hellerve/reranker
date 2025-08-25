from __future__ import annotations
import argparse, json, sys, time
from typing import List
from tabulate import tabulate
from .loader import load_rows, Row
from .retriever import Retriever
from .reranker import Reranker
from .search import search_with_rerank


def _print_results(rows):
    if not rows:
        print("(no results)")
        return
    print(tabulate(rows))


def cmd_search(args: argparse.Namespace) -> int:
    rows = load_rows(args.root)
    if not rows:
        print(f"[error] No markdown files found under {args.root}", file=sys.stderr)
        return 2
    retr = Retriever(args.retriever_model)
    retr.fit(rows, show_progress=False)

    rr = Reranker(
        model_name=args.reranker_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    out = search_with_rerank(
        rows=rows,
        retriever=retr,
        reranker=rr,
        query_text=args.query,
        columns_to_rerank=args.columns_to_rerank,
        num_candidates=args.num_candidates,
        num_results=args.num_results,
    )
    _print_results(out["results"])
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    rows = load_rows(args.root)
    if not rows:
        print(f"[error] No markdown files found under {args.root}", file=sys.stderr)
        return 2

    retr = Retriever(args.retriever_model)
    retr.fit(rows)
    rr = Reranker(
        model_name=args.reranker_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    qrels = json.load(open(args.qrels, "r", encoding="utf-8"))
    ids = [r.id for r in rows]

    def rank_ids(q: str, k_cand: int) -> List[Row]:
        cand = retr.query(q, k_cand)
        ranked = rr.rerank(q, [rows[i] for i in cand], args.columns_to_rerank)
        return ranked

    def recall_at_k(ranked: List[Row], relevant_ids: List[str], k: int) -> float:
        print(relevant_ids)
        print(ranked_ids)
        return 1.0 if set(r.id for r in ranked_ids[:k]) & set(relevant_ids) else 0.0

    table = []
    t0 = time.time()
    for q, rel in qrels.items():
        ranked_ids = rank_ids(q, args.num_candidates)
        table.append(
            dict(
                query=q,
                R_at_5=recall_at_k(ranked_ids, rel, 5),
                R_at_10=recall_at_k(ranked_ids, rel, 10),
            )
        )
    print(tabulate(table, headers="keys"))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    rows = load_rows(args.root)
    if not rows:
        print(f"[error] No markdown files found under {args.root}", file=sys.stderr)
        return 2
    retr = Retriever(args.retriever_model)
    retr.fit(rows)
    rr = Reranker(
        model_name=args.reranker_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    data = []
    for k in args.candidates:
        t0 = time.time()
        cand = retr.query(args.query, k)
        t1 = time.time()
        _ = rr.rerank(args.query, [rows[i] for i in cand], args.columns_to_rerank)
        t2 = time.time()
        data.append(
            {
                "num_candidates": k,
                "retrieval_time_sec": round(t1 - t0, 4),
                "rerank_time_sec": round(t2 - t1, 4),
                "total_time_sec": round(t2 - t0, 4),
            }
        )
    print(tabulate(data, headers="keys"))
    return 0


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reranker", description="Tiny reranker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root", required=True, help="Root directory containing .md files"
    )
    common.add_argument(
        "--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    common.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    common.add_argument(
        "--columns-to-rerank", nargs="+", default=["title", "summary", "body"]
    )
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--max-length", type=int, default=512)

    p_search = sub.add_parser("search", parents=[common], help="Run retrieval + rerank")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--num-candidates", type=int, default=50)
    p_search.add_argument("--num-results", type=int, default=10)
    p_search.add_argument("--columns", nargs="+", default=["id", "title", "url"])
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser(
        "eval", parents=[common], help="Evaluate with Recall@K using qrels.json"
    )
    p_eval.add_argument("--qrels", required=True, help="Path to qrels.json")
    p_eval.add_argument("--num-candidates", type=int, default=50)
    p_eval.set_defaults(func=cmd_eval)

    p_sweep = sub.add_parser(
        "sweep", parents=[common], help="Latency sweep for different candidate counts"
    )
    p_sweep.add_argument("--query", required=True)
    p_sweep.add_argument("--candidates", nargs="+", type=int, default=[25, 50, 100])
    p_sweep.set_defaults(func=cmd_sweep)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
