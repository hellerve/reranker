from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import os, re


h1 = re.compile(r"^#\s+(.*)")


@dataclass
class Row:
    id: str
    title: str
    summary: str
    body: str
    url: str = ""

    def for_embedding(self, max_body_chars: int = 2000) -> str:
        return f"title: {self.title}\nsummary: {self.summary}\nbody: {self.body[:max_body_chars]}"

    def for_rerank(self, columns_to_rerank: Sequence[str]) -> str:
        parts = []
        for c in columns_to_rerank:
            v = getattr(self, c, "")
            if not v:
                continue
            parts.append(f"{c}: {v}")
        return "\n".join(parts)


def read_md(path: str) -> Tuple[str, str, str]:
    with open(path, "r", encoding="utf-8") as f:
        t = f.read()

    title = next(
        (m.group(1).strip() for m in map(h1.match, t.splitlines()) if m),
        os.path.splitext(os.path.basename(path))[0],
    )

    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    summary = next((p for p in paras if not p.lstrip().startswith("#")), t[:200])
    return title, summary, t


def load_rows(root: str) -> List[Row]:
    rows: List[Row] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            title, summary, body = read_md(p)
            rid = os.path.relpath(p, root)
            rows.append(Row(id=rid, title=title, summary=summary, body=body, url=p))
    return rows
