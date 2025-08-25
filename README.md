# reranker

A tiny reranker built on cross-encoding. It:

- loads `.md` posts (title, summary, body)
- retrieves top-N candidates with a **bi-encoder** (fast)
- reorders them with a **cross-encoder** (precise)
- prints results + debug timings
- includes a micro-eval (Recall@K) and a latency sweep

Read more about it [on my blog](https://blog.veitheller.de/Building_a_simple_reranker.html).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

1. Put some posts in `content/`.

2. Run a search:

```bash
python -m reranker.cli search \
  --root content \
  --query "i’m searching for my missing piece" \
  --num-candidates 48 \
  --num-results 8 \
  --columns-to-rerank title summary body \
  --columns id title url
```

## Usage

```
usage: reranker [-h] {search,eval,sweep} ...

Tiny reranker

positional arguments:
  {search,eval,sweep}
    search             Run retrieval + rerank
    eval               Evaluate with Recall@K using qrels.json
    sweep              Latency sweep for different candidate counts

options:
  -h, --help           show this help message and exit
```

<hr/>

Have fun!
