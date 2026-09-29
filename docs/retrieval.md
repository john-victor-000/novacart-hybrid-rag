# Retrieval design

## Dense retrieval

`sentence-transformers/all-MiniLM-L6-v2` creates normalized local embeddings.
Chroma persists chunk embeddings and cosine distances. Query distances are
converted to similarity values while source metadata remains attached.

Dense retrieval works well for paraphrases and semantic policy questions. It
can be weaker for exact SKUs and deterministic numeric filters.

## BM25

The sparse index tokenizes text without destroying identifiers such as
`NCM-24`. BM25 favors exact lexical overlap and is especially useful for SKUs,
policy wording, and numbers. The index is persisted separately from Chroma.

## Reciprocal Rank Fusion

Dense and BM25 scores use different scales and are never added directly.
Reciprocal Rank Fusion combines positions:

```text
RRF(chunk) = sum(1 / (rrf_k + rank_in_retriever))
```

Chunks merge by stable `chunk_id`. Diagnostics preserve dense rank/score, BM25
rank/score, and final RRF score.

## Reranking

When enabled, Hybrid RRF retrieves a wider candidate set and a configurable BGE
cross-encoder scores query/chunk pairs. Only the strongest final candidates are
sent to context construction.

The current ten-question benchmark shows that
`BAAI/bge-reranker-base` is slow on CPU and lowers Hit@5 for one filter query.
It therefore remains configurable through `RERANK_ENABLED` rather than being
treated as an unconditional improvement.

## Structured retrieval

Product queries use a pandas-backed repository over `products.csv`. Supported
operations include SKU and name lookup, price/warranty/stock/category filters,
sorting, and comparison. The repository boundary allows a later SQL migration
without changing query routing or RAG orchestration.

## Routing

The deterministic router favors:

- `STRUCTURED` for exact SKU attributes and product filters,
- `MULTI_SOURCE` for multi-SKU comparisons or SKU-plus-policy questions,
- `HYBRID` for policies, shipping, returns, and general NovaCart knowledge,
- `DENSE` only when forced for baseline evaluation.

`RETRIEVAL_MODE=auto` enables normal routing. `dense` and `hybrid` force a mode
for controlled comparisons.

## Context and citations

The context builder removes exact duplicate evidence, respects item and
character limits, numbers evidence consistently, and retains retrieval
diagnostics. CitationBuilder maps the LLM's numbered labels back to the evidence
objects. Invalid labels cannot create citations, and duplicate citations are
removed.

## Index maintenance

Rebuild both indexes after changing source documents, parsing, chunk size,
overlap, embedding model, or collection configuration:

```powershell
.\.venv\Scripts\python.exe -m scripts.index_vectors --input-dir data/raw
.\.venv\Scripts\python.exe -m scripts.index_bm25 --input-dir data/raw
```
