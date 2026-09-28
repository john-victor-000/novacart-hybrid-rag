# NovaCart retrieval evaluation

## Dataset

The benchmark uses `data/evaluation/evaluation_questions.csv`. It contains 10
NovaCart questions with one or more expected source documents. A plus sign in
`expected_source` represents multiple relevant documents. The separate
`data/evaluation/golden_queries.csv` file contains six high-value router
regression queries.

The judgments are document-level because the supplied dataset identifies source
documents rather than exact chunk IDs. A relevant document receives credit only
on its first appearance in a ranking. Additional chunks from that document do
not inflate the score.

## Metrics

- **Hit Rate@K:** fraction of queries with at least one expected document in the
  first K results.
- **Recall@K:** unique expected documents found in the first K divided by the
  number expected.
- **Precision@K:** unique expected documents found divided by K.
- **MRR@K:** mean reciprocal rank of the first expected document.
- **NDCG@K:** binary document relevance with logarithmic rank discount.

Metrics are macro-averaged over the 10 questions at K=1, K=3, and K=5.
Duplicate chunks from an already matched document have zero additional gain.

## Retrieval modes

The evaluator constructs the existing components without changing their
algorithms:

1. Dense retrieval from the persisted Chroma collection.
2. BM25 retrieval from the persisted sparse index.
3. Dense plus BM25 with Reciprocal Rank Fusion.
4. The same Hybrid RRF candidates followed by `BAAI/bge-reranker-base`.

## Experiment configuration

The measured run completed on 2026-09-28 using CPU inference:

| Setting | Value |
| --- | --- |
| Questions | 10 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker model | `BAAI/bge-reranker-base` |
| Chroma collection | `novacart_chunks` |
| RRF constant | 60 |
| Dense/BM25 fusion depth | 10 / 10 |
| Reranker candidate setting | 20 |
| Evaluated K | 1, 3, 5 |

Each mode receives one warm-up query before measured queries. Model-loading time
is recorded separately in `retrieval_results.json` and excluded from mean query
latency. The first dense warm-up took 12,097.749 ms and the first
hybrid-plus-reranker warm-up took 10,830.000 ms on this machine.

Run the same experiment from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --local-files-only
```

Remove `--local-files-only` to allow model downloads when they are not cached.
The command writes detailed rankings to
`data/evaluation/results/retrieval_results.json` and the aggregate table to
`data/evaluation/results/retrieval_summary.csv`.

## Actual measured results

### K=1

| Retriever | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.800 | 0.650 | 0.800 | 0.800 | 0.800 |
| BM25 | 0.500 | 0.350 | 0.500 | 0.500 | 0.500 |
| Hybrid RRF | 0.800 | 0.650 | 0.800 | 0.800 | 0.800 |
| Hybrid + Reranker | 0.800 | 0.650 | 0.800 | 0.800 | 0.800 |

### K=3

| Retriever | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.900 | 0.850 | 0.367 | 0.850 | 0.816 |
| BM25 | 1.000 | 1.000 | 0.433 | 0.750 | 0.807 |
| Hybrid RRF | 0.900 | 0.900 | 0.400 | 0.850 | 0.855 |
| Hybrid + Reranker | 0.900 | 0.900 | 0.400 | 0.850 | 0.855 |

### K=5 and warm-model latency

| Retriever | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | Retrieval ms | Rerank ms | Total ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.900 | 0.900 | 0.240 | 0.850 | 0.843 | 32.057 | 0.000 | 32.057 |
| BM25 | 1.000 | 1.000 | 0.260 | 0.750 | 0.807 | 0.352 | 0.000 | 0.352 |
| Hybrid RRF | 1.000 | 1.000 | 0.260 | 0.875 | 0.898 | 30.378 | 0.000 | 30.378 |
| Hybrid + Reranker | 0.900 | 0.900 | 0.240 | 0.850 | 0.855 | 33.378 | 8700.835 | 8734.248 |

These are results from an actual local run saved with the repository artifacts.
They should be regenerated after changing documents, chunks, models, indexes,
or retrieval parameters.

## Findings

Hybrid RRF currently gives the strongest overall ranking at K=5. It matches
BM25's perfect Hit@5 and Recall@5 while producing the best MRR and NDCG.
BM25 is substantially faster and reaches perfect recall by K=3, but it places
the first relevant source lower on average.

The cross-encoder reranker does not improve this benchmark. It removes
`products.csv` from the top five for E08, “Which products have more than 12
months warranty?”, reducing Hit@5 and Recall@5 to 0.900. On CPU it also adds
about 8.7 seconds per query. This result supports leaving reranking configurable
and evaluating a smaller model or different candidate strategy before treating
it as a default quality improvement.

Dense retrieval also misses `products.csv` for E08. The production router sends
that query to structured retrieval, but this benchmark intentionally evaluates
each retrieval mode independently.

The dataset is small, source labels are coarse, and low Precision@5 partly
reflects having only one or two judged documents per question. Results should
not be generalized beyond this NovaCart corpus without adding more questions
and chunk-level relevance judgments.

## Optional generation checks

Core retrieval evaluation does not call Groq or any paid API. Generation checks
can be requested explicitly:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --local-files-only --evaluate-generation
```

This records lexical answer-relevance and faithfulness proxies, verifies that
citations refer to retrieved evidence, and checks abstention when there is no
evidence. These transparent heuristics are not substitutes for human review or
an LLM judge, and no generation scores are included in the measured tables
above.
