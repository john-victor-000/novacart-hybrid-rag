# Technical decisions

## Local Chroma for dense storage

**Decision:** use persistent Chroma rather than requiring a separate vector
database service.

**Reason:** it keeps local setup and portfolio deployment small while retaining
a clean `VectorStore` abstraction. A managed or remote store can replace it
later.

## Independent BM25 and dense retrievers

**Decision:** keep sparse and dense retrieval independently callable.

**Reason:** independent rankings support evaluation and make fusion observable.
RRF combines positions instead of incompatible raw scores.

## Deterministic routing before agentic routing

**Decision:** use explicit intent and SKU rules.

**Reason:** the NovaCart domain is small enough for understandable rules.
Routing reasons are logged, easy to test, and do not require another model
call.

## Structured repository for product facts

**Decision:** access `products.csv` through a repository/service boundary.

**Reason:** exact prices, warranties, stock states, and filters should be
deterministic. The boundary also keeps a future CSV-to-SQL migration local.

## Groq behind an LLM interface

**Decision:** use Groq with `llama-3.1-8b-instant` for generation.

**Reason:** the provider is configurable and isolated behind `LLMProvider`.
Retrieval tests and core evaluation do not require an external paid API.

## Server-owned citations

**Decision:** citations are reconstructed from retrieved evidence.

**Reason:** LLM text is not a trusted source of document names, pages, sections,
or chunk IDs.

## Process-local conversation history

**Decision:** start with a thread-safe in-memory repository and deterministic
SKU follow-up rewriting.

**Reason:** it proves the API contract without introducing Redis or an agentic
memory framework. The repository interface leaves room for durable storage.

## Reranking stays optional

**Decision:** retain the reranker but keep it configurable.

**Reason:** measured Stage C results show plain Hybrid RRF outperforming the
current reranked configuration at K=5, with far lower CPU latency.

## React with Vite and Nginx

**Decision:** use a small client-side React application built by Vite and served
by Nginx in Docker.

**Reason:** the UI needs one chat view and does not benefit from a larger
frontend framework. Nginx serves immutable assets and proxies API requests,
avoiding a hard-coded container hostname in browser code.
