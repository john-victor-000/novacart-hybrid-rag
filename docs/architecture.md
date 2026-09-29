# Architecture

NovaCart separates data preparation, retrieval, orchestration, generation, and
delivery so each layer can be tested or replaced independently.

## Runtime request flow

```mermaid
sequenceDiagram
    participant Browser as React
    participant API as FastAPI
    participant Chat as ChatService
    participant Router as QueryRouter
    participant Retrieval as Retrieval services
    participant Context as ContextBuilder
    participant LLM as Groq

    Browser->>API: POST /api/chat
    API->>Chat: validated query + conversation_id
    Chat->>Chat: resolve history and narrow follow-up
    Chat->>Router: effective retrieval query
    Router-->>Chat: route + reason
    Chat->>Retrieval: dense, hybrid, structured, or multi-source
    Retrieval-->>Context: normalized evidence + metadata
    Context-->>LLM: bounded numbered evidence
    LLM-->>Chat: concise answer with source labels
    Chat->>Chat: rebuild citations from evidence
    Chat-->>Browser: answer + route + sources + timing
```

FastAPI routes contain validation and HTTP error mapping. `ChatService` owns
conversation behavior, while `UnifiedRAGService` owns routing, retrieval,
context construction, and generation. Retrieval implementations share a
normalized result model, and all routes become normalized `Evidence` before
context construction.

## Data preparation flow

```mermaid
flowchart LR
    RAW[PDF / DOCX / CSV] --> ING[Normalized ingestion records]
    ING --> CH[Structure-aware chunks]
    CH --> EMB[Local embeddings]
    EMB --> VS[(Persistent Chroma)]
    CH --> BI[(Persistent BM25 index)]
    RAW --> PR[Product CSV repository]
```

PDF pages and DOCX sections retain their structure. CSV rows remain whole
records. Chunk IDs and source metadata survive embedding, indexing, retrieval,
context construction, and citation generation.

## Trust boundaries

- LLM output never creates source metadata. Citation labels are resolved against
  evidence included in the prompt.
- Empty evidence skips generation and returns a fixed not-found response.
- Structured product facts are loaded through a repository abstraction rather
  than read directly inside routes.
- API users receive stable service errors without provider stack traces.
- CORS accepts only explicitly configured origins; wildcard origins are
  rejected during settings validation.

## Deployment

Local development runs Vite and Uvicorn separately. Docker Compose uses:

- an optional one-shot indexer,
- FastAPI with named volumes for indexes and model cache,
- an Nginx container serving the React build and proxying `/api`.

The vector store is embedded Chroma, so no Qdrant service is present.
