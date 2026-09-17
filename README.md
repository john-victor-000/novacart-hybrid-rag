# NovaCart Hybrid RAG Knowledge Assistant

A learning and portfolio project for a fictional e-commerce knowledge assistant,
built incrementally. **Current implementation: Phase 6**: a minimal FastAPI
backend plus local dataset ingestion, chunking, metadata refinement, and
embeddings with persistent dense retrieval and a baseline Groq RAG pipeline.

BM25, hybrid retrieval, RRF, reranking, structured retrieval, routing, RAG/LLM
generation, and the frontend are reserved for later phases.

## Local setup (Windows PowerShell)

Use Python 3.11 or newer. Run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The explicit virtual-environment Python path avoids needing to activate it or
change PowerShell's execution policy. In your IDE, select
`.venv\Scripts\python.exe` as the Python interpreter.

Configuration is optional. If `.env` does not exist, create it from the example:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit `.env` to override these defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `NovaCart Hybrid RAG Knowledge Assistant` | Application title in API documentation |
| `APP_DEBUG` | `false` | FastAPI debug mode; keep disabled outside local development |
| `CHUNK_SIZE` | `900` | Target maximum text characters per non-CSV chunk |
| `CHUNK_OVERLAP` | `150` | Target text overlap between adjacent non-CSV chunks |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Local Sentence Transformers model |
| `EMBEDDING_BATCH_SIZE` | `16` | Number of chunks embedded per provider call |
| `EMBEDDING_LOCAL_FILES_ONLY` | `false` | Load embedding model only from local cache |
| `VECTOR_DB_PATH` | `data/vector_store` | Local ChromaDB persistence directory |
| `VECTOR_COLLECTION_NAME` | `novacart_chunks` | ChromaDB collection used for NovaCart chunks |
| `RETRIEVAL_TOP_K` | `5` | Default number of dense retrieval results |
| `GROQ_API_KEY` | empty | Groq API key; required for answer generation |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible API base URL |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq generation model |
| `LLM_TIMEOUT_SECONDS` | `120` | Groq request timeout in seconds |

`backend/app/core/config.py` uses Pydantic Settings to load and validate
configuration. Environment variables take priority over the root `.env` file,
which takes priority over defaults. The `.env` path is resolved relative to the
source file, independently of the working directory. Unrelated `.env` keys are
ignored. Invalid boolean values cause startup to fail with a validation error.
Restart the server after changing configuration. `.env` is excluded from Git.

## Run the backend

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Uvicorn runs the HTTP server and logs startup, shutdown, and requests to the
terminal. `backend/app/main.py` creates the FastAPI application from settings and
registers the router in `backend/app/api/health.py`. Keeping configuration and
routes separate makes the entry point small as later phases are added.

Interactive API documentation: <http://127.0.0.1:8000/docs>.
Stop the server with `Ctrl+C`.

## Check the health endpoint

With the server running, open a second PowerShell terminal:

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Expected: HTTP `200 OK`, JSON content type, and this body:

```json
{"status":"ok"}
```

This endpoint checks API liveness only; it does not check any external services.

Check installed dependency compatibility with:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

## Ingest source documents

Place NovaCart source files in `data/raw/`. Phase 2 supports PDF, DOCX, and CSV
files:

- PDF files are parsed with PyMuPDF and produce one record per page.
- DOCX files are parsed with `python-docx` and produce one record per detected
  section.
- CSV files are parsed with pandas and produce one record per row.

Each record uses the normalized internal model in
`backend/app/ingestion/models.py` with these fields: `document_id`,
`document_name`, `document_type`, `source`, `page`, `section`, and `text`.

Run ingestion from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest_documents --input-dir data/raw
```

Expected output is a concise summary like:

```text
Ingested 16 records from 7 files
- products.csv | type=csv | records=5 | text_length=683 | rows=1-5
- shipping_policy.pdf | type=pdf | records=1 | text_length=970 | pages=1-1
```

Run the Phase 2 tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ingestion.py -q
```

## Chunk source documents

Phase 3 keeps chunking separate from parsing. Ingestion produces normalized
records, then `backend/app/chunking/chunker.py` turns those records into
retrieval-sized chunks. Each chunk preserves the source metadata from ingestion
and adds a unique `chunk_id` plus `chunk_index`.

For PDF and DOCX records, chunking respects the existing page or section record
boundaries first. Longer text is split on paragraph and sentence boundaries
where possible, with configurable overlap. CSV product records stay as one
logical chunk per row so a product row is not fragmented.

Run ingestion plus chunking from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.chunk_documents --input-dir data/raw
```

Override chunk settings when needed:

```powershell
.\.venv\Scripts\python.exe -m scripts.chunk_documents --input-dir data/raw --chunk-size 700 --chunk-overlap 100
```

Expected output is a concise summary like:

```text
Created 20 chunks from 16 ingested records across 7 files
chunk_size=900 | chunk_overlap=150
- products.csv | chunks=5 | ids=['43307a0aaf99', '2170143e8663', 'd119ef1e387b'] | sample_lengths=[130, 132, 147] | type=csv | page=none | section=row 1 | source=F:\novacart-hybrid-rag\data\raw\products.csv
```

Run the Phase 3 tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chunking.py -q
```

## Embed chunks

Phase 4 keeps embeddings independent from future vector database storage.
`backend/app/embeddings/providers.py` defines the provider interface and the
initial Sentence Transformers provider. `backend/app/embeddings/service.py`
turns `DocumentChunk` objects into embedding records while preserving metadata.

The default model is `sentence-transformers/all-MiniLM-L6-v2`, a compact local
open-source sentence embedding model with 384-dimensional vectors. It is small
enough for local development while still useful for semantic retrieval
experiments.

Run ingestion, chunking, and embedding from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.embed_documents --input-dir data/raw
```

After the model is cached, offline/cache-only runs can skip Hugging Face
metadata checks:

```powershell
.\.venv\Scripts\python.exe -m scripts.embed_documents --input-dir data/raw --local-files-only
```

Expected output is a concise summary like:

```text
Embedded 20 chunks
embedding_model=sentence-transformers/all-MiniLM-L6-v2
embedding_dimension=384
sample_chunk_id=6ca72c8512077a2da68692f340d8bd96f296d65dce77efaa28a3bc26937d5123
sample_metadata=document_id=b021631de09b944daadde43fec39375fc312ef4c3e55e02d545ff99676b4c0e5, document_name=company_information.pdf, document_type=pdf, source=F:\novacart-hybrid-rag\data\raw\company_information.pdf, page=1, section=none
sample_embedding_length=384
```

Run the Phase 4 tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_embeddings.py -q
```

## Index and search vectors

Phase 5 stores embedded chunks in a persistent local ChromaDB collection.
`backend/app/retrieval/vector_store.py` provides a replaceable vector-store
interface and the ChromaDB implementation. `backend/app/retrieval/retriever.py`
embeds a query through the Phase 4 provider before asking the vector store for
the nearest chunks.

Run the complete indexing pipeline from the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.index_vectors --input-dir data/raw --local-files-only
```

Omit `--local-files-only` on the first run if the embedding model is not cached.
The default index is stored under `data/vector_store/`, which is excluded from
Git. Re-running the command skips chunk IDs already stored in the collection.
A successful first run ends with output similar to:

```text
ingested_records=16
chunks=20
embedded_chunks=20
newly_indexed=20
collection=novacart_chunks
db_path=F:\novacart-hybrid-rag\data\vector_store
total_indexed=20
```

Search the persisted index without recreating document embeddings:

```powershell
.\.venv\Scripts\python.exe -m scripts.query_vectors "What is the return policy?" --local-files-only
```

Each result includes a cosine similarity score, chunk ID, document name and
type, page or section, source path, and a text preview. Scores closer to `1.0`
mean stronger cosine similarity; rankings are most useful when compared within
the same query and embedding model.

Run the focused Phase 5 tests or the complete suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

## Ask questions with baseline RAG

Phase 6 sends a question through dense retrieval, builds labeled context, and
asks a configurable Groq model to answer only from that context. Source
objects are constructed from retrieved chunk metadata rather than generated by
the model.

Create a Groq API key, then place it in the root `.env` file. Do not commit the
key:

```dotenv
GROQ_API_KEY="gsk_your_key_here"
GROQ_MODEL="llama-3.1-8b-instant"
```

Index the documents with the Phase 5 command before asking questions. Then use
the terminal RAG client:

```powershell
.\.venv\Scripts\python.exe -m scripts.ask_rag "What is the standard return window?" --local-files-only
```

Add `--show-chunks` to include retrieval scores and text previews. You can also
ask through the API by starting Uvicorn and sending a request from a second
terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
curl.exe -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" --data '{"query":"What is the return period?"}'
```

A successful response has this shape:

```json
{
  "answer": "Most eligible products can be returned within 7 calendar days after delivery. [Source 1]",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "document_name": "return_policy.pdf",
      "document_type": "pdf",
      "source": "data/raw/return_policy.pdf",
      "page": 1,
      "section": null
    }
  ],
  "retrieved_chunks": null
}
```

Set `"include_retrieved_chunks": true` in the request to expose the retrieved
text and similarity scores for debugging. Run the Phase 6 tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag.py -q
```

## Troubleshooting

- **Module not found:** run from the repository root and install requirements
  using the virtual-environment Python shown above.
- **Port 8000 is occupied:** use `--port 8001` and update the health URL to match.
- **Connection refused:** confirm Uvicorn reports successful application startup
  and remains running in the first terminal.
- **Settings validation error:** check `.env` and environment variables;
  `APP_DEBUG=false` is a valid value.
- **No ingestion records:** confirm source files are directly under `data/raw/`
  and use one of `.pdf`, `.docx`, or `.csv`.
- **Parser import error:** reinstall dependencies with
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **Unreadable source file:** open the file locally to confirm it is not corrupt
  or password protected.
- **Chunk settings validation error:** ensure `CHUNK_OVERLAP` is smaller than
  `CHUNK_SIZE`.
- **Unexpectedly large CSV chunk:** CSV rows are intentionally kept whole to
  preserve product record integrity.
- **Model download fails:** confirm internet access to Hugging Face, use
  `--local-files-only` after the model is cached, or use a local model path in
  `EMBEDDING_MODEL_NAME`.
- **Embedding run is slow the first time:** the model is downloaded and cached
  on first use; later runs reuse the cache.
- **Embedding dimension mismatch:** verify one provider is used consistently for
  the whole batch.
- **Empty search results:** run the indexing command first and confirm its
  `collection` and `db_path` match the query command.
- **Chroma dimension error:** rebuild under a new collection name after changing
  the embedding model; one collection must use one embedding dimension.
- **Database file is locked:** close other indexing/query processes using the
  same local ChromaDB path and retry.
- **Missing Groq API key:** set `GROQ_API_KEY` in the root `.env` file and
  restart Uvicorn.
- **Groq 401 error:** verify that the API key is valid and belongs to the
  intended Groq project.
- **Groq 403 or model unavailable:** verify model permissions for the value in
  `GROQ_MODEL`. `llama-3.1-8b-instant` is currently listed for Enterprise use
  and has been retired from free/developer usage.
- **Answer says information was not found:** inspect retrieved chunks with
  `--show-chunks`; dense retrieval may not have supplied relevant evidence.

The implementation follows the official [FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
and [Pydantic Settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
