# NovaCart Hybrid RAG Knowledge Assistant

A learning and portfolio project for a fictional e-commerce knowledge assistant,
built incrementally. **Current implementation: Phase 2**: a minimal FastAPI
backend plus local dataset and document ingestion.

Chunking, embeddings, vector databases, BM25, RAG, reranking, routing, and the
frontend are reserved for later phases. The existing repository directories are
preserved.

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

The implementation follows the official [FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
and [Pydantic Settings documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
