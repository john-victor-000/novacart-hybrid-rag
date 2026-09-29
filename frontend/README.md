# NovaCart React client

This Vite React application is a thin client for the existing FastAPI
`POST /api/chat` endpoint. It contains no retrieval or generation logic.

```powershell
npm.cmd install
npm.cmd run dev
```

Vite proxies API requests to `http://127.0.0.1:8000` during local development.
Copy `.env.example` to `.env` to override the API URL, request timeout, or debug
display. Build production assets with `npm.cmd run build`.
