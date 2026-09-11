"""FastAPI entry point, served by Uvicorn."""

from fastapi import FastAPI

from backend.app.api.health import router as health_router
from backend.app.core.config import Settings

settings = Settings()

app = FastAPI(title=settings.app_name, debug=settings.app_debug)
app.include_router(health_router)
