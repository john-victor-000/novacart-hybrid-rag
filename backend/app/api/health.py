"""Application liveness endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Confirm the API is running and can serve requests."""
    return {"status": "ok"}
