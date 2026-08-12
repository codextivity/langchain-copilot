# app/api/routes/health.py
# Simple health check endpoint.
# Always build this first — it confirms the API started correctly
# before you test any complex logic.

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

from fastapi.responses import RedirectResponse

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


@router.get("/", include_in_schema=False)
async def root():
    """
    Redirects visitors from the root URL to the API documentation.

    include_in_schema=False means this route is hidden from the
    /docs page — it is a utility route, not part of the API contract.
    """
    return RedirectResponse(url="/docs")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns API status and timestamp.
    Use this to confirm the service is running before testing other endpoints.
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )