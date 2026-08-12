# app/api/routes/extract.py
# Handles structured data extraction from ingested documents.
# Uses the Pydantic schemas we defined in schemas.py to extract
# typed, validated data instead of prose answers.

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.core.extraction_chain import build_extraction_chain
from app.core.schemas import GDPDataPoint, EconomicSummary, DocumentInsight

router = APIRouter()

# ── Request and response models ───────────────────────────────────────────────
# These define exactly what the client sends and what they get back.
# FastAPI uses these for automatic validation and documentation.

class ExtractRequest(BaseModel):
    query: str                    # What to search for in the documents
    extraction_target: str        # Description of what data to extract
    schema_type: str              # Which Pydantic schema to use

class ExtractResponse(BaseModel):
    schema_type: str
    data: dict                    # The extracted data as a plain dict

# Map schema type strings to actual Pydantic classes.
# The client sends a string like "gdp_datapoint" — we look up the class here.
# This avoids exposing class names directly in the API.
SCHEMA_MAP = {
    "gdp_datapoint": GDPDataPoint,
    "economic_summary": EconomicSummary,
    "document_insight": DocumentInsight,
}

@router.post("", response_model=ExtractResponse)
async def extract_structured_data(request: Request, body: ExtractRequest):
    """
    Extract structured data from ingested documents.

    schema_type options:
    - "gdp_datapoint"     → extracts year, GDP value, growth rate, page number
    - "economic_summary"  → extracts country, period, growth drivers, challenges
    - "document_insight"  → extracts title, topic, key findings, data quality

    Example request:
    {
        "query": "Cambodia GDP 2013 growth rate",
        "extraction_target": "GDP data for Cambodia in 2013",
        "schema_type": "gdp_datapoint"
    }
    """
    vectorstore = request.app.state.vectorstore

    # Guard: no documents ingested yet
    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Use POST /ingest first."
        )

    # Guard: invalid schema type
    if body.schema_type not in SCHEMA_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid schema_type. Choose from: {list(SCHEMA_MAP.keys())}"
        )

    # Look up the correct Pydantic class
    schema_class = SCHEMA_MAP[body.schema_type]

    # Build and run the extraction chain
    # build_extraction_chain returns a chain that outputs a Pydantic object
    chain = build_extraction_chain(vectorstore, schema_class)

    result = chain.invoke({
        "query": body.query,
        "extraction_target": body.extraction_target
    })

    # .model_dump() converts the Pydantic object to a plain dict
    # so FastAPI can serialize it to JSON
    return ExtractResponse(
        schema_type=body.schema_type,
        data=result.model_dump()
    )