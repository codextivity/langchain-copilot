# ✅ Pydantic data models

# src/schemas.py
# Pydantic models that define the structured data shapes
# the LLM can extract from documents.
#
# Why Pydantic?
# - Automatic type validation (catches "15,228" vs 15228.0)
# - Clear documentation of what each field means via Field(description=)
# - The description strings are critical — they get sent to the LLM
#   as instructions for what to put in each field
# - IDE autocomplete on the returned objects

from pydantic import BaseModel, Field
from typing import Optional

class GDPDataPoint(BaseModel):
    """
    A single year's economic data point.
    Used when extracting one specific year from the document.
    """

    year: int = Field(
        description="The calendar year this data point refers to"
    )
    gdp_usd_millions: float = Field(
        description="GDP value in millions of US dollars"
    )
    growth_rate_percent: Optional[float] = Field(
        default=None,
        description="Annual GDP growth rate as a percentage. None if not available."
    )
    data_source_page: Optional[int] = Field(
        default=None,
        description="Page number in the document where this data was found"
    )

class EconomicSummary(BaseModel):
    """
    A summary of economic performance across a time period.
    Used when the user asks for an overview or comparison.
    """

    country: str = Field(
        description="Name of the country being analyzed"
    )
    period_start: int = Field(
        description="First year of the analysis period"
    )
    period_end: int = Field(
        description="Last year of the analysis period"
    )
    gdp_start_usd_millions: float = Field(
        description="GDP at the start of the period in millions USD"
    )
    gdp_end_usd_millions: float = Field(
        description="GDP at the end of the period in millions USD"
    )
    total_growth_percent: float = Field(
        description="Total percentage growth from period_start to period_end"
    )
    average_annual_growth_percent: float = Field(
        description="Average annual growth rate across the period"
    )
    key_growth_drivers: list[str] = Field(
        description="List of main factors that drove economic growth"
    )
    major_challenges: list[str] = Field(
        description="List of major economic challenges or shocks during the period"
    )
    data_source_pages: list[int] = Field(
        description="List of page numbers where this information was found"
    )

class DocumentInsight(BaseModel):
    """
    High-level structured insight extracted from the full document.
    Used when the user wants a structured overview of the entire document.
    """

    document_title: str = Field(
        description="Title or subject of the document"
    )
    main_topic: str = Field(
        description="One sentence describing what the document is about"
    )
    time_period_covered: str = Field(
        description="The time range covered by the document e.g. 2000-2023"
    )
    key_findings: list[str] = Field(
        description="List of 3-5 most important findings from the document"
    )
    data_quality: str = Field(
        description="Assessment of data completeness: complete, partial, or limited"
    )