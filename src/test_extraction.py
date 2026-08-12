# src/test_extraction.py
# Run this to confirm structured extraction works correctly.

from dotenv import load_dotenv
load_dotenv()

from ingestion import load_existing_vectorstore
from extraction_chain import build_extraction_chain
from schemas import GDPDataPoint, EconomicSummary, DocumentInsight
import json

vectorstore = load_existing_vectorstore()

print("=" * 60)
print("TEST 1: Extract a single GDP data point")
print("=" * 60)

chain = build_extraction_chain(vectorstore, GDPDataPoint)

result = chain.invoke({
    "query": "Cambodia GDP 2013 growth rate",
    "extraction_target": "GDP data for Cambodia in the year 2013 including growth rate"
})

# result is a GDPDataPoint object — fully typed and validated
print(f"Type: {type(result)}")
print(f"Year: {result.year}")
print(f"GDP: ${result.gdp_usd_millions:,.1f} million")
print(f"Growth rate: {result.growth_rate_percent}%")
print(f"Found on page: {result.data_source_page}")

# .model_dump() converts the Pydantic object to a plain dict
# Useful when you need to serialize to JSON or store in a database
print(f"\nAs dict: {json.dumps(result.model_dump(), indent=2)}")

print("\n" + "=" * 60)
print("TEST 2: Extract an economic summary across a period")
print("=" * 60)

chain2 = build_extraction_chain(vectorstore, EconomicSummary)

result2 = chain2.invoke({
    "query": "Cambodia GDP growth drivers challenges 2000 2023",
    "extraction_target": "Economic summary for Cambodia from 2000 to 2023 including growth drivers and challenges"
})

print(f"Country: {result2.country}")
print(f"Period: {result2.period_start} - {result2.period_end}")
print(f"GDP start: ${result2.gdp_start_usd_millions:,.1f}M")
print(f"GDP end: ${result2.gdp_end_usd_millions:,.1f}M")
print(f"Total growth: {result2.total_growth_percent}%")
print(f"Avg annual growth: {result2.average_annual_growth_percent}%")
print(f"\nGrowth drivers:")
for driver in result2.key_growth_drivers:
    print(f"  - {driver}")
print(f"\nMajor challenges:")
for challenge in result2.major_challenges:
    print(f"  - {challenge}")

print("\n" + "=" * 60)
print("TEST 3: Extract document-level insight")
print("=" * 60)

chain3 = build_extraction_chain(vectorstore, DocumentInsight)

result3 = chain3.invoke({
    "query": "document overview main topic findings",
    "extraction_target": "High level overview of what this document covers and its key findings"
})

print(f"Title: {result3.document_title}")
print(f"Topic: {result3.main_topic}")
print(f"Period: {result3.time_period_covered}")
print(f"Data quality: {result3.data_quality}")
print(f"\nKey findings:")
for i, finding in enumerate(result3.key_findings, 1):
    print(f"  {i}. {finding}")