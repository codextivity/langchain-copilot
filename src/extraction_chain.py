
# ✅ structured data extraction

# src/extraction_chain.py
# Chains that extract structured data from retrieved document chunks.
#
# Design decision: why a separate file from chain.py?
# chain.py handles conversational Q&A — its output is prose for humans.
# extraction_chain.py handles structured extraction — its output is
# typed data for downstream processing.
# Keeping them separate makes each easier to test and modify independently.

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from schemas import GDPDataPoint, EconomicSummary, DocumentInsight

# ── Extraction prompt ────────────────────────────────────────────────────────
# This prompt is different from the RAG answer prompt in one key way:
# it explicitly instructs the LLM to extract only what is present
# and not invent data. Hallucination in structured output is dangerous
# because it looks authoritative — a made-up float looks just as valid
# as a real one.

EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
You are a precise data extraction assistant.

Extract the requested information from the context below.

Critical rules:
- Only extract data that is explicitly present in the context
- Do not calculate or infer values unless the calculation is trivial
  and all inputs are present (e.g. percentage from two known values)
- Use null/None for any field where data is not available
- Be exact with numbers — do not round unless the source already rounded

Context:
{context}

Extract: {extraction_target}
""")

def format_docs_for_extraction(docs: list) -> str:
    """
    Same as format_docs in chain.py but optimized for extraction.
    Includes more metadata to help the LLM cite sources accurately.
    """
    if not docs:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("file_name", "unknown")
        formatted.append(
            f"[Document: {source} | Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)

def build_extraction_chain(vectorstore, schema_class):
    """
    Builds a chain that retrieves relevant chunks and extracts
    structured data matching the given Pydantic schema.

    Args:
        vectorstore: the ChromaDB vector store
        schema_class: a Pydantic model class (GDPDataPoint, EconomicSummary, etc.)

    Returns:
        A chain that takes {"query": str, "extraction_target": str}
        and returns a validated instance of schema_class

    Why pass schema_class as a parameter?
    So this one function can build extraction chains for any schema.
    You do not need a separate function for each data type.
    """

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # Note: using gpt-4o here instead of gpt-4o-mini.
    # Structured extraction requires precise instruction following.
    # gpt-4o-mini sometimes drops optional fields or misformats numbers.
    # For production, measure accuracy on both and choose based on results.

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
        # k=6 instead of k=4 because extraction often needs more context
        # to find all the pieces of a structured data point
    )

    # .with_structured_output() does three things:
    # 1. Converts schema_class to a JSON schema
    # 2. Passes it to the LLM as a tool definition
    # 3. Parses and validates the LLM response against the schema
    # The result is a validated Python object, not a string
    structured_llm = llm.with_structured_output(schema_class)

    extraction_chain = (
        {
            # Retrieve relevant chunks using the query
            # then format them into a context string
            "context": RunnableLambda(lambda x: x["query"])
                       | retriever
                       | RunnableLambda(format_docs_for_extraction),

            # Pass the extraction target description through unchanged
            # This tells the LLM what specific data to extract
            "extraction_target": RunnableLambda(lambda x: x["extraction_target"])
        }
        | EXTRACTION_PROMPT
        | structured_llm  # returns validated Pydantic object, not string
    )

    return extraction_chain