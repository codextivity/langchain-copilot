# ✅ calculator, date, growth rate, web search

# src/tools.py
# Tool definitions for the research agent.
#
# The @tool decorator does two things:
# 1. Wraps the function so LangChain can call it
# 2. Uses the function's docstring as the tool description
#    sent to the LLM — so it knows when and how to use each tool
#
# Critical rule: write docstrings as if you are explaining the tool
# to the LLM, not to a human developer. The LLM reads these to decide
# which tool to call.

from langchain_core.tools import tool
from datetime import datetime
import math

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.

    Use this tool whenever the user asks for:
    - Percentage calculations
    - Growth rate calculations
    - Arithmetic on numbers extracted from documents
    - Any computation where precision matters

    Args:
        expression: A valid Python math expression as a string.
                   Examples: "15228 * 1.0835", "(31940 - 15228) / 15228 * 100"

    Returns:
        The result as a string, or an error message if the expression is invalid.
    """
    try:
        # We use a restricted evaluation environment for safety.
        # Only math functions are available — no imports, no file access,
        # no system calls. Never use raw eval() on user input in production.
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            **{name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{result:.4f}" if isinstance(result, float) else str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def get_current_date() -> str:
    """
    Returns today's date and current year.

    Use this tool when:
    - The user asks about the current date or year
    - You need to calculate how many years ago something happened
    - You need to determine if data is recent or outdated

    Returns:
        Current date as a formatted string including day, month, year.
    """
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}. Current year: {now.year}."

@tool
def compute_growth_rate(start_value: float, end_value: float, periods: int = 1) -> str:
    """
    Computes growth rate between two values over a number of periods.

    Use this tool when the user asks about:
    - GDP growth between two years
    - Percentage change between two economic figures
    - Compound annual growth rate (CAGR) over multiple years

    Args:
        start_value: The initial value (e.g. GDP at the start year)
        end_value:   The final value (e.g. GDP at the end year)
        periods:     Number of periods between start and end (default 1).
                    For CAGR over multiple years, pass the number of years.

    Returns:
        A string showing both simple percentage change and CAGR if periods > 1.
    """
    if start_value == 0:
        return "Error: start_value cannot be zero"

    # Simple percentage change — always calculated
    simple_change = ((end_value - start_value) / start_value) * 100

    if periods == 1:
        return f"Growth: {simple_change:.2f}%"

    # CAGR formula: (end/start)^(1/periods) - 1
    # This gives the smoothed annual rate that would produce
    # the same total growth if applied consistently each year
    cagr = ((end_value / start_value) ** (1 / periods) - 1) * 100

    return (
        f"Total growth: {simple_change:.2f}%\n"
        f"CAGR over {periods} periods: {cagr:.2f}% per period"
    )

# Replace the web_search function in src/tools.py

@tool
def web_search(query: str) -> str:
    """
    Searches the web for current information not available in the documents.

    Use this tool when:
    - The user asks about information not in the uploaded documents
    - The user asks to compare document data with current real-world figures
    - The user asks about recent events after the document's time period

    Do NOT use this for questions answerable from the uploaded documents.

    Args:
        query: A clear, specific search query string

    Returns:
        Search results as a formatted string with sources
    """
    import os
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable: TAVILY_API_KEY not set in .env"

    client = TavilyClient(api_key=api_key)

    # search_depth="basic" is faster and cheaper.
    # search_depth="advanced" does deeper research but costs more credits.
    # max_results=3 is enough context without overwhelming the LLM.
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=3
    )

    # Format results into a clean string the LLM can read
    results = []
    for i, result in enumerate(response["results"], 1):
        results.append(
            f"[Result {i}]\n"
            f"Title: {result['title']}\n"
            f"Source: {result['url']}\n"
            f"Content: {result['content']}"
        )

    return "\n\n".join(results) if results else "No results found."