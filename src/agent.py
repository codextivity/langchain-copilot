# # ✅ LangGraph agent with tool calling
# src/agent.py — full updated file
# Connects document retrieval to the tool-calling agent

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from tools import calculator, get_current_date, compute_growth_rate, web_search

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# We inject the retrieved document context directly into the system prompt.
# This is called "context injection" — the agent always sees relevant
# document chunks alongside its tool definitions.
# The agent then decides whether to answer from context or use a tool.

def build_agent_system_prompt(context: str) -> str:
    """
    Builds the system prompt with document context injected.

    Why inject context into the system prompt rather than the human message?
    Because the system prompt is treated as persistent instructions.
    Document context belongs there — it is background knowledge,
    not part of the conversation.
    """
    return f"""You are a research assistant specializing in economic analysis.

You have access to the following document context:
──────────────────────────────────────────────────
{context}
──────────────────────────────────────────────────

You also have access to these tools:
- calculator: for precise arithmetic and percentage calculations
- compute_growth_rate: for GDP growth rate and CAGR calculations
- get_current_date: for date and year information
- web_search: for ANY question not answerable from the document context above

Decision rules — follow these strictly in order:
1. If the question can be answered from the document context above → answer directly
2. If the question requires calculation on known numbers → use calculator or compute_growth_rate
3. If the question is about topics NOT covered in the document context → use web_search immediately
4. Only say you cannot answer if web_search also returns no relevant results

Critical: Never refuse a question without first attempting web_search.
If it is not in the documents, search the web.
"""

def build_research_agent(vectorstore):
    """
    Builds a LangGraph agent connected to the document vector store.

    Args:
        vectorstore: ChromaDB vector store with ingested documents

    Returns:
        A compiled LangGraph application
    """
    tools = [calculator, get_current_date, compute_growth_rate, web_search]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Build retriever to fetch relevant document context
    # for each user question
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_context(docs: list) -> str:
        """Format retrieved docs into a context string for the system prompt."""
        if not docs:
            return "No relevant document context found."
        formatted = []
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "?")
            formatted.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def agent_node(state: AgentState) -> dict:
        """
        Runs the LLM with document context injected into the system prompt.

        On every turn:
        1. Extract the latest human message
        2. Retrieve relevant document chunks for that question
        3. Inject chunks into system prompt
        4. Run LLM with tools available
        """
        # Find the most recent human message to use as retrieval query
        # We search backwards through messages to find the last human turn
        last_human_message = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_human_message = msg.content
                break

        # Retrieve relevant document context for this question
        docs = retriever.invoke(last_human_message)
        context = format_context(docs)

        # Build system prompt with fresh context for this question
        system_prompt = build_agent_system_prompt(context)

        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tools_node = ToolNode(tools)

    def should_continue(state: AgentState) -> str:
        """Route to tools if LLM made a tool call, otherwise end."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(agent, question: str, chat_history: list = None) -> str:
    """Run the agent and return the final answer string."""
    messages = list(chat_history or [])
    messages.append(HumanMessage(content=question))
    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content