# conversational RAG

# Written for LangChain 1.3.x using pure LCEL — no legacy chain helpers.
# This is actually the preferred modern approach in 1.3+.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI

# ── Prompt 1: Question rewriter ──────────────────────────────────────────────
# Only used when chat_history is non-empty.
# Its sole job: convert a follow-up question into a standalone question
# that the retriever can search with effectively.
#
# Example:
#   History:    "What was GDP in 2013?" → "$15,228M"
#   Follow-up:  "What about 5 years later?"
#   Rewritten:  "What was Cambodia's GDP in 2018?"

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given a conversation history and the user's latest question,
rewrite the question as a complete standalone question that can be understood
without any conversation history.

Rules:
- Replace all pronouns (it, they, this, that) with their actual referents
- Replace relative time phrases (later, before, after, then) with specific values
- Keep the rewritten question concise and focused
- If the question is already standalone, return it unchanged
- Return ONLY the rewritten question, nothing else"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# ── Prompt 2: Answer generator ───────────────────────────────────────────────
# Sees the retrieved context + conversation history + current question.
# Generates the final answer with citations.

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant analyzing economic documents.

Answer the question using the context provided below.

Rules:
- If the context contains tables or numbers without clear headers,
  interpret them as year, GDP value (USD millions), and growth rate (%)
- Always include specific numbers from the context in your answer
- Cite the page number your answer comes from like: (Page 4)
- If the data is genuinely not in the context, say so clearly
- Keep answers concise but complete

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


def format_docs(docs: list) -> str:
    """
    Converts List[Document] into a single formatted string.

    Why we label each chunk with its source and page:
    The LLM sees this string as its context. By labeling each chunk,
    the LLM can say "According to Page 4..." in its answer,
    giving you traceable citations.
    """
    if not docs:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Source {i+1} | Page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


def build_conversational_rag_chain(vectorstore):
    """
    Builds a full conversational RAG chain using pure LCEL.

    Data flow:
    {"input": question, "chat_history": [...messages]}
        │
        ▼
    [Rewrite question if history exists]
        │
        ▼
    [Retrieve relevant chunks from vector store]
        │
        ▼
    [Format chunks into context string]
        │
        ▼
    [Generate answer with context + history + question]
        │
        ▼
    {"answer": str, "context": List[Document], "input": str}
    """

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # ── Step 1: Question rewriter ────────────────────────────────────────────
    # Takes {"input": str, "chat_history": list} → returns rewritten string
    # This is a simple chain: prompt → llm → extract string

    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    # ── Step 2: History-aware retriever ─────────────────────────────────────
    # RunnableBranch evaluates a condition and runs different logic per branch.
    #
    # Condition: is chat_history non-empty?
    #   True  → rewrite the question first, then retrieve
    #   False → retrieve with the original question directly
    #
    # Both branches output List[Document].

    history_aware_retriever = RunnableBranch(
        (
            # Condition function — receives the full input dict
            # Returns True if chat_history exists and has messages
            lambda x: bool(x.get("chat_history")),

            # Branch 1: rewrite then retrieve
            # rewrite_chain takes the full dict {input, chat_history}
            # and returns a plain string (the rewritten question).
            # That string goes directly into the retriever.
            rewrite_chain | retriever
        ),

        # Default branch: no history — extract "input" string and retrieve
        # RunnableLambda wraps a plain Python function as a Runnable
        RunnableLambda(lambda x: x["input"]) | retriever
    )

    # ── Step 3: Full RAG chain ───────────────────────────────────────────────
    # This is the complete pipeline written as pure LCEL.
    #
    # Input dict: {"input": str, "chat_history": list}
    #
    # The dict runnable at the top splits into parallel paths:
    #   "context" path:
    #       history_aware_retriever → List[Document]
    #       → format_docs → formatted string
    #   "input" path:
    #       RunnablePassthrough → passes "input" string unchanged
    #   "chat_history" path:
    #       RunnablePassthrough → passes "chat_history" list unchanged
    #
    # All three outputs merge into a dict that fills ANSWER_PROMPT.
    # Then llm generates the response.
    # Then StrOutputParser extracts the string.

    rag_chain = (
        RunnablePassthrough.assign(
            # .assign() adds new keys to the input dict without removing existing ones.
            # This is cleaner than a raw dict when you want to keep all input keys.
            #
            # "context" key: run retriever + format_docs, store result as "context"
            context=RunnableLambda(
                lambda x: format_docs(history_aware_retriever.invoke(x))
            )
        )
        | ANSWER_PROMPT   # receives {input, chat_history, context}
        | llm             # receives formatted messages, returns AIMessage
        | StrOutputParser() # extracts string from AIMessage
    )

    return rag_chain