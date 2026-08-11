from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

"""
What ChatPromptTemplate does:

"""


# In chain.py, replace RAG_PROMPT with this:
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant analyzing economic documents.

Answer the question using the context provided below.

Important rules:
- If the context contains a table or numbers without clear headers, 
  interpret them using surrounding context clues and your knowledge 
  of how such data is typically structured.
- If you see columns of numbers that appear to be year / value / percentage,
  treat them as year, GDP value, and growth rate respectively.
- Always show the specific numbers from the context in your answer.
- Cite which source and page your answer comes from.
- Only say you lack information if the data is genuinely absent,
  not if it requires interpretation.

Context:
{context}"""),
    ("human", "{question}")
])


"""
Why this function exists:
The retriever returns List[Document]. But the prompt template expects a string for {context}. 
This function bridges the gap.
"""
def format_docs(docs: list) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i+1}: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

"""
Builds the RAG chain with the specified vectorstore.
The .as_retriever() method creates a retriever that fetches the top k most relevant documents based on the query.

What k=4 means:
Return the 4 most similar chunks. More chunks = more context for the LLM but higher cost and risk of confusion. 
4 is a reasonable default. You'd increase this for complex questions that need broad context.

What search_type="similarity" means:
Pure cosine similarity — return the k vectors closest to the query vector. 
The alternative, "mmr", penalizes redundancy so you don't get 4 chunks that all say the same thing.
"""
def build_rag_chain(vectorstore):
    """Build and return the RAG chain."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", # Use pure similarity search (cosine distance)
        search_kwargs={"k": 4} # Return the 4 most relevant chunks for context
    )

    # Use a deterministic model for factual answers. gpt-4o-mini is cheaper than gpt-4o and still high quality.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # This is the key pattern — understand each step:
    # 1. RunnablePassthrough() passes the question through unchanged
    # 2. retriever takes the question, returns List[Document]
    # 3. format_docs converts docs to a string
    # 4. The dict feeds both "context" and "question" into the prompt
    
    rag_chain = ({
        "context": retriever | RunnableLambda(format_docs), # Convert retrieved docs to a single string
        "question": RunnablePassthrough() # Pass the question through unchanged
        }
        | RAG_PROMPT # Fill the prompt with context and question
        | llm # Generate the answer
        | StrOutputParser() # Convert the LLM's output to a string
    )
    
    return rag_chain # Return the constructed RAG chain