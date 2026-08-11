# src/debug_retrieval.py

from dotenv import load_dotenv
load_dotenv()  # Must be called before any LangChain imports so API keys are available

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",  """ Rewrite the user's follow-up question into a standalone question 
    that can be understood without any conversation history."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
rewriter_chain = REWRITE_PROMPT | llm | StrOutputParser()

# Simulate a conversation history with two previous messages
chat_history = [
    HumanMessage(content="What was GDP in 2013?"),
    AIMessage(content="Cambodia's GDP in 2013 was $15,228M")
]

# Test the rewriter with a follow-up question
follow_up_question = [
    "What about 5 years later?",
    "What was the GDP growth rate in 2018?",
    "How did Cambodia's economy perform in 2018 compared to 2013?",
    "How much did Cambodia's GDP increase from 2013 to 2018?",
    "What caused the change in GDP between 2013 and 2018?",
    "Was it higher than neighboring countries?"
    ]


print("Questions Rewriting Test:\n")
print("=" * 60)

for question in follow_up_question:
    rewritten_question = rewriter_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
    print(f"Original: {question}")
    print(f"Rewritten: {rewritten_question}\n")
    print("-" * 60)














# from ingestion import load_existing_vectorstore

# # ── Step 1: Load the vector store we already built ──────────────────────────
# vectorstore = load_existing_vectorstore()

# # ── Step 2: Check raw text quality page by page ──────────────────────────────
# # We do this first because if PyPDFLoader extracted garbled text,
# # nothing else matters — the embeddings will be garbage too.

# from langchain_community.document_loaders import PyPDFLoader

# print("=" * 60)
# print("RAW TEXT EXTRACTION CHECK")
# print("=" * 60)

# loader = PyPDFLoader("../data/EconomicDevelopmentinCambodia.pdf")
# raw_pages = loader.load()

# for i, page in enumerate(raw_pages):
#     print(f"\n--- Page {i+1} ---")
#     print(f"Character count: {len(page.page_content)}")
#     # repr() shows hidden characters like \n, \t, and garbled symbols
#     # We only show the first 300 chars to keep output readable
#     print(f"First 300 chars: {repr(page.page_content[:300])}")

# # ── Step 3: Test retrieval with different query phrasings ────────────────────
# # We test multiple phrasings because embedding similarity depends heavily
# # on how close your query vocabulary is to the document vocabulary.

# print("\n")
# print("=" * 60)
# print("RETRIEVAL TEST")
# print("=" * 60)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# test_queries = [
#     "What are GDP and growth rate in 2013 and 10 years later?",  # original query
#     "GDP growth rate 2013",                                       # simplified
#     "Cambodia economic growth",                                   # broader
#     "GDP",                                                        # single keyword
# ]

# for query in test_queries:
#     print(f"\nQuery: '{query}'")
#     print("-" * 40)
    
#     docs = retriever.invoke(query)
    
#     if not docs:
#         print("  ⚠ Retriever returned 0 documents")
#         continue
    
#     for i, doc in enumerate(docs):
#         page = doc.metadata.get("page", "?")
#         # Show page number and first 150 chars of each retrieved chunk
#         print(f"  Chunk {i+1} | Page {page} | {doc.page_content[:150]}...")