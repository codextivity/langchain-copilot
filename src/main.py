import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ingestion import ingest, load_existing_vectorstore
from chain import build_conversational_rag_chain
from memory import ChatHistory

CHROMA_PATH = "chroma_db"

def main():
    # Step 1: Ingest or load
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Ingesting {pdf_path}...")
        vectorstore = ingest(pdf_path)
    elif Path(CHROMA_PATH).exists():
        print("Loading existing vector store...")
        vectorstore = load_existing_vectorstore()
    else:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)
    
    # Step 2: Build chain and history
    chain = build_conversational_rag_chain(vectorstore)
    chat_history = ChatHistory()  # Initialize chat history for this session
    
    # Step 3: Chat loop
    print("\nReady. Commands: 'quit' to exit, 'clear' to reset conversation.\n")

    while True:
        question = input("You: ").strip()
        
        if question.lower() in ("quit", "exit", "q"):
            break

        if question.lower() == "clear":
            chat_history.clear()
            print("Conversation history cleared.")
            continue

        if not question:
            continue
            
          # Add the user's question to history BEFORE calling the chain.
        # This ensures the history is complete when the chain reads it.
        chat_history.add_user_message(question)

        # ── Invoke the chain ─────────────────────────────────────────────────
        # We use invoke instead of stream here because create_retrieval_chain
        # returns a dict — streaming dicts requires extra handling we will
        # add in Week 6 when we move to FastAPI.
        #
        # The chain expects exactly these two keys:
        #   "input"        — the current question
        #   "chat_history" — all previous messages
        
        # ── Invoke the chain ─────────────────────────────────────────────────
        # Chain now returns a plain string — no dict unpacking needed
        answer = chain.invoke({
            "input": question,
            "chat_history": chat_history.get_messages()
        })

        print(f"\nAssistant: {answer}\n")

        # Add AI answer to history after receiving it
        chat_history.add_ai_message(answer)

if __name__ == "__main__":
    main()
