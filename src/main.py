import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ingestion import ingest, load_existing_vectorstore
from chain import build_rag_chain

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
    
    # Step 2: Build chain
    chain = build_rag_chain(vectorstore)
    
    # Step 3: Chat loop
    print("\nReady. Ask questions about your document. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
            
        print("\nAssistant: ", end="", flush=True)
        
        # Stream the response token by token
        """
        Why we use chain.stream() instead of just calling chain.invoke()?
        chain.invoke() would return the entire answer at once, which can be slow for long responses.
        chain.stream() allows us to get the answer token by token, so we can print it as it comes in,
        giving a more interactive feel.

        The flush=True argument ensures that each token is printed immediately, rather than being buffered.
        end="" prevents adding a newline after each token, so the answer appears on the same line.

        The "chunk" variable represents each piece of the answer as it is generated. We print it immediately to the console.
        """
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        
        print("\n")

if __name__ == "__main__":
    main()
