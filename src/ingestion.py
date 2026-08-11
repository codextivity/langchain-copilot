import hashlib
from pathlib import Path
# Third-party integrations (PFD loaders, web scrapers, etc)
from langchain_community.document_loaders import PyPDFLoader
# standalone package for splitting logic
from langchain_text_splitters import RecursiveCharacterTextSplitter
# OpenAI-specific wrappers for embeddings and LLMs
from langchain_openai import OpenAIEmbeddings
# ChromaDB integration (a local vector database)
from langchain_chroma import Chroma

CHROMA_PATH = "chroma_db"

def get_file_hash(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    

"""
PyPDfloader does:
 It reads a PDF file and extracts text from each page, returning a list of Document objects (one per page).
 - page_content: the raw text of that page
 - metadata: a dict containing the page number and source file path

RecursiveCharacterTextSplitter does:
 It takes a list of Document objects and splits them into smaller chunks based on character count and overlap settings.
"""
def load_and_split(pdf_path: str) -> list:
    """Load PDF and split into chunks."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # List[Document], one per page
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      # characters per chunk (not tokens, not words). 1K characters ≈ 200 tokens
        chunk_overlap=400,    # overlap to avoid cutting context mid-sentence
        separators=["\n\n", "\n", ".", " "]  # priority order for splitting
    )
    
    chunks = splitter.split_documents(docs) # List[Document], each chunk has page_content and metadata

    # Add the file hash to every chunk's metadata.
    # This lets us check later whether this exact file
    # has already been ingested
    file_hash = get_file_hash(pdf_path)
    for chunk in chunks:
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["file_name"] = Path(pdf_path).name  # store the original file name for reference
    print(f"Loaded {len(docs)} pages → split into {len(chunks)} chunks")
    return chunks


def is_document_already_ingested(vectorstore: Chroma, file_hash: str) -> bool:
    """
    Checks if a document with this hash already exists in the vector store.

    How it works:
    ChromaDB lets you filter by metadata when querying.
    We search for any chunk that has this file_hash in its metadata.
    If we find even one, the whole document is already there.
    """
    results = vectorstore.get(
        where={"file_hash": {"$eq": file_hash}},
        limit=1  # we only need to find one chunk to confirm it exists
    )

    # results["ids"] is a list of matching chunk IDs
    # If the list is non-empty, the document exists
    return len(results["ids"]) > 0


"""
What an embedding is:
 - An embedding is a vector representation of text in a high-dimensional space.
 aN embedding model converts text into a list of numbers (a vector) for example [0.021, -0.134, 0.987, ...] 
 with hundreds of dimensions.

 - The idea is that semantically similar texts will have embeddings that are close together in this vector space.
 So, "revenue declined" and "sales dropped" will have embeddings that are closer together than 
 "revenue declined" and "the sky is blue".

 - Why text-embedding-3-small? It's a smaller, faster, and cheaper model than text-embedding-3-large,
 but still good enough for many retrieval tasks. If you need higher accuracy and don't mind the cost, 
 you can switch to text-embedding-3-large.

"""
def build_vectorstore(chunks: list) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    """
     Chroma.from_documents():
     1. Takes a list of Document objects and computes embeddings for each chunk.
     2. It then stores these embeddings in a local ChromaDB database at the specified directory.

    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH 
        # The persist_directory: Where the vectorstore will be saved on disk. 
        #Without this, it would only exist in memory and be lost when the program exits.
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def load_existing_vectorstore() -> Chroma:
    """Load an already-built vector store."""
    """
    Why embeddings are needed to load an existing vectorstore:
    Even when loading an existing vectorstore, the embedding function is required because: 
    when you later query the vectorstore, it needs to convert your query into an embedding to compare against 
    the stored embeddings.So, the embedding function is not just for building the vectorstore; it's also needed 
    for querying it.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def ingest(pdf_path: str) -> Chroma:
    """
    Main ingestion function — idempotent.

    If the vector store does not exist yet: create it.
    If it exists but this document is new: add the new document.
    If it exists and this document is already there: skip it.

    This means you can safely run ingest() on the same file
    multiple times without creating duplicates.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    file_hash = get_file_hash(pdf_path)
    file_name = Path(pdf_path).name

    # Case 1: No vector store exists yet — create from scratch
    if not Path(CHROMA_PATH).exists():
        print(f"Creating new vector store...")
        chunks = load_and_split(pdf_path)
        return build_vectorstore(chunks)


    # Case 2: Vector store exists — load it and check for duplicates
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    if is_document_already_ingested(vectorstore, file_hash):
    # Document already exists — skip ingestion entirely
        print(f"'{file_name}' is already in the vector store. Skipping.")
        return vectorstore

    # Case 3: Vector store exists but this document is new — add it
    print(f"Adding '{file_name}' to existing vector store...")
    chunks = load_and_split(pdf_path)

    # .add_documents() appends to the existing store without touching
    # the chunks that are already there
    vectorstore.add_documents(chunks)
    print(f"Added {len(chunks)} new chunks. Vector store now contains multiple documents.")

    return vectorstore
