# app/api/routes/chat.py
# Handles conversational Q&A with the research agent.

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    # Chat history as list of {"role": "human"/"ai", "content": "..."}
    # The client is responsible for maintaining and sending history.
    # This is stateless on the server side — simpler and more scalable
    # than storing sessions on the server.
    history: list[dict] = []

class ChatResponse(BaseModel):
    answer: str
    question: str

def parse_history(history: list[dict]) -> list:
    """
    Converts the client's history format into LangChain message objects.

    Client sends:  [{"role": "human", "content": "..."}, ...]
    LangChain needs: [HumanMessage(...), AIMessage(...), ...]
    """
    messages = []
    for item in history:
        if item["role"] == "human":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "ai":
            messages.append(AIMessage(content=item["content"]))
    return messages

@router.post("", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """
    Send a message and receive an answer from the research agent.

    The agent automatically decides whether to:
    - Answer from document context
    - Use calculator or other tools
    - Search the web for information not in documents
    """
    agent = request.app.state.agent

    if agent is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Use POST /ingest first."
        )

    if not body.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # Convert history from client format to LangChain messages
    chat_history = parse_history(body.history)

    # Import here to avoid circular imports
    from app.core.agent import run_agent

    answer = run_agent(agent, body.message, chat_history)

    return ChatResponse(
        answer=answer,
        question=body.message
    )