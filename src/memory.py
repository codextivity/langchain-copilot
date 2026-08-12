
#  ✅ chat history

# src/memory.py
# Manages conversation history for a single chat session.
# Right now this is in-memory (lost when the script ends).
# In Week 6 we will replace this with Redis for persistence.

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class ChatHistory:
    """
    Stores the conversation as a list of alternating HumanMessage
    and AIMessage objects.

    Why a class instead of a plain list?
    Because we want to add helper methods (add, clear, format for display)
    without scattering that logic across main.py.
    """

    def __init__(self):
        # This list holds the entire conversation in order.
        # LangChain expects this exact format when you pass history
        # to a prompt template via MessagesPlaceholder.
        self.messages: list[BaseMessage] = []

    def add_user_message(self, content: str) -> None:
        """Call this BEFORE sending the question to the chain."""
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        """Call this AFTER receiving the full answer from the chain."""
        self.messages.append(AIMessage(content=content))

    def get_messages(self) -> list[BaseMessage]:
        """
        Returns the full history.
        This gets passed directly into the chain as {chat_history}.
        """
        return self.messages

    def clear(self) -> None:
        """Reset the conversation. Useful for starting a new topic."""
        self.messages = []

    def display(self) -> None:
        """Print the conversation so far. Useful for debugging."""
        for msg in self.messages:
            role = "You" if isinstance(msg, HumanMessage) else "AI"
            print(f"{role}: {msg.content[:100]}...")