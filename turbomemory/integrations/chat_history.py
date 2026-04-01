"""TurboMemory LangChain chat message history integration."""

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory

from turbomemory import TurboMemory


class TurboMemoryChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by TurboMemory.

    Stores conversation turns as TurboMemory session logs and
    optionally extracts memories from the conversation.

    Example:
        from turbomemory.integrations.langchain import TurboMemoryChatMessageHistory

        history = TurboMemoryChatMessageHistory(
            root="my_memory",
            session="chat_session_1",
        )

        history.add_user_message("Hello, what is TurboQuant?")
        history.add_ai_message("TurboQuant is a quantization method...")

        print(history.messages)
    """

    def __init__(
        self,
        root: str = "turbomemory_data",
        session: str = "default",
        extract_memories: bool = False,
        memory_topic: str = "conversation_facts",
        **kwargs: Any,
    ):
        self.tm = TurboMemory(root=root)
        self.session = session
        self.extract_memories = extract_memories
        self.memory_topic = memory_topic
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        """Return the conversation messages."""
        if not self._messages:
            self._load_messages()
        return self._messages

    def _load_messages(self) -> None:
        """Load messages from session logs."""
        import os
        import json

        session_path = os.path.join(self.tm.sessions_dir, self.session + ".jsonl")
        if not os.path.exists(session_path):
            return

        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                role = record.get("role", "")
                text = record.get("text", "")

                if role == "user":
                    self._messages.append(HumanMessage(content=text))
                elif role == "assistant":
                    self._messages.append(AIMessage(content=text))
                elif role == "system":
                    self._messages.append(SystemMessage(content=text))

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history."""
        self._messages.append(message)

        role = "user"
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"

        self.tm.add_turn(role=role, text=message.content, session_file=self.session + ".jsonl")

        if self.extract_memories and isinstance(message, (HumanMessage, AIMessage)):
            if len(message.content) > 20:
                self.tm.add_memory(
                    topic=self.memory_topic,
                    text=message.content,
                    confidence=0.7,
                )

    def add_user_message(self, message: str) -> None:
        """Add a user message."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add an AI message."""
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Clear the chat history."""
        self._messages = []
        import os
        session_path = os.path.join(self.tm.sessions_dir, self.session + ".jsonl")
        if os.path.exists(session_path):
            os.remove(session_path)

    def close(self) -> None:
        """Close TurboMemory connections."""
        self.tm.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
