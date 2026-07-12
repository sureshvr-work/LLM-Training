"""
memory.py — SHORT-TERM memory = the running list of messages.

This IS the "context window" you watch grow in the UI. Every turn we append to
it, and every turn the whole thing is re-sent to the model. The model keeps
nothing between calls — this list is its entire memory of the session.

Long-term memory (durable history / fraud patterns in a database) comes later.
The interface here is kept tiny so a Postgres- or Redis-backed version can drop
in without changing any caller.
"""
from schema import Message


class ShortTermMemory:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        """Append one message (this turn's contribution to the context)."""
        self._messages.append(message)

    def history(self) -> list[Message]:
        """The full context, in order — exactly what gets re-sent each turn."""
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
