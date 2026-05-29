"""
SENTINEL — Session Context Manager (Multi-Turn Clinical Continuity)

Phase 5 implementation.
Maintains an in-memory ring buffer of the last 3 Q&A turns per session_id (Finding #45).
Includes a 30-minute TTL (Time-To-Live) to auto-clear session history on inactivity.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Final

from sentinel.config import SESSION_CONTEXT_TURNS, SESSION_TTL_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    query: str
    answer: str
    condition_codes: list[str]
    timestamp: float = field(default_factory=time.time)


class SessionContext:
    """Represents a single active clinical consultation session."""
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.turns: list[ConversationTurn] = []
        self.last_accessed: float = time.time()

    def add_turn(self, query: str, answer: str, condition_codes: list[str]) -> None:
        self.turns.append(ConversationTurn(query, answer, condition_codes))
        # Keep only the last N turns (ring buffer)
        self.turns = self.turns[-SESSION_CONTEXT_TURNS:]
        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > SESSION_TTL_SECONDS

    def get_history_summary(self) -> str:
        """Formats the last turns for insertion into the LLM system prompt."""
        self.last_accessed = time.time()
        if not self.turns:
            return ""

        summary_lines = ["\n[PREVIOUS DISCUSSION CONTEXT]"]
        for idx, turn in enumerate(self.turns):
            summary_lines.append(f"Turn {idx + 1}:")
            summary_lines.append(f"  Clinician: {turn.query}")
            # Truncate answer for context window size control
            truncated_ans = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer
            summary_lines.append(f"  SENTINEL: {truncated_ans}")
        summary_lines.append("[END OF DISCUSSION CONTEXT]\n")
        return "\n".join(summary_lines)


class SessionContextManager:
    """Manages active session instances in-memory."""
    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}
        self._lock = threading.Lock()

    def get_session(self, session_id: str) -> SessionContext:
        """Retrieves or creates a session context."""
        import threading
        with self._lock:
            # First, clean expired sessions
            self.clean_expired_sessions()
            
            if session_id not in self._sessions:
                logger.info(f"Creating new session context for session_id '{session_id}'")
                self._sessions[session_id] = SessionContext(session_id)
            return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Explicitly deletes a session's history."""
        import threading
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Cleared session context for session_id '{session_id}'")

    def clean_expired_sessions(self) -> None:
        """Garbage collects expired sessions to release memory."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
            logger.info(f"Garbage collected expired session '{sid}' (TTL exceeded).")


# Singleton instance
session_manager = SessionContextManager()
import threading  # Ensure imported in module namespace
