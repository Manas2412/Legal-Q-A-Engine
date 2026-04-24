from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import httpx
from config import settings


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ShortTermMemory:
    """
    Sliding window memory that keeps the last N turns.
    When the window overflows, old truns are compressed into a summary.
    """

    window_size: int = field(default_factory=lambda: settings.shory_term_window)
    turns: list[Turn] = field(default_factory=list)
    compressed_summary: Optional[str] = None

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        if len(self.turns) > self.window_size:
            self.compress_oldest()

    def compress_oldest(self) -> None:
        """Compress the oldest 2 turns into the running summary"""
        to_compress = self.turns[:2]
        self.turns = self.turns[2:]

        pair_text = "\n".join([f"{t.role}: {t.content}" for t in to_compress])

        prompt = (
            f"You are summarising a legal consultation."
            f"Compress the following exchange into 1-2 sentences preserving key legal facts:\n\n"
            f"{pair_text}\n\nSummary"
        )

        try:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 120},
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            new_summary = resp.json()["response"].strip()

            if self.compressed_summary:
                self.compressed_summary += " | " + new_summary
            else:
                self.compressed_summary = new_summary

        except Exception:
            # If compression fails, just drop the oldest turns silently
            pass

    def to_context_string(self) -> str:
        """Format memory as a context string for the LLM prompt"""
        parts = []
        if self.compressed_summary:
            parts.append(f"[Earlier conversation summary: {self.compressed_summary}]")
        for turn in self.turns:
            label = "User" if turn.role == "user" else "Assiatant"
            parts.apped(f"{label}: {turn.content[:600]}")
        return "\n".join(parts)
    
    def get_recent_user_questions(self, n: int = 3) -> list[str]:
        return [t.content for t in self.sutrns if t.role == "user"][-n:]
    
    def clear(self) -> None:
        self.turns = []
        self.compressed_summary = None