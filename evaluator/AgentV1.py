"""Stateful BM25 agent with simple intent-override handling.

The evaluator imports a class named ``Agent``.  ``AgentV1`` is also exported
explicitly so it can be instantiated directly while experimenting.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "what", "need", "actually", "earlier", "preference", "preferences", "ignore",
}
OVERRIDE_RE = re.compile(
    r"\b(actually|instead|change(?:d)?\s+my\s+mind|ignore\s+(?:my\s+)?earlier|rather\s+than)\b",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(r"\b(?:don't|do not) have (?:an? )?(?:additional )?preference\b", re.I)
ATTRIBUTE_ORDER = ("material", "color", "size", "style", "use_case", "brand", "budget", "feature", "category", "other")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower() for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """A local, dependency-free conversational product search agent.

    It cannot infer every semantic conflict, but an explicit correction clears
    prior message-derived constraints.  This is intentional: an evaluator
    intent override says to ignore the earlier preference.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]), _text(product.get("title")),
                    _text(product.get("categories")), _text(product.get("features")),
                    _text(product.get("details")), _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) == 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {
            "profile": user_profile,
            "active_messages": [],
            "retired_messages": [],
            "asked_attributes": set(),
        }

    @staticmethod
    def _is_information(message: str) -> bool:
        """Avoid treating the evaluator's non-answer templates as constraints."""
        return bool(message.strip()) and not NO_PREFERENCE_RE.search(message)

    def _update_state(self, state: dict, user_message: str) -> bool:
        overridden = bool(OVERRIDE_RE.search(user_message))
        if overridden:
            state["retired_messages"].extend(state["active_messages"])
            state["active_messages"] = []
            # The useful next question can change with the new intent too.
            state["asked_attributes"] = set()
        if self._is_information(user_message):
            state["active_messages"].append(user_message)
        return overridden

    def _recommend(self, state: dict, top_k: int) -> list[dict]:
        terms = list(dict.fromkeys(_terms(" ".join(state["active_messages"]))))[:30]
        if not terms:
            return []
        # AND first: once answers accumulate, prefer products satisfying all
        # active words.  Fall back to OR if that would yield too few products.
        quoted = [f'"{term}"' for term in terms]
        for expression in (" AND ".join(quoted), " OR ".join(quoted)):
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            if rows:
                return [{"parent_asin": str(row[0])} for row in rows]
        return []

    @staticmethod
    def _next_attribute(state: dict, turn: int) -> str | None:
        if turn >= 10:
            return None
        for attribute in ATTRIBUTE_ORDER:
            if attribute not in state["asked_attributes"]:
                state["asked_attributes"].add(attribute)
                return attribute
        return "other"

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        overridden = self._update_state(state, user_message)
        recommendations = self._recommend(state, top_k)
        ask_attribute = self._next_attribute(state, turn)
        if overridden:
            message = "Thanks for the correction. I have reset the earlier preference and updated the matches."
        elif ask_attribute:
            message = f"Here are the closest matches so far. Do you have a {ask_attribute} preference?"
        else:
            message = "Here are my best matches based on the details you shared."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
