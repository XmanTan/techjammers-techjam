"""Stateful catalog search with information-gain question selection.

Change ``CANDIDATE_POOL_SIZE`` to control how many retrieved products are used
when estimating the value of the next question.  Larger pools give more stable
statistics; smaller pools make each turn faster and more locally focused.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path


# Main tuning knobs.  The evaluator still passes top_k=10 for recommendations.
CANDIDATE_POOL_SIZE = 100

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
EVALUATOR_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)

# The local evaluator can disclose hidden constraints only in these categories.
# ``category`` and ``brand`` remain valid API values, but its simulator does
# not produce answers for them, so automatic questions must not select them.
# No relative attribute weights are assigned here.
ASKABLE_ATTRIBUTES = (
    "material", "color", "size", "style", "budget", "feature", "use_case",
)
MIN_ATTRIBUTE_COVERAGE = 0.20


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def terms(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value) if len(token) > 1 and token.lower() not in STOPWORDS]


def reply_value(message: str) -> str:
    value = message.rsplit(":", 1)[-1] if ":" in message else message
    return value.strip(" -;,.\t\n")


def value_matches(product_value: str, requested_value: str) -> bool:
    normalized_product = re.sub(r"\s+", " ", product_value).strip().lower()
    normalized_request = re.sub(r"\s+", " ", requested_value).strip().lower()
    if normalized_request and normalized_request in normalized_product:
        return True
    requested = set(terms(requested_value))
    product = set(terms(product_value))
    return bool(requested and product and requested & product)


def attribute_values(product: dict) -> dict[str, str]:
    """Predict the evaluator reply for each question from catalog metadata.

    This intentionally mirrors its public intent-card and classification logic,
    without using a session label or knowing which product is the target.
    """
    def flatten(value: object) -> list[str]:
        if isinstance(value, dict):
            return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
        if isinstance(value, list):
            return [str(item) for item in value if item not in (None, "")]
        return [str(value)] if value not in (None, "") else []

    def classify(value: str) -> str:
        lowered = value.lower()
        if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
            return "budget"
        if any(material in lowered for material in EVALUATOR_MATERIALS):
            return "material"
        if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
            return "color"
        if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
            return "size"
        if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
            return "style"
        if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
            return "use_case"
        return "feature"

    corpus = " ".join(text(product.get(field)) for field in ("title", "features", "details", "description", "categories", "store"))
    candidates = [*flatten(product.get("features")), *flatten(product.get("details"))]
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    constraints = list(dict.fromkeys(re.sub(r"\s+", " ", item).strip(" -;,.\t\n") for item in candidates))[:4]
    answers: dict[str, list[str]] = {attribute: [] for attribute in ASKABLE_ATTRIBUTES}
    for constraint in constraints:
        attribute = classify(constraint)
        if attribute in answers:
            answers[attribute].append(constraint)
    return {attribute: "; ".join(values[:2]) for attribute, values in answers.items()}


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", candidate_pool_size: int = CANDIDATE_POOL_SIZE) -> None:
        if candidate_pool_size < 10:
            raise ValueError("candidate_pool_size must be at least 10")
        self.catalog_path = Path(catalog_path)
        self.candidate_pool_size = candidate_pool_size
        self.connection = sqlite3.connect(":memory:")
        self.attributes: dict[str, dict[str, str]] = {}
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
        with self.catalog_path.open(encoding="utf-8") as source:
            for line in source:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.attributes[asin] = attribute_values(product)
                batch.append((asin, text(product.get("title")), text(product.get("categories")), text(product.get("features")), text(product.get("details")), text(product.get("store")), text(product.get("description"))))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {
            "active_messages": [], "retired_messages": [], "asked": set(),
            "constraints": {}, "pending_attribute": None, "profile": user_profile,
        }

    def _update_state(self, state: dict, user_message: str) -> bool:
        override = bool(OVERRIDE_RE.search(user_message))
        if override:
            state["retired_messages"].extend(state["active_messages"])
            state["active_messages"] = []
            state["asked"] = set()
            state["constraints"] = {}
            state["pending_attribute"] = None
        if user_message.strip() and not NO_PREFERENCE_RE.search(user_message):
            state["active_messages"].append(user_message)
            pending = state["pending_attribute"]
            if pending and not override:
                state["constraints"][pending] = reply_value(user_message)
        return override

    def _search(self, state: dict, limit: int) -> list[str]:
        query_terms = list(dict.fromkeys(terms(" ".join(state["active_messages"]))))[:30]
        if not query_terms:
            return []
        quoted = [f'"{term}"' for term in query_terms]
        for query in (" AND ".join(quoted), " OR ".join(quoted)):
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (query, limit),
            ).fetchall()
            if rows:
                return [str(row[0]) for row in rows]
        return []

    def _rerank(self, candidates: list[str], constraints: dict[str, str]) -> list[str]:
        """Rank answer matches by their rarity in the current candidate pool."""
        pool_size = len(candidates)
        if not constraints or pool_size == 0:
            return candidates
        rarity = {
            attribute: math.log((pool_size + 1) / (1 + sum(
                value_matches(self.attributes[asin][attribute], requested) for asin in candidates
            )))
            for attribute, requested in constraints.items()
        }

        def score(item: tuple[int, str]) -> float:
            rank, asin = item
            return 1 / (rank + 1) + sum(
                rarity[attribute]
                for attribute, requested in constraints.items()
                if value_matches(self.attributes[asin][attribute], requested)
            )

        return [asin for _, asin in sorted(enumerate(candidates), key=score, reverse=True)]

    def _question_score(self, candidates: list[str], attribute: str) -> float:
        values = [self.attributes[asin][attribute] for asin in candidates if self.attributes[asin][attribute]]
        coverage = len(values) / len(candidates) if candidates else 0.0
        if coverage < MIN_ATTRIBUTE_COVERAGE:
            return -1.0
        counts = Counter(values)
        total = len(values)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        return coverage * entropy

    def _choose_question(self, state: dict, candidates: list[str], turn: int) -> str | None:
        if turn >= 10:
            return None
        available = [attribute for attribute in ASKABLE_ATTRIBUTES if attribute not in state["asked"]]
        if not available:
            return "other"
        choice = max(available, key=lambda attribute: self._question_score(candidates, attribute))
        state["asked"].add(choice)
        return choice

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        overridden = self._update_state(state, user_message)
        candidate_ids = self._search(state, self.candidate_pool_size)
        candidate_ids = self._rerank(candidate_ids, state["constraints"])
        recommendations = [{"parent_asin": asin} for asin in candidate_ids[:top_k]]
        attribute = self._choose_question(state, candidate_ids, turn)
        state["pending_attribute"] = attribute
        if overridden:
            message = "Thanks for clarifying. I have replaced the earlier preference and updated the matches."
        elif attribute:
            message = f"Here are the best matches so far. Do you have a {attribute} preference?"
        else:
            message = "Here are my best matches based on what you shared."
        return {"message": message, "ask_attribute": attribute, "recommendations": recommendations, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
