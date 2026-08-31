"""V2 with confidence-aware recommendation exposure.

V3 never withholds recommendations entirely.  When the BM25 scores form a
weakly separated ranking *and* the next question is informative, it exposes
only the strongest few candidates.  This avoids prematurely ending a session
with the hidden target in a low top-10 position.
"""

from __future__ import annotations

import math
from collections import Counter

try:  # Works both as ``evaluator.AgentV3`` and from the evaluator folder.
    from .AgentV2 import Agent as AgentV2
    from .AgentV2 import ASKABLE_ATTRIBUTES, terms
except ImportError:
    from AgentV2 import Agent as AgentV2
    from AgentV2 import ASKABLE_ATTRIBUTES, terms


# Tune these against the public development set.  They are decision thresholds,
# not importance weights assigned to individual attributes.
LOW_CONFIDENCE_TOP_K = 4
MIN_RANKING_MARGIN = 0.12
MIN_QUESTION_STRENGTH = 0.55


class AgentV3(AgentV2):
    def _search_with_scores(self, state: dict, limit: int) -> list[tuple[str, float]]:
        query_terms = list(dict.fromkeys(terms(" ".join(state["active_messages"]))))[:30]
        if not query_terms:
            return []
        quoted = [f'"{term}"' for term in query_terms]
        for query in (" AND ".join(quoted), " OR ".join(quoted)):
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
                "FROM products WHERE products MATCH ? ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (query, limit),
            ).fetchall()
            if rows:
                return [(str(asin), float(score)) for asin, score in rows]
        return []

    def _question_strength(self, candidates: list[str], attribute: str | None) -> float:
        """Return normalized coverage × entropy, in the range 0–1."""
        if not attribute or not candidates:
            return 0.0
        values = [self.attributes[asin][attribute] for asin in candidates if self.attributes[asin][attribute]]
        coverage = len(values) / len(candidates)
        if len(set(values)) < 2:
            return 0.0
        counts = Counter(values)
        total = len(values)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        max_entropy = math.log2(len(counts))
        return coverage * (entropy / max_entropy)

    def _choose_question(self, state: dict, candidates: list[str], turn: int) -> str | None:
        """Do not fall through to V2's unmodelled ``other`` attribute."""
        if turn >= 10:
            return None
        available = [attribute for attribute in ASKABLE_ATTRIBUTES if attribute not in state["asked"]]
        if not available:
            return None
        choice = max(available, key=lambda attribute: self._question_score(candidates, attribute))
        state["asked"].add(choice)
        return choice

    @staticmethod
    def _ranking_margin(scored_candidates: list[tuple[str, float]], top_k: int) -> float:
        """Measure how distinct the leading products are from the shown tail."""
        if len(scored_candidates) <= top_k:
            return 1.0  # No extra tail candidates exist to defer.
        relevance = [-score for _, score in scored_candidates]
        head_size = min(3, top_k)
        head = sum(relevance[:head_size]) / head_size
        tail = sum(relevance[head_size:top_k]) / (top_k - head_size)
        return max(0.0, (head - tail) / max(abs(head), 1e-9))

    def _recommendation_count(
        self,
        scored_candidates: list[tuple[str, float]],
        candidates: list[str],
        attribute: str | None,
        turn: int,
        top_k: int,
    ) -> int:
        if turn >= 10 or not attribute:
            return top_k
        margin = self._ranking_margin(scored_candidates, top_k)
        information = self._question_strength(candidates, attribute)
        if margin < MIN_RANKING_MARGIN and information >= MIN_QUESTION_STRENGTH:
            return min(LOW_CONFIDENCE_TOP_K, top_k)
        return top_k

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        overridden = self._update_state(state, user_message)

        scored = self._search_with_scores(state, self.candidate_pool_size)
        bm25_order = [asin for asin, _ in scored]
        candidate_ids = self._rerank(bm25_order, state["constraints"])
        attribute = self._choose_question(state, candidate_ids, turn)
        state["pending_attribute"] = attribute

        count = self._recommendation_count(scored, candidate_ids, attribute, turn, top_k)
        recommendations = [{"parent_asin": asin} for asin in candidate_ids[:count]]
        if overridden:
            message = "Thanks for clarifying. I have replaced the earlier preference and updated the matches."
        elif attribute:
            message = f"Here are the best matches so far. Do you have a {attribute} preference?"
        else:
            message = "Here are my best matches based on what you shared."
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


Agent = AgentV3
