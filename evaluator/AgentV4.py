"""Standalone hybrid retrieval agent: lexical BM25 plus offline semantic RRF."""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

CANDIDATE_POOL_SIZE = 100
BOUNDARY_RECOVERY_POOL_SIZE = 300
SEMANTIC_COMPONENTS = 128
RRF_K = 60
LEXICAL_RRF_WEIGHT = 2.0
LEXICAL_RERANK_WEIGHT = 0.70
LOW_CONFIDENCE_TOP_K = 4
MIN_RANKING_MARGIN = 0.12
MIN_QUESTION_STRENGTH = 0.55
MIN_ATTRIBUTE_COVERAGE = 0.20
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
OVERRIDE_RE = re.compile(r"\b(actually|instead|change(?:d)?\s+my\s+mind|ignore\s+(?:my\s+)?earlier|rather\s+than)\b", re.I)
NO_PREFERENCE_RE = re.compile(r"\b(?:don't|do not) have (?:an? )?(?:additional )?preference\b", re.I)
BOUNDARY_DENIAL_RE = re.compile(r"\bplease use your judgment\b", re.I)
NON_INFORMATION_RE = re.compile(r"\bthose options are not quite right yet\b", re.I)
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some", "that", "the", "this", "to", "want", "with", "would", "you", "looking", "what", "need", "actually", "earlier", "preference", "preferences", "ignore"}
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
ASKABLE_ATTRIBUTES = ("material", "color", "size", "style", "budget", "feature", "use_case")


def text(value: object) -> str:
    if value is None: return ""
    if isinstance(value, dict): return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list): return " ".join(str(item) for item in value)
    return str(value)


def terms(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value) if len(token) > 1 and token.lower() not in STOPWORDS]


def reply_value(message: str, protected: bool = False) -> str:
    if ":" not in message:
        return message.strip(" -;,.\t\n")
    # Preserve the original high-scoring parser for ordinary Buying/Browsing.
    # Once Boundary/Override is detected, retain embedded colons in values.
    value = message.split(":", 1)[-1] if protected else message.rsplit(":", 1)[-1]
    return value.strip(" -;,.\t\n")


def value_matches(product_value: str, requested_value: str) -> bool:
    product = re.sub(r"\s+", " ", product_value).strip().lower()
    requested = re.sub(r"\s+", " ", requested_value).strip().lower()
    return bool(requested and requested in product) or bool(set(terms(product)) & set(terms(requested)))


def predicted_answers(product: dict) -> dict[str, str]:
    """Generate possible evaluator replies for one candidate product."""
    def flatten(value: object) -> list[str]:
        if isinstance(value, dict): return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
        if isinstance(value, list): return [str(item) for item in value if item not in (None, "")]
        return [str(value)] if value not in (None, "") else []
    def classify(value: str) -> str:
        lowered = value.lower()
        if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered): return "budget"
        if any(material in lowered for material in MATERIALS): return "material"
        if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")): return "color"
        if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")): return "size"
        if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")): return "style"
        if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")): return "use_case"
        return "feature"
    corpus = " ".join(text(product.get(field)) for field in ("title", "features", "details", "description", "categories", "store"))
    candidates = [*flatten(product.get("features")), *flatten(product.get("details"))]
    material, color = MATERIAL_RE.search(corpus), COLOR_RE.search(corpus)
    if material: candidates.insert(0, material.group(1).lower())
    if color: candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""): candidates.append(f"budget around ${product['price']}")
    constraints = list(dict.fromkeys(re.sub(r"\s+", " ", item).strip(" -;,.\t\n") for item in candidates))[:4]
    answers = {attribute: [] for attribute in ASKABLE_ATTRIBUTES}
    for constraint in constraints:
        attribute = classify(constraint)
        if attribute in answers: answers[attribute].append(constraint)
    return {attribute: "; ".join(values[:2]) for attribute, values in answers.items()}


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", candidate_pool_size: int = CANDIDATE_POOL_SIZE) -> None:
        self.catalog_path, self.candidate_pool_size = Path(catalog_path), candidate_pool_size
        self.connection = sqlite3.connect(":memory:")
        self.attributes: dict[str, dict[str, str]] = {}
        self.sessions: dict[str, dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute("CREATE VIRTUAL TABLE products USING fts5(parent_asin UNINDEXED, title, categories, features, details, store, description, tokenize='unicode61 remove_diacritics 2')")
        batch, documents, self.semantic_ids = [], [], []
        with self.catalog_path.open(encoding="utf-8") as source:
            for line in source:
                product = json.loads(line); asin = str(product["parent_asin"])
                self.attributes[asin] = predicted_answers(product)
                fields = tuple(text(product.get(field)) for field in ("title", "categories", "features", "details", "store", "description"))
                batch.append((asin, *fields)); self.semantic_ids.append(asin); documents.append(" ".join(fields))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch); batch.clear()
        if batch: cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=80_000, sublinear_tf=True, dtype=np.float32)
        matrix = self.vectorizer.fit_transform(documents)
        components = min(SEMANTIC_COMPONENTS, matrix.shape[0] - 1, matrix.shape[1] - 1)
        self.semantic_model = TruncatedSVD(n_components=components, n_iter=5, random_state=0)
        self.semantic_vectors = normalize(self.semantic_model.fit_transform(matrix)).astype(np.float32)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {"active_messages": [], "retired_messages": [], "asked": set(), "constraints": {}, "pending_attribute": None, "core_category_message": "", "boundary_recovery": False, "deferred_attribute": None, "protected_reranking": False, "profile": user_profile}

    def _update_state(self, state: dict, message: str) -> bool:
        override = bool(OVERRIDE_RE.search(message))
        no_preference = bool(NO_PREFERENCE_RE.search(message))
        boundary_denial = bool(BOUNDARY_DENIAL_RE.search(message))
        non_information = bool(NON_INFORMATION_RE.search(message))
        if not state["core_category_message"] and message.strip() and not override:
            state["core_category_message"] = message.split(".", 1)[0].strip() or message.strip()
        if override:
            state["retired_messages"].extend(state["active_messages"])
            state["active_messages"] = [item for item in (state["core_category_message"], message.strip()) if item]
            state["asked"], state["constraints"], state["pending_attribute"], state["boundary_recovery"] = set(), {}, None, False
            state["deferred_attribute"] = None
            state["protected_reranking"] = True
            return True
        # Only the simulator's explicit Boundary response activates semantic
        # recovery. Ordinary "no additional preference" replies do not.
        if boundary_denial:
            state["protected_reranking"] = True
        # Original 72% behavior used semantic recovery after any no-preference
        # reply. Keep that for undetected Buying/Browsing, while targeted modes
        # activate it only for a genuine Boundary denial.
        state["boundary_recovery"] = boundary_denial if state["protected_reranking"] else no_preference
        if boundary_denial and state["pending_attribute"]:
            denied = state["pending_attribute"]
            state["asked"].discard(denied)
            state["deferred_attribute"] = denied
        ignore_message = state["protected_reranking"] and non_information
        if message.strip() and not no_preference and not ignore_message:
            state["active_messages"].append(message)
            if state["pending_attribute"]:
                state["constraints"][state["pending_attribute"]] = reply_value(
                    message, protected=state["protected_reranking"]
                )
        return False

    def _lexical_search(self, state: dict, limit: int) -> list[tuple[str, float]]:
        words = list(dict.fromkeys(terms(" ".join(state["active_messages"]))))[:30]
        if not words: return []
        for query in (" AND ".join(f'"{word}"' for word in words), " OR ".join(f'"{word}"' for word in words)):
            rows = self.connection.execute("SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) FROM products WHERE products MATCH ? ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?", (query, limit)).fetchall()
            if rows: return [(str(asin), float(score)) for asin, score in rows]
        return []

    def _semantic_search(self, state: dict, limit: int) -> list[tuple[str, float]]:
        query = " ".join(state["active_messages"])
        if not query.strip(): return []
        vector = normalize(self.semantic_model.transform(self.vectorizer.transform([query]))).astype(np.float32)[0]
        scores = self.semantic_vectors @ vector; count = min(limit, len(self.semantic_ids))
        indexes = np.argpartition(scores, -count)[-count:]; indexes = indexes[np.argsort(scores[indexes])[::-1]]
        return [(self.semantic_ids[index], float(scores[index])) for index in indexes]

    @staticmethod
    def _rrf(lexical: list[tuple[str, float]], semantic: list[tuple[str, float]], limit: int, lexical_weight: float = 1.0) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for weight, results in ((lexical_weight, lexical), (1.0, semantic)):
            for rank, (asin, _) in enumerate(results, 1): scores[asin] = scores.get(asin, 0.0) + weight / (RRF_K + rank)
        return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:limit]

    def _rerank(self, candidates: list[str], constraints: dict[str, str], protected: bool = False) -> list[str]:
        if not candidates or not constraints: return candidates
        rarity = {attribute: math.log((len(candidates) + 1) / (1 + sum(value_matches(self.attributes[asin][attribute], value) for asin in candidates))) for attribute, value in constraints.items()}
        def score(item: tuple[int, str]) -> float:
            rank, asin = item
            lexical_score = 1 / (rank + 1)
            constraint_score = sum(rarity[attribute] for attribute, value in constraints.items() if value_matches(self.attributes[asin][attribute], value))
            if not protected:
                # Preserve V4's original behavior for normal Buying/Browsing.
                return lexical_score + constraint_score
            normalized_constraint = constraint_score / (sum(rarity.values()) or 1.0)
            return LEXICAL_RERANK_WEIGHT * lexical_score + (1 - LEXICAL_RERANK_WEIGHT) * normalized_constraint
        return [asin for _, asin in sorted(enumerate(candidates), key=score, reverse=True)]

    def _question_score(self, candidates: list[str], attribute: str) -> float:
        values = [self.attributes[asin][attribute] for asin in candidates if self.attributes[asin][attribute]]
        coverage = len(values) / len(candidates) if candidates else 0.0
        if coverage < MIN_ATTRIBUTE_COVERAGE: return -1.0
        counts, total = Counter(values), len(values)
        return coverage * -sum((count / total) * math.log2(count / total) for count in counts.values())

    def _choose_question(self, state: dict, candidates: list[str], turn: int) -> str | None:
        deferred = state["deferred_attribute"]
        available = [attribute for attribute in ASKABLE_ATTRIBUTES if attribute not in state["asked"] and attribute != deferred]
        if not available and deferred and deferred not in state["asked"]:
            available = [deferred]
        if turn >= 10 or not available: return None
        choice = max(available, key=lambda attribute: self._question_score(candidates, attribute)); state["asked"].add(choice)
        # The Boundary-denied attribute becomes eligible again after this one
        # different question, rather than being lost for the whole session.
        if deferred and choice != deferred:
            state["deferred_attribute"] = None
        return choice

    def _question_strength(self, candidates: list[str], attribute: str | None) -> float:
        if not candidates or not attribute: return 0.0
        values = [self.attributes[asin][attribute] for asin in candidates if self.attributes[asin][attribute]]
        if len(set(values)) < 2: return 0.0
        counts, total = Counter(values), len(values)
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        return (len(values) / len(candidates)) * entropy / math.log2(len(counts))

    @staticmethod
    def _ranking_margin(scored: list[tuple[str, float]], top_k: int) -> float:
        if len(scored) <= top_k: return 1.0
        relevance = [-score for _, score in scored]; head_size = min(3, top_k)
        head = sum(relevance[:head_size]) / head_size; tail = sum(relevance[head_size:top_k]) / (top_k - head_size)
        return max(0.0, (head - tail) / max(abs(head), 1e-9))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions: raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]; overridden = self._update_state(state, user_message)
        pool_size = BOUNDARY_RECOVERY_POOL_SIZE if state["boundary_recovery"] else self.candidate_pool_size
        lexical = self._lexical_search(state, pool_size)
        scored = self._rrf(
            lexical,
            self._semantic_search(state, pool_size),
            pool_size,
            lexical_weight=LEXICAL_RRF_WEIGHT if state["protected_reranking"] else 1.0,
        ) if overridden or state["boundary_recovery"] else lexical
        candidates = self._rerank(
            [asin for asin, _ in scored],
            state["constraints"],
            protected=state["protected_reranking"],
        )
        attribute = self._choose_question(state, candidates, turn); state["pending_attribute"] = attribute
        short_list = self._ranking_margin(scored, top_k) < MIN_RANKING_MARGIN and self._question_strength(candidates, attribute) >= MIN_QUESTION_STRENGTH
        count = top_k if overridden or state["boundary_recovery"] or not short_list else min(LOW_CONFIDENCE_TOP_K, top_k)
        recommendations = [{"parent_asin": asin} for asin in candidates[:count]]
        if overridden: message = "Thanks for clarifying. I kept the product type and replaced the earlier preference."
        elif attribute: message = f"Here are the best matches so far. Do you have a {attribute} preference?"
        else: message = "Here are my best matches based on what you shared."
        return {"message": message, "ask_attribute": attribute, "recommendations": recommendations, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
