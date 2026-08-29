"""Catalog loading, normalisation and retrieval indexes.

Everything here is pure standard library so the agent runs with no network
access and no third-party dependencies, which is the environment the organizer
reserves the right to enforce at final scoring time.

Three indexes are built once at start-up:

1. ``category_members``  -- coarse category label -> product ids.
   The opening customer turn always names the product family they are shopping
   for, so this is the primary candidate filter.

2. ``phrase_index``      -- normalised attribute phrase -> product ids.
   Customers state requirements that correspond closely to catalog attribute
   text ("a key requirement is: 100% cotton"). Phrase matching resolves those
   to a small candidate set directly.

3. ``postings``          -- BM25 inverted index over the product text.
   The robust fallback. If a requirement is paraphrased and never matches a
   phrase exactly, BM25 still scores it sensibly.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "im", "just", "still", "have", "has", "was", "were", "will", "can",
    "your", "our", "we", "they", "them", "if", "so", "not", "no", "do",
    "does", "what", "which", "when", "how", "about", "there", "their",
    "more", "most", "very", "also", "than", "then", "these", "those",
}

# Field list mirrors the text a shopper would actually see on a product page.
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool",
             "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown",
          "gray", "grey", "purple", "yellow", "orange")

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)

# Top-level department names that carry no discriminative power.
GENERIC_CATEGORY_PARTS = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

def flatten(value: object) -> list[str]:
    """Turn a catalog field of any shape into a list of plain strings."""
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items() if v not in (None, "", [])]
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{k} {v}" for k, v in value.items())
        elif isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)
            if len(t) > 1 and t.lower() not in STOPWORDS]


def normalise_phrase(text: str, limit: int = 180) -> str:
    """Collapse whitespace and trim punctuation so phrases compare stably."""
    cleaned = re.sub(r"\s+", " ", text).strip(" -;,.\t\n")[:limit].rstrip()
    return cleaned.lower()


def coarse_category(values: list[str]) -> str:
    """Reduce a category breadcrumb to the two most specific labels.

    Amazon breadcrumbs run general -> specific, and the leading department is
    the same for every product in this frozen catalog, so only the tail
    carries information.
    """
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in GENERIC_CATEGORY_PARTS:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def attribute_phrases(product: dict, limit: int = 180) -> list[str]:
    """Candidate requirement phrases a shopper might state for this product.

    Drawn from the structured attribute text plus the material, colour and
    price signals that shoppers most often lead with.
    """
    phrases: list[str] = []
    phrases.extend(flatten(product.get("features")))
    phrases.extend(flatten(product.get("details")))

    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    if material:
        phrases.append(material.group(1).lower())
    colour = COLOR_RE.search(corpus)
    if colour:
        phrases.append(f"color: {colour.group(1).lower()}")
    if product.get("price") not in (None, ""):
        phrases.append(f"budget around ${product['price']}")

    out: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        norm = normalise_phrase(phrase, limit)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

class CatalogIndex:
    """Immutable in-memory index over the frozen catalog."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.ids: list[str] = []
        self.id_pos: dict[str, int] = {}
        self.titles: list[str] = []
        self.category_of: list[str] = []
        self.popularity: list[float] = []
        self.avg_rating: list[float] = []
        self.store_of: list[str] = []
        self.doc_len: list[int] = []

        self.category_members: dict[str, list[int]] = defaultdict(list)
        self.phrase_index: dict[str, list[int]] = defaultdict(list)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.neighbours: dict[str, list] = self._load_neighbours()

        self._load(catalog_path)
        self._finalise()

    @staticmethod
    def _load_neighbours() -> dict[str, list]:
        artifact = Path(__file__).with_name("semantic_neighbours.json")
        if not artifact.exists():
            return {}
        try:
            with artifact.open(encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    # -- construction ------------------------------------------------------

    def _load(self, catalog_path: str | Path) -> None:
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for pos, line in enumerate(handle):
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])

                self.ids.append(asin)
                self.id_pos[asin] = pos
                self.titles.append(str(product.get("title") or ""))

                category = coarse_category(product.get("categories") or [])
                self.category_of.append(category)
                self.category_members[category.lower()].append(pos)

                # Real purchases skew towards products with many ratings, so a
                # mild popularity prior is a genuine signal rather than a tie
                # break only.
                try:
                    self.avg_rating.append(float(product.get("average_rating") or 0.0))
                except (TypeError, ValueError):
                    self.avg_rating.append(0.0)
                self.store_of.append(str(product.get("store") or ""))

                ratings = product.get("rating_number") or 0
                try:
                    ratings = float(ratings)
                except (TypeError, ValueError):
                    ratings = 0.0
                self.popularity.append(math.log1p(max(ratings, 0.0)))

                for phrase in attribute_phrases(product):
                    self.phrase_index[phrase].append(pos)

                counts: dict[str, int] = defaultdict(int)
                for token in tokenize(searchable_text(product)):
                    counts[token] += 1
                self.doc_len.append(sum(counts.values()) or 1)
                for token, count in counts.items():
                    self.postings[token].append((pos, count))

    def _finalise(self) -> None:
        self.size = len(self.ids)
        self.avg_len = sum(self.doc_len) / max(self.size, 1)
        self.idf = {
            token: math.log(1.0 + (self.size - len(plist) + 0.5) / (len(plist) + 0.5))
            for token, plist in self.postings.items()
        }
        max_pop = max(self.popularity) if self.popularity else 1.0
        self.popularity = [p / max_pop if max_pop else 0.0 for p in self.popularity]
        self.category_members = dict(self.category_members)
        self.phrase_index = dict(self.phrase_index)

    # -- lookup ------------------------------------------------------------

    def bm25(self, tokens: list[str], k1: float = 1.4, b: float = 0.72,
             restrict: set[int] | None = None) -> dict[int, float]:
        """Sparse BM25 scores over the (optionally restricted) candidate set."""
        return self.bm25_weighted([(t, 1.0) for t in tokens], k1, b, restrict)

    def bm25_weighted(self, weighted_tokens: list[tuple[str, float]],
                      k1: float = 1.4, b: float = 0.72,
                      restrict: set[int] | None = None) -> dict[int, float]:
        """BM25 where each query term carries its own weight.

        Expansion terms enter here at a fraction of an original term's weight,
        so a synonym can promote a product but never outrank a literal match.
        """
        scores: dict[int, float] = defaultdict(float)
        for token, weight in weighted_tokens:
            plist = self.postings.get(token)
            if not plist or weight <= 0.0:
                continue
            idf = self.idf[token]
            for pos, freq in plist:
                if restrict is not None and pos not in restrict:
                    continue
                norm = 1.0 - b + b * (self.doc_len[pos] / self.avg_len)
                scores[pos] += weight * idf * (freq * (k1 + 1.0)) / (freq + k1 * norm)
        return scores

    def expand(self, tokens: list[str], decay: float = 0.45,
               per_term: int = 3) -> list[tuple[str, float]]:
        """Add corpus-learned synonyms to a query.

        BM25 cannot match a requirement phrased in vocabulary the product page
        never uses. The neighbour table, learned by LSA over this catalog,
        supplies the missing bridge. Absent the artifact this returns the
        original query unchanged, so the agent degrades to pure lexical
        matching rather than failing.
        """
        out: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
        if not self.neighbours:
            return out
        original = set(tokens)
        added: dict[str, float] = {}
        for token in tokens:
            for neighbour, similarity in self.neighbours.get(token, [])[:per_term]:
                if neighbour in original:
                    continue
                weight = decay * float(similarity)
                if weight > added.get(neighbour, 0.0):
                    added[neighbour] = weight
        out.extend(added.items())
        return out

    def phrase_lookup(self, phrase: str) -> list[int]:
        return self.phrase_index.get(normalise_phrase(phrase), [])

    def phrase_specificity(self, phrase: str) -> float:
        """How much a phrase match narrows the catalog.

        A requirement satisfied by one product identifies it outright; one
        satisfied by five thousand products barely reorders the pool. Scaling
        the phrase bonus by inverse document frequency is what separates a
        confident rank-1 answer from a merely plausible top ten.
        """
        members = self.phrase_index.get(normalise_phrase(phrase))
        if not members:
            return 0.0
        return math.log(self.size / len(members)) / math.log(self.size)

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        packed = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if len(packed) < 3:
            return {packed} if packed else set()
        return {packed[i:i + 3] for i in range(len(packed) - 2)}

    def category_lookup(self, label: str, floor: float = 0.50,
                        cap: int = 12000) -> list[int]:
        """Resolve a spoken product family to candidate products.

        Exact match wins outright. Otherwise similarity is the better of token
        overlap and character-trigram overlap -- the first survives dropped or
        reordered words, the second survives misspellings.

        Crucially this returns the union of every plausible family rather than
        the single best one. Committing to one label on a near-tie routes the
        entire session into the wrong aisle with no way to recover; pooling the
        contenders keeps the target reachable and lets ranking arbitrate.
        """
        key = normalise_phrase(label)
        direct = self.category_members.get(key)
        if direct:
            return direct

        wanted_tokens = set(tokenize(label))
        wanted_grams = self._trigrams(label)
        if not wanted_tokens and not wanted_grams:
            return []

        scored: list[tuple[float, str]] = []
        for candidate in self.category_members:
            other_tokens = set(tokenize(candidate))
            token_sim = 0.0
            if wanted_tokens and other_tokens:
                token_sim = len(wanted_tokens & other_tokens) / len(wanted_tokens | other_tokens)
            other_grams = self._trigrams(candidate)
            gram_sim = 0.0
            if wanted_grams and other_grams:
                gram_sim = len(wanted_grams & other_grams) / len(wanted_grams | other_grams)
            best = max(token_sim, gram_sim)
            if best >= floor:
                scored.append((best, candidate))

        if not scored:
            return []

        scored.sort(reverse=True)
        top = scored[0][0]
        pooled: list[int] = []
        for similarity, candidate in scored:
            if similarity < top * 0.6:
                break
            pooled.extend(self.category_members[candidate])
            if len(pooled) >= cap:
                break
        return pooled
