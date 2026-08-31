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


def singular(token: str) -> str:
    """Fold a plural noun onto its singular form.

    Catalog labels are plural ("coats jackets & vests"); shoppers speak in the
    singular ("a winter jacket"). Matching them literally means a customer who
    names their product family in the singular resolves to *no* family at all
    and gets ranked against the whole catalog by popularity. This is the whole
    fix -- deliberately not a stemmer, which would also fold "dress"/"dres" and
    cost precision everywhere else.

    "-es" is only a two-letter plural after a sibilant ("boxes", "watches"). On
    anything else it is a plain "-s" on a stem that already ended in "e", so
    stripping both letters turns "shoes" into "sho".
    """
    if len(token) <= 3 or not token.endswith("s") or token.endswith("ss"):
        return token
    if token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    return token[:-1]


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
        self.store_of: list[str] = []
        self.doc_len: list[int] = []

        self.category_members: dict[str, list[int]] = defaultdict(list)
        self.phrase_index: dict[str, list[int]] = defaultdict(list)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        self._load(catalog_path)
        self._finalise()

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
        """Sparse BM25 scores over the (optionally restricted) candidate set.

        Sparse because only documents containing a query term are ever touched:
        the score map is built from the postings lists rather than by scanning
        all fifty thousand products, which is what keeps a turn under ~13 ms
        with no index beyond the standard library.
        """
        scores: dict[int, float] = defaultdict(float)
        for token in tokens:
            plist = self.postings.get(token)
            if not plist:
                continue
            idf = self.idf[token]
            for pos, freq in plist:
                if restrict is not None and pos not in restrict:
                    continue
                norm = 1.0 - b + b * (self.doc_len[pos] / self.avg_len)
                scores[pos] += idf * (freq * (k1 + 1.0)) / (freq + k1 * norm)
        return scores

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

    def resolve_categories(self, text: str, floor: float = 0.50,
                           top_n: int = 8, cap: int = 15000) -> list[int]:
        """Find the product families a free-form sentence is talking about.

        `category_lookup` compares a spoken label against catalog labels, which
        needs the caller to have isolated the label first. That only works when
        the sentence follows the simulator's template. Here the whole utterance
        is the input, so similarity is measured as *label coverage* -- what share
        of the catalog label's own words appear in the message -- rather than
        Jaccard overlap, which a long sentence would drive to nearly zero.

        As with `category_lookup`, every plausible family above the floor is
        pooled instead of committing to the best one.
        """
        spoken = {singular(t) for t in tokenize(text)}
        if not spoken:
            return []

        scored: list[tuple[float, int, str]] = []
        for label in self.category_members:
            words = {singular(w) for w in tokenize(label)}
            if not words:
                continue
            coverage = len(words & spoken) / len(words)
            if coverage >= floor:
                # More words matched is stronger evidence than a short label
                # matching by luck, so it breaks ties ahead of the label itself.
                scored.append((coverage, len(words & spoken), label))
        if not scored:
            return []

        scored.sort(reverse=True)
        pooled: list[int] = []
        for _, _, label in scored[:top_n]:
            pooled.extend(self.category_members[label])
            if len(pooled) >= cap:
                break
        return list(dict.fromkeys(pooled))

    def category_lookup(self, label: str, floor: float = 0.50,
                        cap: int = 12000) -> list[int]:
        """Resolve a named product family to candidate products.

        The label reaching this function is the model's paraphrase of the
        family, not the shopper's own words, so it rarely matches a catalog
        label verbatim. Exact match wins outright; otherwise similarity is the
        better of token overlap and character-trigram overlap -- the first
        absorbs dropped or reordered words ("jackets" for "coats jackets &
        vests"), the second absorbs suffix and spacing drift ("t-shirt" for
        "tshirts").

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
