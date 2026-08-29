"""Personalized context distillation.

The harness supplies an anonymised aggregate profile, not a purchase history.
It carries three usable signals:

* ``preference_tags``  -- a closed vocabulary of nine shopping concerns.
* ``rating_style``     -- how the customer rates: positive, mixed, critical.
* ``average_prior_rating`` -- the level they rate at.

Distillation converts those into two things the ranker can act on: catalog
vocabulary to search for, and a per-session adjustment to how much the
popularity prior is trusted. Nothing here is a lookup of the individual; the
profile is aggregate by construction and stays that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Preference tags are abstract concerns. Products describe themselves in
# concrete words. This bridges the two.
TAG_VOCABULARY = {
    "fit": ("fit", "fitted", "tailored", "true", "sizing", "regular"),
    "comfort": ("comfortable", "comfort", "soft", "cushioned", "cozy", "smooth"),
    "material": ("cotton", "polyester", "fabric", "blend", "leather", "wool"),
    "style": ("style", "classic", "design", "modern", "elegant", "casual"),
    "durability": ("durable", "sturdy", "reinforced", "lasting", "strong"),
    "performance": ("performance", "athletic", "moisture", "wicking", "breathable"),
    "warmth": ("warm", "insulated", "fleece", "thermal", "winter"),
    "weather": ("waterproof", "rain", "windproof", "weather", "resistant"),
    "general shopping": (),
}

# Which clarification attribute a tag most plausibly resolves to. Used when
# open questions stop producing information.
TAG_ATTRIBUTE = {
    "fit": "size",
    "comfort": "feature",
    "material": "material",
    "style": "style",
    "durability": "feature",
    "performance": "use_case",
    "warmth": "use_case",
    "weather": "use_case",
}

RATING_STYLE_TRUST = {
    "usually positive": 1.0,  # their high ratings track broad appeal
    "mixed": 0.85,
    "critical": 0.7,           # discriminating; crowd favourites mean less
}


@dataclass
class DistilledProfile:
    """A profile reduced to things the ranker can use."""

    query_terms: list[str] = field(default_factory=list)
    attribute_order: list[str] = field(default_factory=list)
    popularity_trust: float = 1.0
    rating_target: float | None = None
    tags: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One line for logs and demo output."""
        tags = ", ".join(self.tags) or "no tags"
        return (f"tags[{tags}] popularity_trust={self.popularity_trust:.2f} "
                f"rating_target={self.rating_target}")


def distill(profile: dict | None) -> DistilledProfile:
    profile = profile or {}
    tags = [str(t).lower() for t in (profile.get("preference_tags") or [])]

    terms: list[str] = []
    for tag in tags:
        terms.extend(TAG_VOCABULARY.get(tag, (tag,)))

    # Tags are listed most-emphasised first, so preserve that order when
    # deciding which attribute to ask about.
    attribute_order: list[str] = []
    for tag in tags:
        attribute = TAG_ATTRIBUTE.get(tag)
        if attribute and attribute not in attribute_order:
            attribute_order.append(attribute)

    style = str(profile.get("rating_style") or "").lower()
    trust = RATING_STYLE_TRUST.get(style, 0.9)

    rating_target = profile.get("average_prior_rating")
    if not isinstance(rating_target, (int, float)):
        rating_target = None

    return DistilledProfile(
        query_terms=list(dict.fromkeys(terms)),
        attribute_order=attribute_order,
        popularity_trust=trust,
        rating_target=float(rating_target) if rating_target is not None else None,
        tags=tags,
    )
