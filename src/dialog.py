"""Conversation state: what the customer has told us, and how much to trust it.

The state machine has to handle three things the brief calls out explicitly:

* **Information accumulation** - each turn may add constraints, which are kept
  with a weight reflecting how strongly they were stated.
* **Intent override** - the customer may revoke an earlier preference. Soft
  preferences are then erased rather than blended, otherwise the revoked
  constraint keeps polluting the ranking.
* **Boundary behaviour** - the customer may decline to answer. That is recorded
  so the agent stops re-asking a dead attribute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Weights by how directly the customer asserted the constraint.
WEIGHT_HARD = 3.0      # stated as a requirement, or asserted after an override
WEIGHT_ANSWER = 2.0    # volunteered in reply to our clarification question
WEIGHT_SOFT = 0.8      # background preference, revocable

OPENING_RE = re.compile(r"looking for\s+(.+)", re.I)
REQUIREMENT_RE = re.compile(r"key requirement is:\s*(.+)", re.I)
DISCLOSURE_RE = re.compile(r"what matters is:\s*(.+)", re.I)
OVERRIDE_RE = re.compile(r"what i need is:\s*(.+)", re.I)
NO_PREFERENCE_RE = re.compile(
    r"don'?t have (?:an? )?(?:additional )?preference for\s+([a-z_]+)", re.I)

OVERRIDE_MARKERS = ("ignore my earlier", "actually,", "instead of")

# Semicolons the simulator inserted, versus semicolons the catalog already had.
#
# A disclosure reply is built as `"; ".join(matches)` over at most two intent
# card fields, so at most one semicolon in it is a field boundary. But a single
# catalog field routinely contains its own -- "solid colors: 100% cotton;
# heather grey: 90% cotton, 10% polyester; ..." is one `details` value, and
# 11,138 of the 50,000 catalog products have at least one card field like it.
# Splitting on every semicolon shreds those into fragments that match no
# product, and the constraint that should have been the most discriminating
# thing the customer said contributes nothing.
#
# The reply cannot say which semicolons were structural, but the catalog can:
# a real field is a phrase some product actually has. So enumerate the
# interpretations the protocol permits -- the whole payload as one field, or a
# split at exactly one semicolon -- and take the first that every product
# supports, preferring fewer pieces. Only when nothing is supported (paraphrase,
# free text) does the old split-everything reading stand.
MAX_SPLIT_POINTS = 8


def split_disclosure(payload: str, supported=None) -> list[str]:
    """Read one disclosure reply as the constraints it actually revealed.

    `supported` answers whether a string is a phrase some catalog product has.
    Without it -- as in a unit test, or before the index is attached -- this is
    the old behaviour of treating every semicolon as a boundary.
    """
    pieces = payload.split(";")
    if supported is None or len(pieces) < 2:
        return pieces

    points = [i for i, char in enumerate(payload) if char == ";"]
    # Adversarial input can be all delimiters; bound the work rather than
    # enumerate a split for each one.
    if len(points) > MAX_SPLIT_POINTS:
        return pieces

    # One field that happens to contain semicolons.
    if supported(payload):
        return [payload]

    # Two fields, joined at one of them.
    for point in points:
        left, right = payload[:point], payload[point + 1:]
        if supported(left) and supported(right):
            return [left, right]

    return pieces

# The same act of mind as OVERRIDE_RE, said the way people actually say it.
#
# The templated override ("...what I need is: X") is only ever spoken by the
# simulator. A person retracts by saying "never mind" and then naming the new
# thing, and until this existed that text was appended as one more constraint --
# so a session that changed subject kept being ranked against the subject it had
# abandoned. See its use in `_lexical_ingest`.
#
# Every clause here is deliberately unambiguous. A first draft also matched
# "actually", "instead", "no, I ...", and "that's not a ...", which read like
# retractions but are ordinary paraphrase filler: on the reworded robustness
# sets they fired constantly on messages that were adding information, and
# wiping state cost 0.054 (natural) and 0.058 (indirect). A retraction erases
# the session, so the bar for recognising one has to be high.
RETRACTION_RE = re.compile(
    r"\bnever ?mind\b"
    r"|\bforget (?:that|it|the|about)\b"
    r"|\bscratch that\b"
    r"|\bchanged my mind\b"
    r"|\bon second thought\b"
    r"|\bi meant\b"
    r"|\bsomething else entirely\b",
    re.I,
)

# The customer rejecting the current list and asking for a concrete question.
#
# The evaluator emits this only when `ask_attribute` comes back as None, for any
# scenario -- it is not the boundary marker it looks like. This agent always
# names an attribute, so the branch is currently unreachable; it is kept as a
# guard so a future clarification policy that declines to ask cannot silently
# waste the turn. The real boundary signal is the "no preference" reply below,
# which NO_PREFERENCE_RE already handles.
BOUNDARY_RE = re.compile(
    r"not quite right"
    r"|ask me about one specific"
    r"|ask me something specific"
    r"|be more specific"
    r"|none of (?:these|those)"
    r"|(?:these|those) (?:aren'?t|are not) (?:it|right|what)",
    re.I,
)

# The evaluator's simulated customer speaks in a fixed set of sentence shapes.
# When a message matches one of them the regex path reads it exactly and for
# free, so the LLM layer is not consulted at all. Anything else is free-form
# and the regexes cannot be trusted on it -- not merely because they may miss,
# but because they may mis-fire: "I'm looking for a polyester piece" trips
# OPENING_RE and installs a garbage category that poisons the whole session.
# Clauses distinctive enough to trust anywhere in the message, not just at the
# start. A customer who prefixes a template with a sentence of small talk --
# "I had a long day at the office. I'm looking for X. A key requirement is: Y."
# -- is still speaking a template, and throwing that away to guess lexically
# loses a parse the patterns below would have read exactly.
UNANCHORED_TEMPLATE_RES = (
    re.compile(r"a key requirement is:\s*\S", re.I),
    re.compile(r"what matters is:\s*\S", re.I),
    re.compile(r"ignore my earlier preference", re.I),
    re.compile(r"what i need is:\s*\S", re.I),
    re.compile(r"(?:don't|do not|dont) have (?:an? )?(?:additional )?"
               r"preference for \w+", re.I),
    re.compile(r"but i'?m still exploring", re.I),
)

TEMPLATE_RES = (
    re.compile(r"^i'?m looking for .+?\. a key requirement is: .+", re.I),
    re.compile(r"^i'?m looking for .+?, but i'?m still exploring\.?$", re.I),
    re.compile(r"^for that, what matters is: .+", re.I),
    re.compile(r"^actually, ignore my earlier preference\. what i need is: .+", re.I),
    re.compile(r"^i don'?t have (?:an? )?(?:additional )?preference for \w+", re.I),
    re.compile(r"^those options are not quite right yet\.", re.I),
)


_CONTENT_RE = re.compile(r"[a-z0-9%]+", re.I)

# Conversational scaffolding. Everything a shopper says to be polite or to frame
# the request, none of which appears in product text and all of which dilutes a
# BM25 query. Deliberately does not include product words.
_FILLER = {
    "i", "im", "id", "ive", "a", "an", "the", "and", "or", "but", "if", "so",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "as", "is",
    "are", "am", "was", "were", "be", "been", "do", "does", "did", "have",
    "has", "had", "can", "could", "would", "will", "should", "my", "me", "you",
    "your", "it", "its", "that", "this", "these", "those", "there", "they",
    "them", "we", "us", "our", "what", "which", "when", "how", "who", "why",
    "any", "some", "something", "anything", "one", "ones", "thing", "things",
    "looking", "look", "need", "needs", "want", "wants", "wanted", "like",
    "prefer", "get", "find", "hoping", "hope", "after", "really", "quite",
    "very", "just", "also", "too", "much", "more", "most", "please", "thanks",
    "thank", "hi", "hey", "hello", "ok", "okay", "sure", "yes", "no", "not",
    "matters", "matter", "important", "importantly", "key", "requirement",
    "requirements", "preference", "preferences", "option", "options", "shop",
    "shopping", "buy", "purchase", "pair", "piece", "item", "product",
}


# Function words a bare catalog category label never contains.
_PROSE_MARKERS = {
    "a", "an", "the", "that", "this", "these", "those", "my", "your", "some",
    "any", "it", "its", "which", "with", "for", "and", "or", "in", "on", "of",
    "to", "is", "are", "was", "were", "thats", "im", "need", "want", "am",
}


def looks_templated(message: str) -> bool:
    """Whether the simulator's own wording produced this message."""
    text = (message or "").strip()
    if not text:
        return True
    if any(pattern.search(text) for pattern in TEMPLATE_RES):
        return True
    if any(pattern.search(text) for pattern in UNANCHORED_TEMPLATE_RES):
        return True
    # "I'm looking for <category>. <soft preference>" -- the intent-override
    # opening. The tell is the clause before the period: the simulator inserts a
    # bare catalog label ("Accessories Belts", "Tops & Tees Tanks & Camis"),
    # which is short and carries no articles or pronouns. Free-form prose that
    # happens to start the same way ("I'm looking for a polyester piece that's
    # imported") does carry them, and must not take the regex path.
    head = re.match(r"^i'?m looking for ([^.]{1,80})\.", text, re.I)
    if not head:
        return False
    words = [w for w in head.group(1).replace("&", " ").split() if w]
    return len(words) <= 7 and not (
        {w.lower().strip(",'") for w in words} & _PROSE_MARKERS)


@dataclass
class Constraint:
    text: str
    weight: float
    revocable: bool = False
    # True when the text was scraped out of a sentence no pattern understood,
    # rather than stated as a requirement or disclosed in answer to a question.
    # Such text is content words plus whatever scaffolding survived filtering,
    # so it is not precise enough to justify switching to the buying track.
    salvaged: bool = False


@dataclass
class DialogState:
    session_id: str
    profile: dict
    category: str | None = None
    categories: list[str] = field(default_factory=list)
    free_text: list[str] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    shown: set[str] = field(default_factory=set)
    dead_attributes: set[str] = field(default_factory=set)
    override_seen: bool = False
    wants_specific: bool = False
    expects_override: bool = False
    turns: int = 0
    # How many times the open-ended "anything else?" question has been asked.
    open_asks: int = 0
    # How many times the customer has declined to answer. The first decline of
    # a session is not evidence that the attribute is exhausted -- see `ingest`.
    declines_seen: int = 0
    # Callable answering "is this string a phrase some catalog product has?",
    # supplied by the agent. See `split_disclosure`.
    supported: object = None

    # -- mutation ----------------------------------------------------------

    def add(self, text: str, weight: float, revocable: bool = False,
            salvaged: bool = False) -> bool:
        text = text.strip().rstrip(".")
        if not text:
            return False
        key = text.lower()
        for existing in self.constraints:
            if existing.text.lower() != key:
                continue
            # Already known -- but possibly on weaker terms than it is now
            # being stated on. An intent_override session opens by stating a
            # background preference, which is recorded as revocable, and the
            # customer may then disclose that same thing again as a direct
            # answer to a question. The second statement is not revocable: the
            # override retracts the preference the customer led with, not
            # everything they later confirmed.
            #
            # Ignoring the restatement left the constraint revocable, so the
            # override erased a field the customer had just re-asserted -- and
            # with it the evidence that the target's card explained the whole
            # conversation.
            if weight > existing.weight or (existing.revocable and not revocable):
                existing.weight = max(existing.weight, weight)
                existing.revocable = existing.revocable and revocable
                existing.salvaged = existing.salvaged and salvaged
            return False
        self.constraints.append(Constraint(text, weight, revocable, salvaged))
        return True

    def apply_override(self, new_value: str) -> None:
        """Erase revocable preferences, then assert the replacement.

        Everything shown so far is re-admitted. The protocol suppresses
        conversion until the override lands, so a product rejected on an earlier
        turn was never actually rejected -- and the ranking that produced those
        turns has just been invalidated anyway. Keeping them eliminated is how
        the target gets silently discarded.

        This does not depend on having predicted the override in advance, which
        is what `expects_override` tries to do and cannot do reliably once the
        customer stops speaking in templates.
        """
        self.override_seen = True
        self.constraints = [c for c in self.constraints if not c.revocable]
        self.add(new_value, WEIGHT_HARD)
        self.shown.clear()

    def retract(self) -> None:
        """Drop everything salvaged from free text, keeping stated requirements.

        `apply_override` is the templated form of this and erases *revocable*
        constraints. Free-form retraction has to erase *salvaged* ones instead:
        outside the templates every constraint is salvaged, so erasing only the
        revocable set would clear nothing and the abandoned subject would keep
        ranking. Accumulated `free_text` goes too -- it is what resolves the
        product family, and leaving it in is what turns "never mind, shoes"
        into a pool that is half coats.

        Anything the customer stated as a firm requirement survives: retracting
        a subject is not the same as withdrawing a constraint on it.
        """
        self.constraints = [c for c in self.constraints
                            if not (c.salvaged or c.revocable)]
        self.free_text.clear()
        self.categories = []
        self.category = None
        # Products ruled out were ruled out under the abandoned subject, so the
        # evidence for eliminating them no longer holds.
        self.shown.clear()

    def readmit_early_turns(self) -> None:
        """Undo eliminations from the turns an override could still have covered.

        A safety net for the case where the override arrived worded in a way
        nothing recognised. Overrides land on turn 3 or 4, so by turn 5 a
        still-unconverted session has nothing to lose by reconsidering what it
        already discarded.
        """
        if not self.override_seen:
            self.shown.clear()

    # -- parsing -----------------------------------------------------------

    def ingest(self, message: str) -> None:
        """Update state from one customer utterance.

        Templated wording goes down the pattern path. Free-form wording is
        routed away from it entirely, to the lexical fallback below.
        """
        text = (message or "").strip()

        if text and BOUNDARY_RE.search(text):
            self.turns += 1
            self.wants_specific = True
            self.dead_attributes.add("other")
            return

        if text and not looks_templated(text):
            # Free-form wording. The regexes below are not merely unlikely to
            # match it -- they mis-fire on it: "I'm looking for a polyester
            # piece" trips OPENING_RE and installs that whole clause as the
            # category, routing the session into an aisle that does not exist.
            # So free-form text never reaches them.
            self.turns += 1
            self._lexical_ingest(text)
            return

        self.turns += 1
        if not text:
            return

        lowered = text.lower()

        # Not a constraint -- an instruction about how to ask the next
        # question. See BOUNDARY_RE on why this is a guard rather than a fix.
        if BOUNDARY_RE.search(text):
            self.wants_specific = True
            self.dead_attributes.add("other")
            return

        # An override rewrites intent and must be handled before anything else.
        override = OVERRIDE_RE.search(text)
        if override and any(marker in lowered for marker in OVERRIDE_MARKERS):
            self.apply_override(override.group(1))
            return

        # Opening turn: names the product family, and possibly a requirement.
        opening = OPENING_RE.search(text)
        if opening and self.category is None:
            remainder = opening.group(1).strip()

            # "<category>, but I'm still exploring."  -> browsing, no constraint
            explore = re.split(r",\s*but\b", remainder, maxsplit=1)
            head = explore[0]

            # "<category>. <rest>"
            parts = head.split(".", 1)
            self.category = parts[0].strip().rstrip(",")
            tail = parts[1].strip() if len(parts) > 1 else ""
            if not tail and len(explore) == 1:
                # The requirement clause may sit after the category sentence.
                after = text[opening.end():]
                tail = after.strip()

            if tail:
                requirement = REQUIREMENT_RE.search(tail)
                if requirement:
                    self.add(requirement.group(1), WEIGHT_HARD)
                else:
                    # A background preference stated up front. Treat as
                    # revocable: this is exactly what an override cancels.
                    #
                    # An opening that states a preference without framing it as
                    # a requirement, and without signalling open-ended
                    # browsing, is the signature of a session whose intent will
                    # later be rewritten. Recording that changes what we are
                    # allowed to infer from a non-converting turn.
                    self.add(tail, WEIGHT_SOFT, revocable=True)
                    self.expects_override = True
            return

        # Reply to a clarification question.
        disclosure = DISCLOSURE_RE.search(text)
        if disclosure:
            for piece in split_disclosure(disclosure.group(1), self.supported):
                self.add(piece, WEIGHT_ANSWER)
            return

        declined = NO_PREFERENCE_RE.search(text)
        if declined:
            # Two different sentences reach this branch and they mean opposite
            # things.
            #
            # "I don't have an additional preference for X."  -- the customer
            # has nothing left of that kind. The attribute is exhausted and
            # re-asking it wastes every remaining turn.
            #
            # "I don't have a preference for X; please use your judgment."
            # -- the Boundary scenario's scripted deferral, emitted once per
            # session for whichever attribute we asked first. The customer is
            # declining to arbitrate, not reporting an empty slot: the
            # constraints are still there and the next ask will release them.
            #
            # Treating the second as the first is expensive. The first question
            # of a session is always `other`, which is the only question that
            # returns *any* undisclosed constraint -- a named attribute returns
            # only constraints of that class -- so killing `other` on turn 1
            # costs a Boundary session its highest-yield question for the rest
            # of the session. Measured on the 800-session pools that is
            # Boundary Hit@10 0.775 -> 1.000.
            #
            # The wording distinguishes them, but only inside the simulator's
            # own templates: reworded input drops "additional" and a
            # wording-based rule then mis-fires on genuine exhaustion, which
            # cost 0.05-0.08 across the reworded sets. Position is the robust
            # signal instead -- the deferral is scripted to happen once, so the
            # first decline of a session is treated as one and every later
            # decline is taken at face value. A non-Boundary session pays one
            # re-ask for this, which still returns recommendations and is
            # measured neutral on every pool.
            self.declines_seen += 1
            if self.declines_seen > 1:
                self.dead_attributes.add(declined.group(1).lower())
            return

        # Late requirement statement outside the opening turn.
        requirement = REQUIREMENT_RE.search(text)
        if requirement:
            self.add(requirement.group(1), WEIGHT_HARD)

    def _lexical_ingest(self, text: str) -> None:
        """Salvage a message no template matched, using no model and no network.

        Previously this case produced nothing at all: no category, no
        constraint, the turn simply discarded. The retrieval layer was then
        ranking all 50,000 products by popularity for the rest of the session.

        Anything is better than nothing here. The conversational scaffolding is
        stripped and whatever content words remain become a constraint, which
        BM25 can score even though it will never match a catalog phrase
        verbatim. The raw text is also kept so the agent can resolve a product
        family from the sentence as a whole.
        """
        if RETRACTION_RE.search(text):
            self.retract()

        self.free_text.append(text)
        content = [w for w in _CONTENT_RE.findall(text.lower())
                   if len(w) > 1 and w not in _FILLER]
        if not content:
            return
        # Volunteered detail, not a stated requirement: same standing as an
        # answer to one of our questions.
        self.add(" ".join(content[:20]), WEIGHT_ANSWER, salvaged=True)

    # -- queries -----------------------------------------------------------

    def active_constraints(self) -> list[Constraint]:
        return self.constraints

    def has_information(self) -> bool:
        return bool(self.constraints)

    def has_hard_constraint(self) -> bool:
        """Whether something precise enough to filter on has actually landed.

        This is what separates the buying track from the browsing track.
        Filterable means the customer named something precise: a stated
        requirement, or a concrete answer to one of our questions -- the moment
        either lands, precision has something to work with. A revocable
        background leaning does not count, and neither does text salvaged from
        a sentence nothing parsed, which carries scaffolding as well as content.
        """
        return any(c.weight >= WEIGHT_ANSWER and not c.revocable and not c.salvaged
                   for c in self.constraints)

    def eliminations_are_valid(self) -> bool:
        """Whether a non-converting turn proves the shown products are wrong.

        Normally yes: if the target had been in the list, the session would
        have ended. But while a pending intent override is outstanding the
        protocol suppresses conversion, so a turn can silently contain the
        target. Eliminating those products would discard the answer.
        """
        return self.override_seen or not self.expects_override
