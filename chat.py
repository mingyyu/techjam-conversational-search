"""Interactive terminal chat with the shopping agent.

`demo.py` replays scripted sessions from the official simulator. This is the
other half: you type, the agent answers, and there is no hidden target and no
scoring. It is the same `Agent` the evaluator drives -- no special casing, no
staged replies -- so what you see here is exactly what gets scored.

    python chat.py                     # talk to it
    python chat.py --state             # show the agent's internal state each turn
    python chat.py --browse            # always show ten results (see the note below)
    python chat.py --tags fit,warmth   # start from a preference profile

Two things are worth knowing before you read the output.

**It shows one product for the first three turns.** That is deliberate and it is
where most of the score comes from: the evaluator fixes the target's rank the
first time it appears, so committing to a single best pick while the customer
still has something to say beats hedging with ten. See COMMIT_TURNS in
`src/shopping_agent.py`. For a shopping demo where you would rather browse a
list, `--browse` turns it off -- but that is presentation, not the scored agent.

**Typing freely exercises the fallback path.** The simulator speaks in fixed
sentence shapes ("I'm looking for X. A key requirement is: Y."). Anything else
is routed to a lexical salvage path instead of the template parser, which is the
same code that handles reworded input in the robustness runs. `--state` labels
which path each message took, so you can watch it happen.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import src.shopping_agent as sa
from src.dialog import looks_templated
from starter.agent import Agent

WIDTH = 74
RULE = "=" * WIDTH
THIN = "-" * WIDTH


def safe(text) -> str:
    """Catalog titles carry curly quotes and CJK; a cp1252 console cannot print
    them. Never let a product name kill the session."""
    return str(text).encode("ascii", "replace").decode("ascii")


def load_products(path: str) -> dict:
    products = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                product = json.loads(line)
                products[str(product["parent_asin"])] = product
    return products


def say(prefix: str, text: str) -> None:
    pad = " " * len(prefix)
    for i, line in enumerate(textwrap.wrap(safe(text), WIDTH - len(prefix)) or [""]):
        print((prefix if i == 0 else pad) + line)


def show_product(rank: int, asin: str, product: dict) -> None:
    title = safe(product.get("title") or "(untitled)")
    print(f"  {rank:>2}. {title[:64]}")
    bits = [asin]
    price = product.get("price")
    if price not in (None, "", "None"):
        bits.append(f"${price}")
    rating, count = product.get("average_rating"), product.get("rating_number")
    if rating:
        bits.append(f"{rating}* ({count or 0} ratings)")
    store = product.get("store")
    if store:
        bits.append(safe(str(store))[:28])
    print("      " + "  |  ".join(bits))


def show_state(agent: Agent) -> None:
    impl = agent._impl
    state = impl.state
    print("  " + THIN[:60])
    print(f"  track       {impl._track.name.upper()}"
          f"   (popularity w={impl._track.w_popularity}, bm25 w={impl._track.w_bm25})")
    print(f"  category    {state.category or '(unresolved)'}")
    if state.constraints:
        for constraint in state.constraints:
            tag = ("revocable" if constraint.revocable
                   else "salvaged" if constraint.salvaged else "firm")
            print(f"  slot        {safe(constraint.text)[:46]!r} "
                  f"w={constraint.weight} [{tag}]")
    else:
        print("  slots       (none yet)")
    print(f"  pool        {len(impl._candidates or [])} candidates"
          f"   | mode={impl.orchestrator.mode}")
    print(f"  ruled out   {len(state.shown)} products")
    if state.dead_attributes:
        print(f"  exhausted   {', '.join(sorted(state.dead_attributes))}")
    print("  " + THIN[:60])


HELP = """
  /state     show the agent's internal state after each reply (toggle)
  /more      show ten results for the next reply, whatever the turn
  /reset     start a fresh session, forgetting everything
  /help      this
  /quit      leave
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat with the shopping agent")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--state", action="store_true",
                        help="show internal state after each reply")
    parser.add_argument("--browse", action="store_true",
                        help="always return ten results (not the scored behaviour)")
    parser.add_argument("--tags", default="",
                        help="comma-separated preference tags, e.g. fit,comfort,warmth")
    args = parser.parse_args()

    if not Path(args.catalog).exists():
        print(f"No catalog at {args.catalog}. See docs/participant_kit_README.md "
              f"for the download step.")
        return 1

    if args.browse:
        sa.COMMIT_TURNS = 0

    print("Loading the catalog and building the index (a few seconds) ...")
    products = load_products(args.catalog)
    agent = Agent(args.catalog)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    profile = {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.0,
        "rating_style": "usually positive",
        "preference_tags": tags,
        "summary": ("Prior purchases emphasize " + ", ".join(tags) + "."
                    if tags else "No strong prior signal."),
    }

    show_internals = args.state
    session = 0
    turn = 1
    agent.reset(f"chat-{session}", profile)

    print(RULE)
    print(f"  {len(products):,} products.  Offline: no network, no model, no tokens.")
    print(f"  Type what you are shopping for.  /help for commands, /quit to leave.")
    if not args.browse:
        print(f"  Heads up: it shows ONE pick for the first {sa.COMMIT_TURNS} turns "
              f"by design --")
        print(f"  answer its question and it keeps narrowing.  /more or --browse "
              f"to see ten.")
    print(RULE)

    force_wide = False
    while True:
        try:
            message = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0

        if not message:
            continue
        low = message.lower()
        if low in ("/quit", "/exit", "/q"):
            print("bye.")
            return 0
        if low == "/help":
            print(HELP)
            continue
        if low == "/state":
            show_internals = not show_internals
            print(f"  (state display {'on' if show_internals else 'off'})")
            continue
        if low == "/more":
            force_wide = True
            print("  (next reply will show ten)")
            continue
        if low == "/reset":
            session += 1
            turn = 1
            agent.reset(f"chat-{session}", profile)
            print("  (new session -- everything forgotten)")
            continue

        saved = sa.COMMIT_TURNS
        if force_wide:
            sa.COMMIT_TURNS = 0
        try:
            response = agent.respond(f"chat-{session}", message, turn, 10)
        finally:
            sa.COMMIT_TURNS = saved
            force_wide = False

        print()
        say("agent > ", response["message"])
        print()
        for rank, item in enumerate(response["recommendations"], 1):
            asin = item["parent_asin"]
            show_product(rank, asin, products.get(asin, {}))

        if show_internals:
            print()
            parsed = "template parser" if looks_templated(message) else "lexical salvage"
            print(f"  read via   {parsed}")
            show_state(agent)

        turn += 1
        if turn > 10:
            print("\n  (ten turns -- the scored protocol would have ended here. "
                  "/reset to start over.)")


if __name__ == "__main__":
    sys.exit(main())
