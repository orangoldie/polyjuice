"""Enumerate recent LoL events and check slug-pattern compliance."""
import json
import re
import urllib.request
from collections import Counter


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "polyjuice-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


SLUG_PAT = re.compile(r"^lol-([a-z0-9\-]+?)-(\d{4}-\d{2}-\d{2})$")


def main():
    # Pull recent LoL events. Try a few different orderings to find recent ones.
    print("=" * 70)
    print("Recent LoL events via series_slug=league-of-legends")
    print("=" * 70)
    events = []
    seen = set()
    # Pull both closed and open with multiple offset windows.
    for closed in (False, True):
        for offset in (0, 200, 400, 600, 800):
            try:
                url = (f"https://gamma-api.polymarket.com/events"
                       f"?series_slug=league-of-legends&limit=200&offset={offset}&closed={str(closed).lower()}")
                ev = get(url)
                new = [e for e in ev if e.get("id") not in seen]
                for e in new:
                    seen.add(e.get("id"))
                events.extend(new)
                print(f"  closed={closed} offset={offset:4d} -> {len(ev)} returned, {len(new)} new")
                if len(ev) < 200:
                    break
            except Exception as e:
                print(f"  closed={closed} offset={offset} ERROR {e}")
                break

    print(f"\ntotal unique LoL events: {len(events)}")

    # Sort by endDate descending
    def end_key(e):
        return e.get("endDate") or ""

    events_sorted = sorted(events, key=end_key, reverse=True)

    # Slug-pattern compliance
    print()
    print("=" * 70)
    print("Slug pattern: lol-{team}-{team}-{YYYY-MM-DD}")
    print("=" * 70)
    matches = 0
    misses = []
    for e in events:
        slug = e.get("slug") or ""
        if SLUG_PAT.match(slug):
            matches += 1
        else:
            misses.append(slug)
    print(f"  match : {matches}/{len(events)}")
    print(f"  miss  : {len(misses)}/{len(events)}")
    print(f"  example misses (first 15):")
    for m in misses[:15]:
        print(f"    {m}")

    # Date distribution
    print()
    print("=" * 70)
    print("Date distribution (by endDate, year-month)")
    print("=" * 70)
    months = Counter()
    for e in events:
        d = (e.get("endDate") or "")[:7]
        if d:
            months[d] += 1
    for m in sorted(months):
        print(f"  {m}  {months[m]:4d} events")

    # Top 30 most-recent
    print()
    print("=" * 70)
    print("30 most recent events")
    print("=" * 70)
    for e in events_sorted[:30]:
        title = (e.get("title") or "")[:55]
        slug = (e.get("slug") or "")[:60]
        end = (e.get("endDate") or "")[:10]
        vol = float(e.get("volume") or 0)
        n_markets = len(e.get("markets", []) or [])
        is_closed = e.get("closed")
        print(f"  end={end} closed={is_closed!s:5s} mkts={n_markets:3d} vol=${vol:>11,.0f}  {slug}")

    # Save the full list for downstream
    out = "data/exploration/29_polymarket_event_inventory.json"
    summary = [
        {
            "id": e.get("id"),
            "slug": e.get("slug"),
            "title": e.get("title"),
            "startDate": e.get("startDate"),
            "endDate": e.get("endDate"),
            "closed": e.get("closed"),
            "volume": float(e.get("volume") or 0),
            "n_markets": len(e.get("markets", []) or []),
        }
        for e in events_sorted
    ]
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out} with {len(summary)} entries")


if __name__ == "__main__":
    main()
