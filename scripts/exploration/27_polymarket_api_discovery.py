"""Probe Polymarket Gamma API for LoL markets."""
import json
import urllib.request
import urllib.parse


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "polyjuice-probe"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    # 1. Look for LoL-flavored series
    print("=" * 70)
    print("1) Series matching LoL keywords")
    print("=" * 70)
    series = get("https://gamma-api.polymarket.com/series?limit=500")
    print(f"total series returned: {len(series)}")
    keys = ["lol", "league of legends", "lck", "lpl", "lec", "lcs", "lcp",
            "riot", "msi", "worlds", "esports", "league-of-legends"]
    matched_series = []
    for s in series:
        blob = (str(s.get("slug", "")) + " " + str(s.get("title", ""))).lower()
        if any(k in blob for k in keys):
            matched_series.append(s)
            print(f"  {s.get('slug'):40s} | {s.get('title'):50s} | rec={s.get('recurrence')}")
    print(f"matched: {len(matched_series)}")

    # 2. Try direct series_slug pulls for likely LoL slugs
    print()
    print("=" * 70)
    print("2) Direct lookup by candidate series slugs")
    print("=" * 70)
    candidates = ["lol-worlds", "lol", "league-of-legends", "lck", "lpl",
                  "lec", "lcs", "msi", "lol-msi", "lol-lck", "worlds-lol"]
    for slug in candidates:
        try:
            url = f"https://gamma-api.polymarket.com/events?series_slug={slug}&limit=200&closed=true"
            ev = get(url)
            if ev:
                vol = sum(float(e.get("volume") or 0) for e in ev)
                dates = sorted([e.get("endDate") for e in ev if e.get("endDate")])
                date_range = f"{dates[0][:10]} -> {dates[-1][:10]}" if dates else "n/a"
                print(f"  {slug:25s} events={len(ev):4d} vol=${vol:>12,.0f} {date_range}")
            else:
                print(f"  {slug:25s} events=   0")
        except Exception as e:
            print(f"  {slug:25s} ERROR {e}")

    # 3. Search via the events endpoint with text search
    print()
    print("=" * 70)
    print("3) Text searches over events")
    print("=" * 70)
    for q in ["league of legends", "LoL", "Worlds 2024", "LCK", "LPL"]:
        try:
            url = f"https://gamma-api.polymarket.com/events?limit=50&closed=true&q={urllib.parse.quote(q)}"
            ev = get(url)
            print(f"  q={q!r:30s} -> {len(ev)} events")
            for e in ev[:5]:
                print(f"     {e.get('slug')[:60]:60s} | end={str(e.get('endDate'))[:10]} | vol=${float(e.get('volume') or 0):,.0f}")
        except Exception as e:
            print(f"  q={q!r} ERROR {e}")


if __name__ == "__main__":
    main()
