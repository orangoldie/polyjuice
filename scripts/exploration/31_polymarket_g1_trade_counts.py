"""For recent settled LoL events, count trades on the Game 1 winner market."""
import json
import urllib.request

from _tee import tee_stdout
tee_stdout('31_polymarket_g1_trade_counts.txt')


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "polyjuice-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_all_trades(condition_id, max_pages=200):
    all_trades = []
    seen = set()
    page = 0
    while page < max_pages:
        url = (f"https://data-api.polymarket.com/trades"
               f"?market={condition_id}&limit=500&offset={page * 500}")
        try:
            r = get(url, timeout=30)
        except Exception as e:
            print(f"    page {page} ERROR {e}")
            break
        if not isinstance(r, list) or not r:
            break
        new = []
        for t in r:
            key = (t.get("transactionHash"), t.get("timestamp"), t.get("price"),
                   t.get("size"), t.get("side"))
            if key in seen:
                continue
            seen.add(key)
            new.append(t)
        all_trades.extend(new)
        if len(r) < 500 or not new:
            break
        page += 1
    return all_trades


def main():
    with open("data/exploration/29_polymarket_event_inventory.json") as f:
        inv = json.load(f)

    # Filter: closed (settled) events with non-trivial volume, sort by endDate desc.
    settled = [e for e in inv if e.get("closed") and (e.get("volume") or 0) > 100]
    settled.sort(key=lambda e: e.get("endDate") or "", reverse=True)

    print(f"settled events with vol > $100: {len(settled)}")
    print(f"sampling 12 most recent\n")
    print(f"  {'endDate':10s}  {'vol':>10s}  {'trades':>7s}  {'fills/$':>8s}  slug")

    for e in settled[:12]:
        # Pull full event to get markets
        try:
            ev_full = get(f"https://gamma-api.polymarket.com/events?slug={e['slug']}")
            if not ev_full:
                continue
            ev = ev_full[0]
        except Exception as exc:
            print(f"  {e['slug']:50s}  ERROR {exc}")
            continue

        g1 = None
        for m in ev.get("markets", []) or []:
            q = (m.get("question") or "").lower()
            if "game 1 winner" in q:
                g1 = m
                break
        if not g1:
            print(f"  {e['endDate'][:10]}  {'-':>10s}  {'-':>7s}  {'-':>8s}  {e['slug']}  (no G1)")
            continue

        trades = fetch_all_trades(g1["conditionId"])
        vol = float(g1.get("volume") or 0)
        rate = (len(trades) / vol) if vol > 0 else 0.0
        print(f"  {e['endDate'][:10]}  ${vol:>9,.0f}  {len(trades):>7d}  {rate:>8.3f}  {e['slug']}")


if __name__ == "__main__":
    main()
