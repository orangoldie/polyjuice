"""Probe a single live LoL event end-to-end: markets, history, trades."""
import json
import urllib.request


def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "polyjuice-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


SLUG = "lol-we-blg-2026-04-29"


def main():
    print(f"target slug: {SLUG}")
    print("=" * 70)
    print("1) Event metadata via Gamma")
    print("=" * 70)
    ev_list = get(f"https://gamma-api.polymarket.com/events?slug={SLUG}")
    if not ev_list:
        print("no event found at that slug")
        return
    ev = ev_list[0]
    print(f"  id={ev.get('id')}  title={ev.get('title')}")
    print(f"  start={ev.get('startDate')}  end={ev.get('endDate')}  closed={ev.get('closed')}")
    print(f"  volume=${float(ev.get('volume') or 0):,.2f}  liquidity=${float(ev.get('liquidity') or 0):,.2f}")
    markets = ev.get("markets", []) or []
    print(f"  markets in event: {len(markets)}")
    for i, m in enumerate(markets):
        print(f"\n  --- market {i} ---")
        print(f"    id={m.get('id')}  conditionId={m.get('conditionId')}")
        print(f"    Q: {m.get('question')}")
        print(f"    startDate={m.get('startDate')}  endDate={m.get('endDate')}")
        print(f"    outcomes={m.get('outcomes')}")
        print(f"    volume=${float(m.get('volume') or 0):,.2f}  spread={m.get('spread')}")
        print(f"    bestBid={m.get('bestBid')}  bestAsk={m.get('bestAsk')}  lastTrade={m.get('lastTradePrice')}")
        print(f"    clobTokenIds={m.get('clobTokenIds')}")
        print(f"    closed={m.get('closed')}  active={m.get('active')}  closedTime={m.get('closedTime')}")

    # 2. prices-history on the first market's first token
    if markets:
        m0 = markets[0]
        tokens = json.loads(m0.get("clobTokenIds") or "[]")
        if tokens:
            print()
            print("=" * 70)
            print("2) prices-history (token 0 = outcome 0)")
            print("=" * 70)
            for interval in ["1m", "1h", "max"]:
                try:
                    r = get(f"https://clob.polymarket.com/prices-history?market={tokens[0]}&interval={interval}")
                    hist = r.get("history", []) if isinstance(r, dict) else r
                    print(f"  interval={interval}: {len(hist)} points")
                    if hist:
                        print(f"    first={hist[0]}")
                        print(f"    last ={hist[-1]}")
                        if len(hist) >= 3:
                            print(f"    mid  ={hist[len(hist)//2]}")
                except Exception as e:
                    print(f"  interval={interval} ERROR {e}")

            # 3. CLOB trades — try both common paths
            print()
            print("=" * 70)
            print("3) CLOB trades for this market")
            print("=" * 70)
            for url in [
                f"https://clob.polymarket.com/trades?market={m0.get('conditionId')}",
                f"https://data-api.polymarket.com/trades?market={m0.get('conditionId')}&limit=20",
                f"https://data-api.polymarket.com/trades?conditionId={m0.get('conditionId')}&limit=20",
            ]:
                try:
                    r = get(url, timeout=15)
                    print(f"\n  {url}")
                    if isinstance(r, list):
                        print(f"    list len={len(r)}")
                        if r:
                            print(f"    first={json.dumps(r[0])[:300]}")
                    elif isinstance(r, dict):
                        print(f"    keys={list(r.keys())}")
                except Exception as e:
                    print(f"\n  {url}\n    ERROR {e}")


if __name__ == "__main__":
    main()
