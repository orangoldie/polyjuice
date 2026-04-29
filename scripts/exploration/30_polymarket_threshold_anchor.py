"""For one settled LoL game, pull all trades and find threshold-cross times."""
import json
import urllib.request
import datetime as dt
from collections import Counter


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "polyjuice-probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# Take the WE vs BLG event we already inspected. Game 1 is resolved.
SLUG = "lol-we-blg-2026-04-29"


def fetch_all_trades(condition_id, max_pages=200):
    """Paginate via offset against data-api/trades."""
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
            key = (t.get("transactionHash"), t.get("timestamp"), t.get("price"), t.get("size"), t.get("side"))
            if key in seen:
                continue
            seen.add(key)
            new.append(t)
        all_trades.extend(new)
        if len(r) < 500:
            break
        if not new:
            break
        page += 1
    return all_trades


def analyze(trades, asset_token_we, asset_token_blg):
    """Trades are mixed YES/NO leg. Bucket by asset token, normalize to 'P(BLG wins)' since BLG won G1."""
    # Each trade has `asset` = clob token id of one outcome; `side` = BUY/SELL of that token at `price`.
    # P(BLG wins) at the time of a fill = price if asset == BLG_token else 1 - price.
    series = []
    for t in trades:
        ts = int(t.get("timestamp"))
        p = float(t.get("price"))
        sz = float(t.get("size"))
        side = t.get("side")
        asset = t.get("asset")
        if asset == asset_token_blg:
            p_blg = p
        elif asset == asset_token_we:
            p_blg = 1.0 - p
        else:
            continue
        series.append((ts, p_blg, sz, side, asset))
    series.sort(key=lambda x: x[0])
    return series


def find_first_cross(series, threshold):
    """First timestamp at which p_blg >= threshold and never dips back below."""
    # Walk from the end backward; find the last point < threshold; return the next point's time.
    last_below = None
    for i, (ts, p, *_) in enumerate(series):
        if p < threshold:
            last_below = i
    if last_below is None:
        return series[0][0] if series else None
    if last_below + 1 >= len(series):
        return None
    return series[last_below + 1][0]


def main():
    print(f"event slug: {SLUG}")
    ev = get(f"https://gamma-api.polymarket.com/events?slug={SLUG}")[0]
    print(f"  title={ev.get('title')}  endDate={ev.get('endDate')}")

    # Find Game 1 winner market
    g1 = None
    for m in ev.get("markets", []):
        q = (m.get("question") or "").lower()
        if "game 1 winner" in q:
            g1 = m
            break
    if not g1:
        print("Game 1 winner market not found")
        return

    cond = g1["conditionId"]
    tokens = json.loads(g1["clobTokenIds"])
    outcomes = json.loads(g1["outcomes"])
    print(f"\nGame 1 market:")
    print(f"  Q: {g1['question']}")
    print(f"  outcomes: {outcomes}")
    print(f"  tokens : {tokens}")
    print(f"  volume=${float(g1.get('volume') or 0):,.2f}  bestBid={g1.get('bestBid')} bestAsk={g1.get('bestAsk')}")

    # outcomes[0]=Team WE, outcomes[1]=Bilibili Gaming. tokens correspond positionally.
    we_token = tokens[0]
    blg_token = tokens[1]
    print(f"  WE token  = {we_token}")
    print(f"  BLG token = {blg_token}")

    # Pull all trades
    print(f"\nfetching trades for conditionId={cond} ...")
    trades = fetch_all_trades(cond)
    print(f"  total trades fetched: {len(trades)}")
    if not trades:
        return

    series = analyze(trades, we_token, blg_token)
    print(f"  usable points: {len(series)}")
    print(f"  time range: {dt.datetime.utcfromtimestamp(series[0][0])} -> "
          f"{dt.datetime.utcfromtimestamp(series[-1][0])}")
    print(f"  first p_blg = {series[0][1]:.4f}")
    print(f"  last  p_blg = {series[-1][1]:.4f}")

    # Threshold crosses
    print(f"\nThreshold-cross analysis for P(BLG wins Game 1):")
    for thr in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
        ts = find_first_cross(series, thr)
        if ts:
            print(f"  first sustained cross >= {thr:.3f}: "
                  f"{dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')} UTC  "
                  f"(unix {ts})")
        else:
            print(f"  first sustained cross >= {thr:.3f}: NEVER")

    # Volume-weighted price curve, 1-min buckets, last 60 minutes
    print(f"\nMinute-by-minute summary (last 30 1-min buckets with trades):")
    buckets = {}
    for ts, p, sz, side, asset in series:
        m = ts // 60
        b = buckets.setdefault(m, {"n": 0, "vol": 0.0, "pmin": 1.0, "pmax": 0.0, "plast": None})
        b["n"] += 1
        b["vol"] += sz
        b["pmin"] = min(b["pmin"], p)
        b["pmax"] = max(b["pmax"], p)
        b["plast"] = p
    last_minutes = sorted(buckets.keys())[-30:]
    print(f"  {'time UTC':19s}  {'n':>4s}  {'vol':>8s}  {'pmin':>6s}  {'pmax':>6s}  {'plast':>6s}")
    for m in last_minutes:
        b = buckets[m]
        ts = m * 60
        print(f"  {dt.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}  "
              f"{b['n']:>4d}  {b['vol']:>8.1f}  {b['pmin']:>6.3f}  {b['pmax']:>6.3f}  {b['plast']:>6.3f}")

    # Also report the LAST trade and a sense of "settle moment"
    last_ts = series[-1][0]
    print(f"\nLast trade: {dt.datetime.utcfromtimestamp(last_ts)} UTC, p_blg={series[-1][1]:.4f}, sz={series[-1][2]}, side={series[-1][3]}")


if __name__ == "__main__":
    main()
