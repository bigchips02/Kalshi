#!/usr/bin/env python3
"""
Live Probability & Spread Scanner for Kalshi (public v2 REST API).

Beginner-friendly script: fetches open markets, computes implied probabilities
and bid-ask spreads, filters illiquid markets, and prints ranked terminal tables.

No API key required — uses Kalshi's public trade-api endpoint.
"""

import requests
import pandas as pd
from tabulate import tabulate


# ---------------------------------------------------------------------------
# API configuration — exact public endpoint (do not invent alternate URLs)
# ---------------------------------------------------------------------------
API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
REQUEST_PARAMS = {
    "status": "open",  # only markets that are currently tradeable
    "limit": 200,      # how many markets to pull in one request
    # Multivariate (combo) markets often fill the first page with empty books.
    # Excluding them keeps the same endpoint/URL while returning tradeable singles.
    "mve_filter": "exclude",
}
REQUEST_TIMEOUT_SECONDS = 10


def _to_float(value, default=0.0):
    """Safely convert API values (int/float/str/None) to float."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dollars_to_cents(value):
    """
    Convert a dollar price string/number to cents.

    Example: "0.42" dollars → 42.0 cents.
    Kalshi's newer public payloads often use *_dollars fields instead of
    integer-cent fields; we normalize everything to cents for the rest of
    this educational pipeline.
    """
    return _to_float(value) * 100.0


def fetch_open_markets():
    """
    Call Kalshi's public markets endpoint and return the raw markets list.

    Why requests.get?
      HTTP GET asks the server for data without changing anything.
      timeout=10 means we wait at most 10 seconds, then fail instead of hanging.

    Why response.json()?
      The API returns a JSON string in the body. .json() parses it into a
      Python dict/list so we can use normal indexing (.get, [], etc.).
    """
    try:
        # GET with a hard timeout so a slow network cannot freeze the program.
        response = requests.get(
            API_URL,
            params=REQUEST_PARAMS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        # Raise an error for HTTP status codes like 404 or 500.
        response.raise_for_status()

        # Parse JSON body → Python dict. Expected shape: {"markets": [...], ...}
        payload = response.json()
        markets = payload.get("markets", [])
        return markets

    except requests.exceptions.Timeout:
        print("Error: Request timed out after "
              f"{REQUEST_TIMEOUT_SECONDS} seconds. Check your connection.")
        return None
    except requests.exceptions.HTTPError as exc:
        print(f"Error: HTTP problem talking to Kalshi API: {exc}")
        return None
    except requests.exceptions.RequestException as exc:
        # Catches connection errors, DNS failures, and other network issues.
        print(f"Error: Network/request failed: {exc}")
        return None
    except ValueError as exc:
        # Raised when response body is not valid JSON.
        print(f"Error: Could not parse JSON response: {exc}")
        return None


def normalize_market(m):
    """
    Map one raw API market dict into a consistent row of cent-based fields.

    Preferred fields (classic docs / assignment): yes_bid, yes_ask, volume,
    open_interest as integers in cents / counts.

    Current public payloads often omit those and instead return:
      yes_bid_dollars / yes_ask_dollars (strings like "0.42")
      volume_fp / open_interest_fp (fixed-point strings)
      yes_sub_title instead of subtitle

    We prefer the classic keys when present, then fall back so the script
    stays runnable against today's live API.
    """
    # Bids/asks: classic cent ints first, else dollars → cents.
    yes_bid = m.get("yes_bid")
    yes_ask = m.get("yes_ask")
    no_bid = m.get("no_bid")
    no_ask = m.get("no_ask")

    if yes_bid is None:
        yes_bid = dollars_to_cents(m.get("yes_bid_dollars"))
    if yes_ask is None:
        yes_ask = dollars_to_cents(m.get("yes_ask_dollars"))
    if no_bid is None:
        no_bid = dollars_to_cents(m.get("no_bid_dollars"))
    if no_ask is None:
        no_ask = dollars_to_cents(m.get("no_ask_dollars"))

    volume = m.get("volume")
    if volume is None:
        volume = m.get("volume_fp")

    open_interest = m.get("open_interest")
    if open_interest is None:
        open_interest = m.get("open_interest_fp")

    # Subtitle: classic "subtitle", else yes_sub_title / no_sub_title.
    subtitle = m.get("subtitle") or m.get("yes_sub_title") or m.get("no_sub_title") or ""

    return {
        "ticker": m.get("ticker", "") or "",
        "title": m.get("title", "") or "",
        "subtitle": subtitle,
        "yes_bid": _to_float(yes_bid),
        "yes_ask": _to_float(yes_ask),
        "no_bid": _to_float(no_bid),
        "no_ask": _to_float(no_ask),
        "volume": _to_float(volume),
        "open_interest": _to_float(open_interest),
    }


def markets_to_dataframe(markets):
    """
    Convert the API's list of market dicts into a pandas DataFrame.

    Why pandas?
      DataFrames make filtering, sorting, and column math much easier than
      looping over raw dicts by hand — especially for table-style output.
    """
    rows = [normalize_market(m) for m in markets]
    return pd.DataFrame(rows)


def filter_and_enrich(df):
    """
    Drop illiquid / unusable quotes, then add Spread, Mid Price, Implied Prob.

    Filter rules (all must pass):
      - volume > 0              → some trading activity exists
      - yes_bid > 0             → someone is bidding to buy Yes
      - yes_ask > 0             → someone is offering to sell Yes
      - yes_ask <= 99           → ask is not a degenerate near-100¢ quote

    Why filter before ranking?
      Illiquid markets have huge spreads or one-sided books; they look like
      "opportunities" but you often cannot actually trade at the printed price.
    """
    if df.empty:
        return df

    # Boolean mask: keep only rows that satisfy every liquidity condition.
    liquid = (
        (df["volume"] > 0)
        & (df["yes_bid"] > 0)
        & (df["yes_ask"] > 0)
        & (df["yes_ask"] <= 99)
    )
    df = df.loc[liquid].copy()

    if df.empty:
        return df

    # Spread in cents: how far ask is above bid. Tighter = cheaper to trade.
    df["spread_cents"] = df["yes_ask"] - df["yes_bid"]

    # Mid price in cents: halfway between bid and ask (a fair-value estimate).
    df["mid_price_cents"] = (df["yes_bid"] + df["yes_ask"]) / 2.0

    # Implied probability (%): on Kalshi, price in cents ≈ probability × 100.
    # Example: mid 42¢ → market implies ~42% chance Yes pays out.
    df["implied_prob_pct"] = df["mid_price_cents"]

    # Human-readable name: title + subtitle when subtitle exists; max 55 chars.
    def build_market_name(row):
        title = str(row["title"]).strip()
        subtitle = str(row["subtitle"]).strip()
        name = f"{title} — {subtitle}" if subtitle else title
        if len(name) > 55:
            return name[:55]
        return name

    df["market_name"] = df.apply(build_market_name, axis=1)
    return df


def format_display_table(df):
    """Select and rename columns for pretty terminal printing."""
    display = pd.DataFrame({
        "Ticker": df["ticker"],
        "Market Name": df["market_name"],
        "Yes Bid (¢)": df["yes_bid"].round(1),
        "Yes Ask (¢)": df["yes_ask"].round(1),
        "Spread (¢)": df["spread_cents"].round(1),
        "Implied Prob (%)": df["implied_prob_pct"].round(1),
        "Volume": df["volume"].round(1),
        "Open Interest": df["open_interest"].round(1),
    })
    return display


def print_table(df, title):
    """Print a DataFrame with a nice bordered style (fancy_grid / rounded_outline)."""
    print()
    print(title)
    print()
    # Prefer fancy_grid; fall back to rounded_outline if the style name differs.
    try:
        print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    except ValueError:
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))


def show_main_tables(df):
    """
    Table 1: Top 15 by volume (most actively traded).
    Table 2: Top 10 tightest spreads (best for low-slippage bot entries).
    """
    # --- Table 1: highest volume first ---
    by_volume = df.sort_values("volume", ascending=False).head(15)
    print_table(
        format_display_table(by_volume),
        "=== Table 1: Top 15 Open Markets by Volume ===",
    )

    # --- Table 2: tightest spread, then higher volume as tie-breaker ---
    print()
    print(
        "Why tight spreads matter for trading bots: a smaller bid–ask gap means "
        "less slippage when you buy and sell. Bots that enter and exit quickly "
        "lose less to the spread on every round-trip, so tight markets are usually "
        "better candidates for automated strategies than wide, illiquid ones."
    )
    by_spread = df.sort_values(
        ["spread_cents", "volume"],
        ascending=[True, False],
    ).head(10)
    print_table(
        format_display_table(by_spread),
        "=== Table 2: Top 10 Tightest Spreads (then Volume) ===",
    )


def keyword_filter_loop(df):
    """
    Bonus interactive filter: type a keyword to search title/subtitle/ticker,
    or press Enter to exit. Case-insensitive substring match.
    """
    prompt = (
        "Enter a keyword to filter markets "
        "(e.g., BTC, ETH, Fed, Weather) or press Enter to exit: "
    )
    while True:
        try:
            keyword = input(prompt).strip()
        except EOFError:
            # Non-interactive runs (pipes / smoke tests) hit EOF — exit cleanly.
            print()
            break

        if not keyword:
            break

        # Case-insensitive search across title, subtitle, and ticker.
        key = keyword.lower()
        mask = (
            df["title"].astype(str).str.lower().str.contains(key, na=False)
            | df["subtitle"].astype(str).str.lower().str.contains(key, na=False)
            | df["ticker"].astype(str).str.lower().str.contains(key, na=False)
        )
        matches = df.loc[mask].sort_values("volume", ascending=False)

        if matches.empty:
            print(f"\nNo markets matched keyword: {keyword!r}\n")
        else:
            print_table(
                format_display_table(matches),
                f"=== Markets matching {keyword!r} "
                f"({len(matches)} result(s)) ===",
            )


def main():
    print("Kalshi Live Probability & Spread Scanner")
    print(f"Fetching open markets from: {API_URL}")
    print()

    markets = fetch_open_markets()
    if markets is None:
        return
    if not markets:
        print("No open markets returned by the API.")
        return

    print(f"Received {len(markets)} markets from API.")

    # Load into a DataFrame so we can filter/sort with pandas.
    df = markets_to_dataframe(markets)
    df = filter_and_enrich(df)

    if df.empty:
        print("No markets passed the liquidity filters "
              "(volume > 0 and valid Yes bid/ask).")
        return

    print(f"After liquidity filters: {len(df)} markets remain.")

    # Always print the two ranked tables first (works for non-interactive smoke tests).
    show_main_tables(df)

    # Then offer the optional keyword filter (Enter / EOF exits).
    print()
    keyword_filter_loop(df)
    print("Done. Good luck scanning!")


if __name__ == "__main__":
    main()
