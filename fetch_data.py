"""
Market Update Dashboard — Data Fetcher
Pulls data from free public APIs and writes data.json for the React frontend.

Sources:
  - Yahoo Finance (yfinance): equities, FX, commodities, crypto, P/E
  - FRED (fredapi): US 10Y, German Bund, 2s10s, IG/HY OAS, US CPI
  - SNB data portal: Swiss 10Y yield, SNB policy rate
  - ECB: ECB deposit rate
  - The Economist GitHub: Big Mac Index
  - PMI: manual config (no free API)

Requirements:
  pip install yfinance requests pandas

Usage:
  # Set your free FRED API key (get one at https://fred.stlouisfed.org/docs/api/api_key.html)
  export FRED_API_KEY="your_key_here"
  python fetch_data.py

  # Or pass inline:
  FRED_API_KEY=your_key python fetch_data.py

Schedule via cron or GitHub Actions for automatic updates.
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Configuration ───
OUTPUT_PATH = Path(__file__).parent / "data.json"
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
LOOKBACK_MONTHS = 16  # for rebased charts

# Yahoo Finance tickers
EQUITY_TICKERS = {
    "SMI": "^SSMI",
    "EuroStoxx 50": "^STOXX50E",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "MSCI EM": "EEM",
    "Nikkei 225": "^N225",
}

FX_TICKERS = {
    "EUR/CHF": "EURCHF=X",
    "USD/CHF": "USDCHF=X",
    "GBP/CHF": "GBPCHF=X",
    "JPY/CHF": "JPYCHF=X",
}

COMMODITY_TICKERS = {
    "Brent": "BZ=F",
    "Nat Gas": "NG=F",
    "Copper": "HG=F",
    "Gold": "GC=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}

# FRED series IDs
FRED_SERIES = {
    "US 10Y": "DGS10",
    "German Bund 10Y": "IRLTLT01DEM156N",
    "US 2s10s Spread": "T10Y2Y",
    "US IG OAS": "BAMLC0A0CM",
    "US HY OAS": "BAMLH0A0HYM2",
    "US CPI YoY": "CPIAUCSL",
    "Fed Funds Rate": "DFEDTARU",
}

# PMI manual config — update these monthly
PMI_DATA = {
    "countries": ["US", "Eurozone", "Switz.", "China", "Japan"],
    "current": [52.1, 47.8, 48.5, 50.8, 49.2],
    "prior": [51.5, 48.2, 49.1, 50.2, 48.8],
    "six_months_ago": [49.8, 46.5, 47.2, 49.5, 50.1],
    "last_updated": "2026-04-01",
}

# CPI manual overrides for non-FRED sources
CPI_MANUAL = {
    "Eurozone": {"latest": "2.2%", "prior": "2.4%", "target": "2.0%"},
    "Switz.": {"latest": "0.6%", "prior": "0.8%", "target": "0-2%"},
    "China": {"latest": "0.1%", "prior": "0.3%", "target": "3.0%"},
    "Japan": {"latest": "3.2%", "prior": "2.8%", "target": "2.0%"},
}


def fetch_yahoo_history(tickers: dict, period: str = "16mo", interval: str = "1mo") -> dict:
    """Fetch monthly historical close prices from Yahoo Finance."""
    result = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                log.warning(f"No data for {name} ({ticker})")
                result[name] = []
                continue
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df["Close"].dropna()
            result[name] = [
                {"date": d.strftime("%Y-%m-%d"), "close": round(float(v), 4)}
                for d, v in closes.items()
            ]
        except Exception as e:
            log.error(f"Failed to fetch {name} ({ticker}): {e}")
            result[name] = []
    return result


def fetch_yahoo_current(tickers: dict) -> dict:
    """Fetch current price, 1W/1M/YTD/1Y returns, and trailing P/E from Yahoo Finance."""
    result = {}
    today = datetime.now()
    start_ytd = datetime(today.year, 1, 1)

    for name, ticker in tickers.items():
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            current_price = info.get("regularMarketPrice") or info.get("previousClose")
            if current_price is None:
                hist = tk.history(period="5d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])

            if current_price is None:
                log.warning(f"No current price for {name}")
                result[name] = {"price": None, "w1": None, "m1": None, "ytd": None, "y1": None, "pe": None}
                continue

            # Historical prices for return calculation
            hist_1y = tk.history(period="1y", auto_adjust=True)
            hist_ytd = tk.history(start=start_ytd, auto_adjust=True)

            def pct_change(current, past_series, offset_days):
                if past_series.empty:
                    return None
                target = today - timedelta(days=offset_days)
                # Localize target to match pandas index timezone if needed
                if past_series.index.tz is not None:
                    import pytz
                    target = past_series.index.tz.localize(target) if target.tzinfo is None else target
                try:
                    idx = past_series.index.get_indexer([target], method="nearest")[0]
                except Exception:
                    return None
                if idx < 0 or idx >= len(past_series):
                    return None
                past_price = float(past_series.iloc[idx])
                if past_price == 0:
                    return None
                return round((current - past_price) / past_price * 100, 2)

            w1 = pct_change(current_price, hist_1y["Close"], 7)
            m1 = pct_change(current_price, hist_1y["Close"], 30)

            ytd = None
            if not hist_ytd.empty:
                ytd_start = float(hist_ytd["Close"].iloc[0])
                if ytd_start != 0:
                    ytd = round((current_price - ytd_start) / ytd_start * 100, 2)

            y1 = None
            if not hist_1y.empty:
                y1_start = float(hist_1y["Close"].iloc[0])
                if y1_start != 0:
                    y1 = round((current_price - y1_start) / y1_start * 100, 2)

            pe = info.get("trailingPE")
            if pe is not None:
                pe = round(float(pe), 1)

            result[name] = {
                "price": round(current_price, 4),
                "w1": w1,
                "m1": m1,
                "ytd": ytd,
                "y1": y1,
                "pe": pe,
            }
        except Exception as e:
            log.error(f"Failed current data for {name} ({ticker}): {e}")
            result[name] = {"price": None, "w1": None, "m1": None, "ytd": None, "y1": None, "pe": None}
    return result


def rebase_to_100(history: dict) -> list:
    """Convert raw price history to rebased (first value = 100) monthly data."""
    if not history:
        return []

    # Find common date range
    all_dates = set()
    for name, series in history.items():
        for point in series:
            all_dates.add(point["date"])
    if not all_dates:
        return []
    sorted_dates = sorted(all_dates)

    # Build lookup
    lookup = {}
    for name, series in history.items():
        lookup[name] = {p["date"]: p["close"] for p in series}

    result = []
    base_values = {}
    for date in sorted_dates:
        row = {"month": datetime.strptime(date, "%Y-%m-%d").strftime("%b-%y")}
        for name in history:
            val = lookup[name].get(date)
            if val is None:
                continue
            if name not in base_values:
                base_values[name] = val
            base = base_values[name]
            row[name] = round(val / base * 100, 2) if base != 0 else 100
        result.append(row)
    return result


def fetch_fred_series(series_id: str, lookback_days: int = 365) -> list:
    """Fetch a FRED data series."""
    if not FRED_API_KEY:
        log.warning(f"No FRED_API_KEY set, skipping {series_id}")
        return []
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={FRED_API_KEY}"
            f"&observation_start={start}&observation_end={end}"
            f"&file_type=json&sort_order=desc&limit=30"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        return [
            {"date": o["date"], "value": o["value"]}
            for o in observations
            if o["value"] != "."
        ]
    except Exception as e:
        log.error(f"FRED fetch failed for {series_id}: {e}")
        return []


def fetch_fred_latest(series_id: str) -> float | None:
    """Get the most recent value from a FRED series."""
    obs = fetch_fred_series(series_id, lookback_days=90)
    if obs:
        try:
            return round(float(obs[0]["value"]), 2)
        except (ValueError, IndexError):
            return None
    return None


def fetch_fred_yields_and_spreads() -> dict:
    """Fetch yields and credit spreads from FRED with 2-week and YTD deltas."""
    result = {}
    for name, series_id in FRED_SERIES.items():
        if name == "US CPI YoY" or name == "Fed Funds Rate":
            continue
        try:
            obs = fetch_fred_series(series_id, lookback_days=400)
            if not obs:
                result[name] = {"current": None, "d2w": None, "dytd": None}
                continue

            current = float(obs[0]["value"])

            # Find ~2 weeks ago
            d2w = None
            for o in obs:
                d = datetime.strptime(o["date"], "%Y-%m-%d")
                if (datetime.now() - d).days >= 12:
                    d2w = round(current - float(o["value"]), 2)
                    break

            # Find YTD start
            dytd = None
            for o in obs:
                d = datetime.strptime(o["date"], "%Y-%m-%d")
                if d.year < datetime.now().year:
                    dytd = round(current - float(o["value"]), 2)
                    break

            # Format based on type
            is_spread = "OAS" in name or "Spread" in name
            if is_spread:
                fmt_current = f"{round(current)} bp"
                fmt_d2w = f"{d2w:+.0f} bp" if d2w is not None else "N/A"
                fmt_dytd = f"{dytd:+.0f} bp" if dytd is not None else "N/A"
            else:
                fmt_current = f"{current:.2f}%"
                fmt_d2w = f"{round(d2w * 100):+d} bp" if d2w is not None else "N/A"
                fmt_dytd = f"{round(dytd * 100):+d} bp" if dytd is not None else "N/A"

            result[name] = {"current": fmt_current, "d2w": fmt_d2w, "dytd": fmt_dytd}
        except Exception as e:
            log.error(f"Failed processing FRED {name}: {e}")
            result[name] = {"current": None, "d2w": None, "dytd": None}
    return result


def fetch_us_cpi_yoy() -> dict:
    """Fetch US CPI YoY from FRED (CPIAUCSL is monthly index, we compute YoY %)."""
    try:
        obs = fetch_fred_series("CPIAUCSL", lookback_days=500)
        if len(obs) < 14:
            log.warning(f"Not enough CPI observations ({len(obs)}), need at least 14")
            return {"latest": None, "prior": None, "target": "2.0%"}
        # obs is sorted desc
        latest_val = float(obs[0]["value"])
        prior_val = float(obs[1]["value"])
        yoy_latest_base = float(obs[12]["value"])
        yoy_prior_base = float(obs[13]["value"]) if len(obs) > 13 else None

        latest_yoy = round((latest_val / yoy_latest_base - 1) * 100, 1)
        prior_yoy = round((prior_val / yoy_prior_base - 1) * 100, 1) if yoy_prior_base else None

        return {
            "latest": f"{latest_yoy}%",
            "prior": f"{prior_yoy}%" if prior_yoy else "N/A",
            "target": "2.0%",
        }
    except Exception as e:
        log.error(f"Failed US CPI: {e}")
        return {"latest": None, "prior": None, "target": "2.0%"}


def fetch_big_mac_index() -> list:
    """Fetch Big Mac Index from The Economist's GitHub."""
    url = "https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/output-data/big-mac-full-index.csv"
    try:
        df = pd.read_csv(url)
        # Get latest date
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].copy()

        # Filter to our target countries
        country_map = {
            "Switzerland": "Switzerland",
            "Euro area": "Eurozone",
            "Britain": "UK",
            "Japan": "Japan",
            "China": "China",
        }

        result = []
        for bm_name, display_name in country_map.items():
            row = latest[latest["name"] == bm_name]
            if not row.empty:
                # The column name varies across CSV versions
                usd_raw = None
                for col in ["USD_raw", "dollar_ppp", "dollar_raw"]:
                    if col in row.columns:
                        usd_raw = row.iloc[0].get(col)
                        break
                if usd_raw is not None and pd.notna(usd_raw):
                    result.append({
                        "country": display_name,
                        "value": round(float(usd_raw) * 100, 1),
                    })
        return result
    except Exception as e:
        log.error(f"Failed Big Mac Index: {e}")
        return []


def fetch_policy_rates() -> list:
    """Fetch policy rates. Fed from FRED, SNB and ECB are slow-changing so semi-manual."""
    fed_rate = fetch_fred_latest("DFEDTARU")

    # SNB and ECB rates change infrequently — update manually or scrape
    snb_rate = 0.25  # As of March 2025 decision
    ecb_rate = 2.75  # As of Jan 2025 decision

    return [
        {"name": "SNB", "policy": snb_rate, "yield10": None},
        {"name": "Fed", "policy": fed_rate or 4.50, "yield10": None},
        {"name": "ECB", "policy": ecb_rate, "yield10": None},
    ]


def fetch_swiss_10y() -> dict:
    """Fetch Swiss 10Y from SNB data portal."""
    # SNB provides data via their API but format is complex.
    # Fallback: use the Yahoo Finance proxy for Swiss 10Y govt bond
    try:
        # There's no great free ticker for Swiss 10Y on Yahoo.
        # We'll try to get it from SNB's data portal
        url = "https://data.snb.ch/api/cube/rendoblim/data/csv/en"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            # Parse the last line for most recent 10Y yield
            for line in reversed(lines):
                parts = line.split(";")
                if len(parts) >= 3:
                    try:
                        val = float(parts[-1])
                        return {"current": f"{val:.2f}%", "source": "SNB"}
                    except ValueError:
                        continue
        log.warning("Could not parse SNB 10Y data, using fallback")
        return {"current": "0.52%", "source": "manual"}
    except Exception as e:
        log.warning(f"SNB fetch failed: {e}, using manual fallback")
        return {"current": "0.52%", "source": "manual"}


def build_data() -> dict:
    """Main function: fetch all data and build the JSON structure."""
    log.info("Starting data fetch...")
    timestamp = datetime.now().isoformat()

    # 1. Equity data
    log.info("Fetching equity history...")
    equity_history = fetch_yahoo_history(EQUITY_TICKERS)
    equity_rebased = rebase_to_100(equity_history)

    log.info("Fetching equity current data...")
    equity_current = fetch_yahoo_current(EQUITY_TICKERS)

    # 2. FX data
    log.info("Fetching FX history...")
    fx_history = fetch_yahoo_history(FX_TICKERS)
    fx_rebased = rebase_to_100(fx_history)

    log.info("Fetching FX current data...")
    fx_current = fetch_yahoo_current(FX_TICKERS)

    # 3. Commodities & Crypto
    log.info("Fetching commodities & crypto...")
    commodity_current = fetch_yahoo_current(COMMODITY_TICKERS)

    # 4. VIX
    log.info("Fetching VIX...")
    vix_data = fetch_yahoo_current({"VIX": "^VIX"})
    vix = vix_data.get("VIX", {}).get("price")

    # 5. FRED yields & spreads
    log.info("Fetching FRED yields & spreads...")
    yields_spreads = fetch_fred_yields_and_spreads()

    # Add Swiss 10Y
    swiss_10y = fetch_swiss_10y()
    yields_spreads["Swiss 10Y"] = {
        "current": swiss_10y["current"],
        "d2w": "N/A",  # Would need historical SNB data
        "dytd": "N/A",
    }

    # 6. Policy rates
    log.info("Fetching policy rates...")
    policy_rates = fetch_policy_rates()

    # Fill in 10Y yields from FRED data
    fred_yield_map = {
        "SNB": yields_spreads.get("Swiss 10Y", {}).get("current"),
        "Fed": yields_spreads.get("US 10Y", {}).get("current"),
        "ECB": yields_spreads.get("German Bund 10Y", {}).get("current"),
    }
    for rate in policy_rates:
        yield_str = fred_yield_map.get(rate["name"])
        if yield_str:
            try:
                rate["yield10"] = float(yield_str.replace("%", "").replace(" bp", ""))
            except (ValueError, AttributeError):
                pass

    # 7. CPI
    log.info("Fetching CPI data...")
    us_cpi = fetch_us_cpi_yoy()
    cpi_data = {
        "countries": ["US", "Eurozone", "Switz.", "China", "Japan"],
        "latest": [us_cpi["latest"]] + [CPI_MANUAL[c]["latest"] for c in ["Eurozone", "Switz.", "China", "Japan"]],
        "prior": [us_cpi["prior"]] + [CPI_MANUAL[c]["prior"] for c in ["Eurozone", "Switz.", "China", "Japan"]],
        "target": [us_cpi["target"]] + [CPI_MANUAL[c]["target"] for c in ["Eurozone", "Switz.", "China", "Japan"]],
    }

    # 8. Big Mac Index
    log.info("Fetching Big Mac Index...")
    big_mac = fetch_big_mac_index()

    # 9. Listed RE (via Yahoo Finance ETFs as proxies)
    log.info("Fetching RE data...")
    re_tickers = {
        "CH": "SRECHA.SW",   # SXI Real Estate Funds
        "EU": "IPRP.L",     # iShares European Property
        "US": "VNQ",        # Vanguard Real Estate
        "APAC": "1659.T",   # iShares Asia REIT
    }
    re_history = fetch_yahoo_history(re_tickers, period="16mo", interval="1mo")
    re_rebased = rebase_to_100(re_history)

    # Build output
    data = {
        "timestamp": timestamp,
        "demo": False,
        "equity": {
            "rebased": equity_rebased,
            "current": equity_current,
        },
        "fx": {
            "rebased": fx_rebased,
            "current": fx_current,
        },
        "commodities": commodity_current,
        "vix": vix,
        "rates": {
            "policy": policy_rates,
            "yields_spreads": yields_spreads,
        },
        "pmi": PMI_DATA,
        "cpi": cpi_data,
        "real_estate": {
            "rebased": re_rebased,
        },
        "big_mac": big_mac,
    }

    return data


def main():
    try:
        data = build_data()
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"Data written to {OUTPUT_PATH}")
        log.info(f"Timestamp: {data['timestamp']}")

        # Summary
        eq_count = sum(1 for v in data["equity"]["current"].values() if v.get("price") is not None)
        fx_count = sum(1 for v in data["fx"]["current"].values() if v.get("price") is not None)
        cmd_count = sum(1 for v in data["commodities"].values() if v.get("price") is not None)
        log.info(f"Equities: {eq_count}/{len(EQUITY_TICKERS)}, FX: {fx_count}/{len(FX_TICKERS)}, Commodities: {cmd_count}/{len(COMMODITY_TICKERS)}")

    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
