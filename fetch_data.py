"""
Market Update Dashboard — Data Fetcher (v2)
Pulls data from free public APIs and writes data.json for the dashboard.

Sources:
  - Yahoo Finance (yfinance): equities, FX, commodities, crypto, P/E, gold miners, DXY
  - FRED (fredapi): US 10Y, German Bund, 2s10s, IG/HY OAS, US CPI, Real Yield
  - SNB data portal: Swiss 10Y yield, SNB policy rate
  - ECB: ECB deposit rate
  - The Economist GitHub: Big Mac Index
  - PMI: manual config (no free API)

Requirements:
  pip install yfinance requests pandas

Usage:
  export FRED_API_KEY="your_key_here"
  python fetch_data.py
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
LOOKBACK_MONTHS = 16

# Yahoo Finance tickers — equity indices
EQUITY_TICKERS = {
    "SMI": "^SSMI",
    "EuroStoxx 50": "^STOXX50E",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "MSCI EM": "EEM",
    "Nikkei 225": "^N225",
}

# ETF proxies for total return (used in rebased charts)
EQUITY_ETF_TICKERS = {
    "SMI": "EWL",         # iShares Switzerland (closest US-listed proxy)
    "EuroStoxx 50": "FEZ",
    "S&P 500": "SPY",
    "NASDAQ": "QQQ",
    "MSCI EM": "EEM",
    "Nikkei 225": "EWJ",  # iShares Japan
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

# Gold miners
MINER_TICKERS = {
    "Agnico Eagle": "AEM",
    "Alamos Gold": "AGI",
    "New Gold": "NGD",
    "Newmont": "NEM",
    "Perpetua": "PPTA",
    "Wesdome": "WDO.TO",
}

# Extra tickers
DXY_TICKER = "DX-Y.NYB"
VIX_TICKER = "^VIX"

# FRED series IDs
FRED_SERIES = {
    "US 10Y": "DGS10",
    "German Bund 10Y": "IRLTLT01DEM156N",
    "US 2s10s Spread": "T10Y2Y",
    "US IG OAS": "BAMLC0A0CM",
    "US HY OAS": "BAMLH0A0HYM2",
    "US CPI YoY": "CPIAUCSL",
    "Fed Funds Rate": "DFEDTARU",
    "US Real Yield 10Y": "DFII10",
}

# PMI manual config
PMI_DATA = {
    "countries": ["US", "Eurozone", "Switz.", "China", "Japan"],
    "current": [52.1, 47.8, 48.5, 50.8, 49.2],
    "prior": [51.5, 48.2, 49.1, 50.2, 48.8],
    "six_months_ago": [49.8, 46.5, 47.2, 49.5, 50.1],
    "last_updated": "2026-04-01",
}

CPI_MANUAL = {
    "Eurozone": {"latest": "2.2%", "prior": "2.4%", "target": "2.0%"},
    "Switz.": {"latest": "0.6%", "prior": "0.8%", "target": "0-2%"},
    "China": {"latest": "0.1%", "prior": "0.3%", "target": "3.0%"},
    "Japan": {"latest": "3.2%", "prior": "2.8%", "target": "2.0%"},
}


def safe_download(ticker, **kwargs):
    """Download with retries and error handling."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, progress=False, auto_adjust=True, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed for {ticker}: {e}")
            if attempt == 2:
                return pd.DataFrame()
    return pd.DataFrame()


def get_quote(ticker):
    """Get current quote using yf.download for reliability (info endpoint is flaky)."""
    try:
        df = safe_download(ticker, period="5d", interval="1d")
        if df.empty:
            return None, None
        price = float(df["Close"].iloc[-1])
        # Try to get P/E from info
        pe = None
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            pe = info.get("trailingPE")
            if pe is not None:
                pe = round(float(pe), 1)
        except Exception:
            pass
        return round(price, 4), pe
    except Exception as e:
        log.error(f"Quote failed for {ticker}: {e}")
        return None, None


def compute_returns(ticker, current_price):
    """Compute 1W, 1M, YTD, 1Y returns from historical data."""
    if current_price is None:
        return None, None, None, None
    try:
        today = datetime.now()
        df = safe_download(ticker, period="13mo", interval="1d")
        if df.empty or len(df) < 5:
            return None, None, None, None
        closes = df["Close"].dropna()

        def get_past(days_ago):
            target_idx = max(0, len(closes) - days_ago - 1)
            if target_idx < 0 or target_idx >= len(closes):
                return None
            return float(closes.iloc[target_idx])

        def pct(past):
            if past is None or past == 0:
                return None
            return round((current_price - past) / past * 100, 2)

        w1 = pct(get_past(5))
        m1 = pct(get_past(21))

        # YTD
        ytd = None
        start_of_year = datetime(today.year, 1, 1)
        ytd_data = closes[closes.index >= pd.Timestamp(start_of_year).tz_localize(closes.index.tz) if closes.index.tz else pd.Timestamp(start_of_year)]
        if len(ytd_data) > 0:
            ytd_start = float(ytd_data.iloc[0])
            if ytd_start != 0:
                ytd = round((current_price - ytd_start) / ytd_start * 100, 2)

        # 1Y
        y1 = pct(get_past(252)) if len(closes) > 252 else pct(float(closes.iloc[0]))

        return w1, m1, ytd, y1
    except Exception as e:
        log.error(f"Returns failed for {ticker}: {e}")
        return None, None, None, None


def fetch_current_data(tickers):
    """Fetch current prices and returns for a dict of {name: ticker}."""
    result = {}
    for name, ticker in tickers.items():
        log.info(f"  Fetching {name} ({ticker})...")
        price, pe = get_quote(ticker)
        w1, m1, ytd, y1 = compute_returns(ticker, price)
        result[name] = {
            "price": price,
            "w1": w1,
            "m1": m1,
            "ytd": ytd,
            "y1": y1,
            "pe": pe,
        }
    return result


def fetch_yahoo_history(tickers, period="16mo", interval="1mo"):
    """Fetch monthly historical close prices."""
    result = {}
    for name, ticker in tickers.items():
        try:
            df = safe_download(ticker, period=period, interval=interval)
            if df.empty:
                result[name] = []
                continue
            closes = df["Close"].dropna()
            result[name] = [
                {"date": d.strftime("%Y-%m-%d"), "close": round(float(v), 4)}
                for d, v in closes.items()
            ]
        except Exception as e:
            log.error(f"History failed for {name} ({ticker}): {e}")
            result[name] = []
    return result


def rebase_to_100(history):
    """Convert raw price history to rebased (first value = 100) monthly data."""
    if not history:
        return []
    all_dates = set()
    for name, series in history.items():
        for point in series:
            all_dates.add(point["date"])
    if not all_dates:
        return []
    sorted_dates = sorted(all_dates)
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


def fetch_fred_series(series_id, lookback_days=365):
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


def fetch_fred_latest(series_id):
    """Get the most recent value from a FRED series."""
    obs = fetch_fred_series(series_id, lookback_days=90)
    if obs:
        try:
            return round(float(obs[0]["value"]), 2)
        except (ValueError, IndexError):
            return None
    return None


def fetch_fred_yields_and_spreads():
    """Fetch yields and credit spreads from FRED."""
    result = {}
    skip = {"US CPI YoY", "Fed Funds Rate", "US Real Yield 10Y"}
    for name, series_id in FRED_SERIES.items():
        if name in skip:
            continue
        try:
            obs = fetch_fred_series(series_id, lookback_days=400)
            if not obs:
                result[name] = {"current": None, "d2w": None, "dytd": None}
                continue
            current = float(obs[0]["value"])
            d2w = None
            for o in obs:
                d = datetime.strptime(o["date"], "%Y-%m-%d")
                if (datetime.now() - d).days >= 12:
                    d2w = round(current - float(o["value"]), 2)
                    break
            dytd = None
            for o in obs:
                d = datetime.strptime(o["date"], "%Y-%m-%d")
                if d.year < datetime.now().year:
                    dytd = round(current - float(o["value"]), 2)
                    break
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
            log.error(f"Failed FRED {name}: {e}")
            result[name] = {"current": None, "d2w": None, "dytd": None}
    return result


def fetch_us_cpi_yoy():
    """Fetch US CPI YoY from FRED."""
    try:
        obs = fetch_fred_series("CPIAUCSL", lookback_days=500)
        if len(obs) < 14:
            return {"latest": None, "prior": None, "target": "2.0%"}
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


def fetch_big_mac_index():
    """Fetch Big Mac Index from The Economist's GitHub."""
    url = "https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/output-data/big-mac-full-index.csv"
    try:
        df = pd.read_csv(url)
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].copy()
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


def fetch_policy_rates():
    """Fetch policy rates."""
    fed_rate = fetch_fred_latest("DFEDTARU")
    snb_rate = 0.25
    ecb_rate = 2.75
    return [
        {"name": "SNB", "policy": snb_rate, "yield10": None},
        {"name": "Fed", "policy": fed_rate or 4.50, "yield10": None},
        {"name": "ECB", "policy": ecb_rate, "yield10": None},
    ]


def fetch_swiss_10y():
    """Fetch Swiss 10Y from SNB data portal."""
    try:
        url = "https://data.snb.ch/api/cube/rendoblim/data/csv/en"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            for line in reversed(lines):
                parts = line.split(";")
                if len(parts) >= 3:
                    try:
                        val = float(parts[-1])
                        return {"current": f"{val:.2f}%", "source": "SNB"}
                    except ValueError:
                        continue
        return {"current": "0.52%", "source": "manual"}
    except Exception as e:
        log.warning(f"SNB fetch failed: {e}")
        return {"current": "0.52%", "source": "manual"}


def build_data():
    """Main function: fetch all data and build the JSON structure."""
    log.info("Starting data fetch...")
    timestamp = datetime.now().isoformat()

    # 1. Equity rebased
    log.info("Fetching equity history...")
    equity_history = fetch_yahoo_history(EQUITY_TICKERS)
    equity_rebased = rebase_to_100(equity_history)

    # 2. Equity current
    log.info("Fetching equity current data...")
    equity_current = fetch_current_data(EQUITY_TICKERS)

    # 3. FX
    log.info("Fetching FX history...")
    fx_history = fetch_yahoo_history(FX_TICKERS)
    fx_rebased = rebase_to_100(fx_history)

    log.info("Fetching FX current data...")
    fx_current = fetch_current_data(FX_TICKERS)

    # 4. Commodities
    log.info("Fetching commodities & crypto...")
    commodity_current = fetch_current_data(COMMODITY_TICKERS)

    # 5. VIX
    log.info("Fetching VIX...")
    vix_price, _ = get_quote(VIX_TICKER)
    vix = round(vix_price, 1) if vix_price else None

    # 6. DXY
    log.info("Fetching DXY...")
    dxy_price, _ = get_quote(DXY_TICKER)
    dxy_w1, dxy_m1, dxy_ytd, dxy_y1 = compute_returns(DXY_TICKER, dxy_price)
    dxy_data = {
        "price": round(dxy_price, 2) if dxy_price else None,
        "w1": dxy_w1, "m1": dxy_m1, "ytd": dxy_ytd, "y1": dxy_y1,
    }

    # 7. Gold miners
    log.info("Fetching gold miners...")
    miners_data = {}
    for name, ticker in MINER_TICKERS.items():
        log.info(f"  Miner: {name} ({ticker})...")
        price, pe = get_quote(ticker)
        w1, m1, ytd, y1 = compute_returns(ticker, price)
        # Market cap
        mcap = None
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            mcap = info.get("marketCap")
            if mcap:
                mcap = round(mcap / 1e9, 1)  # billions
        except Exception:
            pass
        miners_data[name] = {
            "price": price, "pe": pe,
            "w1": w1, "m1": m1, "ytd": ytd, "y1": y1,
            "mcap": mcap,
        }

    # 8. FRED yields & spreads
    log.info("Fetching FRED yields & spreads...")
    yields_spreads = fetch_fred_yields_and_spreads()

    # Swiss 10Y
    swiss_10y = fetch_swiss_10y()
    yields_spreads["Swiss 10Y"] = {
        "current": swiss_10y["current"], "d2w": "N/A", "dytd": "N/A",
    }

    # 9. Real yield
    log.info("Fetching real yield...")
    real_yield = fetch_fred_latest("DFII10")
    real_yield_data = {"value": real_yield}

    # 10. Policy rates
    log.info("Fetching policy rates...")
    policy_rates = fetch_policy_rates()
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

    # 11. CPI
    log.info("Fetching CPI...")
    us_cpi = fetch_us_cpi_yoy()
    cpi_data = {
        "countries": ["US", "Eurozone", "Switz.", "China", "Japan"],
        "latest": [us_cpi["latest"]] + [CPI_MANUAL[c]["latest"] for c in ["Eurozone", "Switz.", "China", "Japan"]],
        "prior": [us_cpi["prior"]] + [CPI_MANUAL[c]["prior"] for c in ["Eurozone", "Switz.", "China", "Japan"]],
        "target": [us_cpi["target"]] + [CPI_MANUAL[c]["target"] for c in ["Eurozone", "Switz.", "China", "Japan"]],
    }

    # 12. Big Mac
    log.info("Fetching Big Mac Index...")
    big_mac = fetch_big_mac_index()

    # 13. Listed RE
    log.info("Fetching RE data...")
    re_tickers = {
        "CH": "SRECHA.SW",
        "EU": "IPRP.L",
        "US": "VNQ",
        "APAC": "1659.T",
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
        "dxy": dxy_data,
        "gold_miners": miners_data,
        "real_yield": real_yield_data,
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


def sanitize(obj):
    """Recursively replace NaN/Infinity with None for valid JSON."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def main():
    try:
        data = build_data()
        data = sanitize(data)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"Data written to {OUTPUT_PATH}")
        log.info(f"Timestamp: {data['timestamp']}")

        eq_count = sum(1 for v in data["equity"]["current"].values() if v.get("price") is not None)
        fx_count = sum(1 for v in data["fx"]["current"].values() if v.get("price") is not None)
        cmd_count = sum(1 for v in data["commodities"].values() if v.get("price") is not None)
        miner_count = sum(1 for v in data["gold_miners"].values() if v.get("price") is not None)
        log.info(f"Equities: {eq_count}/{len(EQUITY_TICKERS)}, FX: {fx_count}/{len(FX_TICKERS)}, "
                 f"Commodities: {cmd_count}/{len(COMMODITY_TICKERS)}, Miners: {miner_count}/{len(MINER_TICKERS)}")
        log.info(f"VIX: {data['vix']}, DXY: {data['dxy']['price']}")
    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
