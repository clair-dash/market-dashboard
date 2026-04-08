# Market Update Dashboard

A live market dashboard that automatically refreshes with data from free public sources.
Dark theme matching the Gold Dashboard design language.

## What's included

- `index.html` — The dashboard (single HTML file, no build step needed)
- `fetch_data.py` — Python script that pulls live data and writes `data.json`
- `data.json` — Generated data file (created by fetch_data.py)
- `.github/workflows/update-data.yml` — GitHub Actions automation

## Data Sources

| Data | Source | Update |
|------|--------|--------|
| Equities, FX, Commodities, Crypto, P/E | Yahoo Finance | Automatic |
| Yields, Credit Spreads, 2s10s, US CPI | FRED | Automatic |
| Policy Rates (Fed) | FRED | Automatic |
| Policy Rates (SNB, ECB) | Manual config | After rate decisions |
| PMI | Manual config | Monthly |
| CPI (non-US) | Manual config | Monthly |
| Big Mac Index | The Economist GitHub | Automatic (~2x/year) |
| Listed Real Estate | Yahoo Finance (ETF proxies) | Automatic |

## Setup

See the step-by-step guide provided separately.
