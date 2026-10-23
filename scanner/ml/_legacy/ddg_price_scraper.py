"""
scanner/ml/_legacy/ddg_price_scraper.py — Archived DuckDuckGo online pharmacy scraper.
Preserved for reference. Disabled in MediScan offline mode.
"""

import re
import time
import logging
import urllib.parse
from typing import List, Optional
import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 1.5
DEFAULT_MAX_RETRIES = 0


class DirectPharmacySearchProvider:
    """
    Public product lookup across Indian pharmacy directories via DuckDuckGo HTML.
    Preserved in _legacy.
    """
    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS, max_retries: int = DEFAULT_MAX_RETRIES):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MediScanPriceVerifier/2.0 (Clinical-Research; Ethical-Price-Audit)",
            "Accept": "application/json, text/html",
        })

    def search(self, query: str) -> List[dict]:
        results = []
        clean_q = re.sub(r'[^a-zA-Z0-9\s]', ' ', query).strip()
        if not clean_q:
            return results

        for attempt in range(self.max_retries + 1):
            try:
                search_url = "https://html.duckduckgo.com/html/"
                data = {"q": f"{clean_q} site:1mg.com OR site:pharmeasy.in OR site:apollopharmacy.in OR site:truemeds.in"}
                resp = self.session.post(search_url, data=data, timeout=self.timeout)

                if resp.status_code == 429:
                    logger.warning("Search provider received HTTP 429 (rate limited)")
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    return results

                if resp.status_code in (500, 502, 503, 504):
                    logger.warning("Search provider received HTTP %d", resp.status_code)
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    return results

                if resp.status_code == 200:
                    results = self._parse_ddg_html(resp.text)
                    return results
            except Exception as exc:
                logger.warning("Direct pharmacy search error: %s", exc)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return results

        return results

    def _parse_ddg_html(self, html: str) -> List[dict]:
        listings = []
        snippets = re.findall(
            r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>.*?</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        for url, raw_snip in snippets[:6]:
            url_clean = url.strip()
            if 'uddg=' in url_clean:
                m_u = re.search(r'uddg=([^&]+)', url_clean)
                if m_u:
                    url_clean = urllib.parse.unquote(m_u.group(1))

            snippet = re.sub(r'<[^>]+>', '', raw_snip).strip()
            source_name = "Online Pharmacy Listing"
            if "1mg.com" in url_clean:
                source_name = "1mg"
            elif "pharmeasy.in" in url_clean:
                source_name = "PharmEasy"
            elif "apollopharmacy.in" in url_clean:
                source_name = "Apollo Pharmacy"
            elif "truemeds.in" in url_clean:
                source_name = "Truemeds"

            listings.append({
                "snippet": snippet,
                "url": url_clean,
                "source_name": source_name,
            })
        return listings
