import re
import time
import csv
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, asdict

import pandas as pd
import phonenumbers
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ===================== CONFIG =====================
START_URL = "https://www.zomato.com/hyderabad/jubilee-hills-restaurants"
MAX_RESTAURANTS = 50              # cap for how many restaurant pages to visit
PAGE_LOAD_TIMEOUT_MS = 60000
PER_PAGE_SLEEP_SEC = 2.0          # polite delay between restaurant page fetches
SCROLL_STEPS = 8                  # how many times to scroll on the listing page
SCROLL_PAUSE_SEC = 1.0
# ==================================================


@dataclass
class RestaurantRow:
    name: str
    phone: str
    address: str
    url: str

TEL_RE = re.compile(r'(?:\+91[-\s]?)?[6-9]\d{9}')

def normalize_in_phone(num_raw: str) -> str | None:
    """Return E.164 +91XXXXXXXXXX if valid, else None."""
    try:
        pn = phonenumbers.parse(num_raw, "IN")
        if phonenumbers.is_valid_number(pn):
            return phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164)
    except Exception:
        pass
    return None

def extract_phones_from_html(html: str) -> list[str]:
    """Find phone-like strings and normalize; also look for tel: links."""
    soup = BeautifulSoup(html, "lxml")

    # 1) tel: links
    tel_links = []
    for a in soup.select('a[href^="tel:"]'):
        href = a.get("href", "")
        tel_links.append(href.replace("tel:", "").strip())

    # 2) phone-like text
    text_matches = TEL_RE.findall(soup.get_text(" "))

    raw_candidates = set([*tel_links, *text_matches])

    normalized = []
    seen = set()
    for raw in raw_candidates:
        # strip non-digits except leading + 
        cleaned = re.sub(r"[^\d+]", "", raw)
        norm = normalize_in_phone(cleaned)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized

def extract_name_and_address(html: str) -> tuple[str, str]:
    """Try common spots for restaurant name/address; fall back gracefully."""
    soup = BeautifulSoup(html, "lxml")

    # Name candidates: <h1>, meta og:title, title tag
    name = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        name = h1.get_text(strip=True)
    if not name:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            name = og["content"].strip()
    if not name and soup.title and soup.title.string:
        name = soup.title.string.strip()

    # Address candidates: look for common classes/labels or microdata
    address = ""
    # microdata
    addr_nodes = soup.select('[itemprop="address"]') or soup.select('[data-testid*="address"]')
    for node in addr_nodes:
        txt = node.get_text(" ", strip=True)
        if txt and len(txt) > 10:
            address = txt
            break

    # heuristics
    if not address:
        for tag in soup.find_all(["p", "div", "span"]):
            txt = tag.get_text(" ", strip=True)
            if not txt:
                continue
            # crude heuristic: contains road/plot/floor/Hyderabad etc.
            if any(k in txt.lower() for k in ["road", "rd", "plot", "floor", "hills", "hyderabad", "telangana", "lane", "street", "nagar"]):
                if 20 <= len(txt) <= 240:
                    address = txt
                    break

    return name, address

def collect_restaurant_links(page, start_url: str, limit: int) -> list[str]:
    """
    On the listing page, scroll a bit and collect unique restaurant detail links.
    We’ll later visit /info for reliable contact details.
    """
    page.goto(start_url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")

    # Gentle scrolling to load lazy content
    for _ in range(SCROLL_STEPS):
        page.mouse.wheel(0, 2400)
        time.sleep(SCROLL_PAUSE_SEC)

    html = page.content()
    soup = BeautifulSoup(html, "lxml")

    anchors = soup.select("a[href]")
    links = []
    seen = set()
    for a in anchors:
        href = a.get("href", "")
        # Keep only Zomato internal links within Hyderabad that look like restaurant detail pages
        if not href.startswith("http"):
            full = urljoin(start_url, href)
        else:
            full = href

        # Very light filtering to avoid category pages; accept /hyderabad/… but not /info yet
        try:
            parsed = urlparse(full)
            if "zomato.com" not in parsed.netloc:
                continue
            if "/hyderabad/" not in parsed.path:
                continue
            if any(seg in parsed.path for seg in ["/order", "/photos", "/reviews", "/book", "/menu", "/info"]):
                # We'll append /info ourselves later
                continue
            # paths with two+ segments often are restaurant slugs; keep a broad net
            if full not in seen:
                seen.add(full)
                links.append(full)
        except Exception:
            continue

        if len(links) >= limit * 2:  # collect a bit extra; some may be non-restaurant pages
            break

    # De-dup while preserving order
    unique_links = []
    seen2 = set()
    for u in links:
        if u not in seen2:
            seen2.add(u)
            unique_links.append(u)

    return unique_links[:max(limit, 10)]  # keep at least some

def to_info_url(url: str) -> str:
    """Ensure we hit the /info page (where phone/address usually live)."""
    return url.rstrip("/") + "/info"

def scrape():
    rows: list[RestaurantRow] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()  # fresh context per run
        page = context.new_page()

        print("Collecting restaurant links from listing…")
        candidates = collect_restaurant_links(page, START_URL, MAX_RESTAURANTS * 2)

        # Visit each candidate’s /info page until we accumulate MAX_RESTAURANTS
        count = 0
        for base_url in candidates:
            if count >= MAX_RESTAURANTS:
                break

            info_url = to_info_url(base_url)
            try:
                page.goto(info_url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")
            except PWTimeout:
                print(f"Timeout loading: {info_url}")
                time.sleep(PER_PAGE_SLEEP_SEC)
                continue
            except Exception as e:
                print(f"Error loading {info_url}: {e}")
                time.sleep(PER_PAGE_SLEEP_SEC)
                continue

            # Some pages lazy-load; small wait helps
            page.wait_for_timeout(1500)

            # Try clicking any obvious “Show contact” buttons (best-effort, safe to ignore errors)
            for selector in [
                'text="Call"',          # generic
                'button:has-text("Call")',
                '[data-testid*="call"]',
                'text="Show Contact"',
                'button:has-text("Show")',
            ]:
                try:
                    btns = page.locator(selector)
                    if btns.count() > 0:
                        btns.first.click(timeout=2000)
                        page.wait_for_timeout(800)
                        break
                except Exception:
                    pass

            html = page.content()
            name, address = extract_name_and_address(html)
            phones = extract_phones_from_html(html)

            if not name and not phones and not address:
                # Probably not a restaurant detail page
                time.sleep(PER_PAGE_SLEEP_SEC)
                continue

            phone_join = "; ".join(phones) if phones else ""
            rows.append(RestaurantRow(
                name=name or "",
                phone=phone_join,
                address=address or "",
                url=info_url
            ))
            count += 1
            print(f"[{count}/{MAX_RESTAURANTS}] {name} | {phone_join}")

            time.sleep(PER_PAGE_SLEEP_SEC)

        browser.close()

    # Deduplicate by (name, phone, address)
    dedup = []
    seen = set()
    for r in rows:
        key = (r.name.strip().lower(), r.phone, r.address.strip().lower())
        if key not in seen:
            seen.add(key)
            dedup.append(r)

    df = pd.DataFrame([asdict(r) for r in dedup])
    out_csv = "restaurants_jubilee_hills.csv"
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSaved {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    scrape()
