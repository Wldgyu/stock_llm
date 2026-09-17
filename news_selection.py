"""Deterministic news relevance ranking; scores are not return predictions."""
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from html import unescape

COMPANIES = {
    '삼성전자': ('Samsung Electronics', ('samsung', '삼성전자')),
    '엔비디아': ('NVIDIA', ('nvidia', '엔비디아')),
    '인텔': ('Intel', ('intel', '인텔')),
}
TOPICS = {
    '실적·자본정책': (3, r'\b(earnings|profit|revenue|guidance|dividend|buyback|loss|sales)\b|실적|영업이익|매출|배당|자사주'),
    '반도체·수요': (2, r'\b(chips?|semiconductors?|hbm|dram|nand|foundry|gpu|data centers?|demand)\b|반도체|파운드리|수요'),
    '계약·투자': (3, r'\b(contracts?|orders?|acquisition|merger|investment|capex|supply deal)\b|수주|인수|합병|설비투자|공급계약'),
    '규제·생산위험': (3, r'\b(strike|walkout|shutdown|production halt|export|sanctions?|tariffs?|antitrust|recall)\b|파업|생산중단|수출규제|관세|리콜'),
}


def select_news(entries, name, accepted_sources=None, limit=1, now=None):
    """Require company mention, dated recent news, and an impact-related topic."""
    now = now or datetime.now(timezone.utc)
    aliases = COMPANIES.get(name, (name, (name.lower(),)))[1]
    ranked = []
    for entry in entries:
        source = entry.get('source', {}).get('title', '')
        if accepted_sources is not None and source not in accepted_sources:
            continue
        title = entry.get('title', '')
        text = unescape(re.sub(r'<[^>]+>', ' ', title + ' ' + entry.get('summary', ''))).lower()
        if not any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text) for alias in aliases):
            continue
        # Avoid other Samsung subsidiaries being mistaken for Samsung Electronics.
        if name == '삼성전자' and re.search(r'samsung (sdi|biologics|heavy|life|fire|c&t)', text) and 'samsung electronics' not in text:
            continue
        try:
            published = parsedate_to_datetime(entry.get('published', ''))
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            continue
        age = now - published
        if age < timedelta(0) or age > timedelta(days=14):
            continue
        reasons = [label for label, (_, pattern) in TOPICS.items() if re.search(pattern, text)]
        if not reasons:
            continue
        impact = sum(TOPICS[label][0] for label in reasons)
        freshness = 3 if age <= timedelta(days=2) else 1 if age <= timedelta(days=7) else 0
        ranked.append((impact + freshness, published, entry, reasons))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected, seen = [], set()
    for score, _, entry, reasons in ranked:
        key = re.sub(r'\W+', '', entry.get('title', '').lower())
        if key in seen:
            continue
        seen.add(key)
        selected.append((entry, score, reasons))
        if len(selected) >= limit:
            break
    return selected
