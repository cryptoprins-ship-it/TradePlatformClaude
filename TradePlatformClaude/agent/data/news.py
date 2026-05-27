import html
import re
from dataclasses import dataclass
from typing import Callable

import feedparser

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

NewsSource = Callable[[], list[tuple[str, str]]]


@dataclass
class NewsItem:
    headline: str
    summary: str


def sanitize(text: str, max_len: int = 280) -> str:
    text = html.unescape(text)
    text = _CONTROL.sub("", text)
    text = text.replace("```", "'''")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


class NewsFeed:
    def __init__(self, source: NewsSource | None) -> None:
        self._source = source

    def fetch(self) -> list[NewsItem]:
        if self._source is None:
            return []
        return [
            NewsItem(
                headline=sanitize(headline, 160),
                summary=sanitize(summary, 280),
            )
            for headline, summary in self._source()
        ]


def rss_source(url: str, limit: int = 10) -> NewsSource:
    def _fetch() -> list[tuple[str, str]]:
        feed = feedparser.parse(url)
        return [
            (entry.get("title", ""), entry.get("summary", ""))
            for entry in feed.entries[:limit]
        ]

    return _fetch
