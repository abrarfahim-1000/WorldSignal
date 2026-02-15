"""RSS feed and API ingestion pipeline."""
from datetime import datetime
from typing import List, Dict, Literal
from email.utils import parsedate_to_datetime
import feedparser
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.db.models import NewsItem, SessionLocal
from app.embeddings.base import get_embedder
from app.rag.vector_store import get_vector_store


SOURCES = [
    {"type": "rss", "url": "https://finance.yahoo.com/news/rssindex", "category": "finance"},
    {"type": "rss", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "category": "finance"},
    {"type": "rss", "url": "https://feeds.bloomberg.com/markets/news.rss", "category": "finance"},
    {"type": "rss", "url": "https://www.reuters.com/rssFeed/worldNews", "category": "geopolitics"},
    {"type": "rss", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "category": "geopolitics"},
    {"type": "rss", "url": "https://www.theguardian.com/world/rss", "category": "geopolitics"},
    {"type": "api", "name": "newsapi", "category": "finance", "endpoint": "top-headlines"},
]


class IngestionPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
    
    def fetch_rss(self, url: str) -> List[Dict]:
        feed = feedparser.parse(url)
        return [
            {
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "content": entry.get("summary", entry.get("description", "")),
                "published_at": self._parse_date(str(entry.get("published", "")))
            }
            for entry in feed.entries
        ]
    
    def fetch_api(self, name: str, **params) -> List[Dict]:
        if name == "newsapi" and self.settings.newsapi_key:
            return self._fetch_newsapi(params.get("endpoint", "top-headlines"))
        elif name == "guardian" and self.settings.guardian_api_key:
            return self._fetch_guardian(params.get("section", "world"))
        elif name == "nyt" and self.settings.nyt_api_key:
            return self._fetch_nyt(params.get("section", "world"))
        elif name == "finnhub" and self.settings.finnhub_api_key:
            return self._fetch_finnhub()
        return []
    
    def _fetch_newsapi(self, endpoint: str) -> List[Dict]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"https://newsapi.org/v2/{endpoint}",
                params={"apiKey": self.settings.newsapi_key, "country": "us", "pageSize": 20}
            )
            articles = resp.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "content": a.get("description", "") + " " + a.get("content", ""),
                    "published_at": self._parse_iso_date(a.get("publishedAt"))
                }
                for a in articles if a.get("url")
            ]
    
    def _fetch_guardian(self, section: str) -> List[Dict]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                "https://content.guardianapis.com/search",
                params={"api-key": self.settings.guardian_api_key, "section": section, "show-fields": "body", "page-size": 20}
            )
            articles = resp.json().get("response", {}).get("results", [])
            return [
                {
                    "title": a.get("webTitle", ""),
                    "url": a.get("webUrl", ""),
                    "content": a.get("fields", {}).get("body", ""),
                    "published_at": self._parse_iso_date(a.get("webPublicationDate"))
                }
                for a in articles
            ]
    
    def _fetch_nyt(self, section: str) -> List[Dict]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"https://api.nytimes.com/svc/topstories/v2/{section}.json",
                params={"api-key": self.settings.nyt_api_key}
            )
            articles = resp.json().get("results", [])
            return [
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "content": a.get("abstract", ""),
                    "published_at": self._parse_iso_date(a.get("published_date"))
                }
                for a in articles if a.get("url")
            ]
    
    def _fetch_finnhub(self) -> List[Dict]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                "https://finnhub.io/api/v1/news",
                params={"token": self.settings.finnhub_api_key, "category": "general"}
            )
            articles = resp.json()
            return [
                {
                    "title": a.get("headline", ""),
                    "url": a.get("url", ""),
                    "content": a.get("summary", ""),
                    "published_at": datetime.fromtimestamp(a.get("datetime", 0)) if a.get("datetime") else datetime.utcnow()
                }
                for a in articles if isinstance(a, dict) and a.get("url")
            ]
    
    def _parse_date(self, date_str: str) -> datetime:
        if not date_str:
            return datetime.utcnow()
        try:
            return parsedate_to_datetime(date_str)
        except Exception:
            return datetime.utcnow()
    
    def _parse_iso_date(self, date_str: str | None) -> datetime:
        if not date_str:
            return datetime.utcnow()
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.utcnow()
    
    def is_duplicate(self, url: str) -> bool:
        db = SessionLocal()
        try:
            return db.query(NewsItem).filter(NewsItem.url == url).first() is not None
        finally:
            db.close()
    
    def store_article(self, article: Dict, category: str) -> NewsItem | None:
        db = SessionLocal()
        try:
            news_item = NewsItem(
                title=article["title"],
                url=article["url"],
                content=article["content"],
                category=category,
                published_at=article.get("published_at")
            )
            db.add(news_item)
            db.commit()
            db.refresh(news_item)
            return news_item
        except IntegrityError:
            db.rollback()
            return None
        finally:
            db.close()
    
    def process_article(self, article: Dict, category: str):
        if self.is_duplicate(article["url"]):
            return
        
        news_item = self.store_article(article, category)
        if not news_item:
            return
        
        chunks = self.chunker.split_text(article["content"])
        if not chunks:
            return
        
        vectors = self.embedder.embed(chunks)
        payloads = [
            {
                "source_id": news_item.id,
                "category": category,
                "timestamp": news_item.published_at.isoformat() if news_item.published_at is not None else datetime.utcnow().isoformat(),
                "content_chunk": chunk,
                "title": article["title"],
                "url": article["url"]
            }
            for chunk in chunks
        ]
        self.vector_store.upsert_vectors(vectors, payloads)
    
    def run(self):
        for source in SOURCES:
            try:
                if source["type"] == "rss":
                    articles = self.fetch_rss(source["url"])
                elif source["type"] == "api":
                    articles = self.fetch_api(source["name"], **{k: v for k, v in source.items() if k not in ["type", "name", "category"]})
                else:
                    continue
                
                for article in articles:
                    try:
                        self.process_article(article, source["category"])
                    except Exception:
                        continue
            except Exception:
                continue


def run_ingestion():
    IngestionPipeline().run()

__all__ = ["IngestionPipeline", "run_ingestion"]
