"""RSS feed ingestion pipeline."""
from datetime import datetime
from typing import List, Dict
from email.utils import parsedate_to_datetime
import feedparser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.db.models import NewsItem, SessionLocal
from app.embeddings.base import get_embedder
from app.rag.vector_store import get_vector_store


# RSS Feed sources
RSS_FEEDS = [
    {"url": "https://finance.yahoo.com/news/rssindex", "category": "finance"},
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "category": "finance"},
    {"url": "https://feeds.bloomberg.com/markets/news.rss", "category": "finance"},
    {"url": "https://www.reuters.com/rssFeed/worldNews", "category": "geopolitics"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "category": "geopolitics"},
    {"url": "https://www.theguardian.com/world/rss", "category": "geopolitics"},
]


class IngestionPipeline:
    def __init__(self):
        settings = get_settings()
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    def fetch_feed(self, feed_url: str) -> List[Dict]:
        feed = feedparser.parse(feed_url)
        return [
            {
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "content": entry.get("summary", entry.get("description", "")),
                "published_at": self._parse_date(str(entry.get("published", "")))
            }
            for entry in feed.entries
        ]
    
    def _parse_date(self, date_str: str) -> datetime:
        if not date_str:
            return datetime.utcnow()
        try:
            return parsedate_to_datetime(date_str)
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
        for feed_config in RSS_FEEDS:
            try:
                articles = self.fetch_feed(feed_config["url"])
                for article in articles:
                    try:
                        self.process_article(article, feed_config["category"])
                    except Exception:
                        continue
            except Exception:
                continue


def run_ingestion():
    IngestionPipeline().run()

__all__ = ["IngestionPipeline", "run_ingestion"]
