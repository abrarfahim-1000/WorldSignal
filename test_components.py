"""Quick component test for Phase 2."""
from app.db.models import init_db, SessionLocal, NewsItem
from app.rag.vector_store import get_vector_store
from app.embeddings.base import get_embedder
from app.config import get_settings


def test_components():
    print("=== Quick Component Test ===\n")
    
    # 1. Config
    print("1. Testing config...")
    try:
        settings = get_settings()
        print(f"   ✓ Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
        print(f"   ✓ Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
        print(f"   ✓ Embedding model: {settings.embedding_model}")
    except Exception as e:
        print(f"   ✗ Config failed: {e}")
        return
    
    # 2. Database
    print("\n2. Testing database connection...")
    try:
        init_db()
        db = SessionLocal()
        count = db.query(NewsItem).count()
        db.close()
        print(f"   ✓ Database connected (articles: {count})")
    except Exception as e:
        print(f"   ✗ Database failed: {e}")
        return
    
    # 3. Vector Store
    print("\n3. Testing Qdrant connection...")
    try:
        vs = get_vector_store()
        vs.init_collection()
        print(f"   ✓ Qdrant connected (collection: {vs.collection_name})")
    except Exception as e:
        print(f"   ✗ Qdrant failed: {e}")
        return
    
    # 4. Embedder
    print("\n4. Testing embedder...")
    try:
        embedder = get_embedder()
        test_text = "This is a test sentence for embedding."
        vector = embedder.embed([test_text])[0]
        print(f"   ✓ Embedder working (vector size: {len(vector)})")
    except Exception as e:
        print(f"   ✗ Embedder failed: {e}")
        return
    
    # 5. RSS Feed Test
    print("\n5. Testing RSS feed fetch...")
    try:
        import feedparser
        feed = feedparser.parse("https://finance.yahoo.com/news/rssindex")
        print(f"   ✓ RSS fetch working (entries: {len(feed.entries)})")
        if feed.entries:
            title = feed.entries[0].get('title', 'N/A')
            print(f"   Sample: {(title or 'N/A')[:60]}...")
    except Exception as e:
        print(f"   ✗ RSS fetch failed: {e}")
        return
    
    print("\n=== All Components Working ===")


if __name__ == "__main__":
    test_components()
