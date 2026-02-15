"""Test script for Phase 2: Ingestion Pipeline."""
import sys
from app.db.models import init_db, SessionLocal, NewsItem
from app.rag.vector_store import get_vector_store
from app.fetcher.pipeline import run_ingestion


def test_phase2():
    print("=== Phase 2 Test: Ingestion Pipeline ===\n")
    
    # Step 1: Initialize storage
    print("1. Initializing storage...")
    try:
        init_db()
        created = get_vector_store().init_collection()
        print(f"   ✓ Database initialized")
        print(f"   ✓ Qdrant collection {'created' if created else 'exists'}\n")
    except Exception as e:
        print(f"   ✗ Storage initialization failed: {e}")
        return False
    
    # Step 2: Check initial state
    print("2. Checking initial state...")
    db = SessionLocal()
    try:
        initial_count = db.query(NewsItem).count()
        print(f"   • Existing articles in DB: {initial_count}\n")
    finally:
        db.close()
    
    # Step 3: Run ingestion
    print("3. Running ingestion pipeline...")
    try:
        run_ingestion()
        print("   ✓ Ingestion completed\n")
    except Exception as e:
        print(f"   ✗ Ingestion failed: {e}")
        return False
    
    # Step 4: Verify data
    print("4. Verifying stored data...")
    db = SessionLocal()
    try:
        total_articles = db.query(NewsItem).count()
        finance_count = db.query(NewsItem).filter(NewsItem.category == "finance").count()
        geo_count = db.query(NewsItem).filter(NewsItem.category == "geopolitics").count()
        
        print(f"   • Total articles: {total_articles}")
        print(f"   • Finance articles: {finance_count}")
        print(f"   • Geopolitics articles: {geo_count}")
        print(f"   • New articles added: {total_articles - initial_count}\n")
        
        if total_articles > initial_count:
            print("   ✓ Articles successfully stored in PostgreSQL")
            
            # Show sample
            sample = db.query(NewsItem).first()
            if sample:
                print(f"\n   Sample article:")
                print(f"   - Title: {sample.title[:60]}...")
                print(f"   - Category: {sample.category}")
                print(f"   - URL: {sample.url[:60]}...")
        else:
            print("   ℹ No new articles (might be duplicates)")
            
    finally:
        db.close()
    
    # Step 5: Test vector search
    print("\n5. Testing vector search...")
    try:
        from app.embeddings.base import get_embedder
        embedder = get_embedder()
        vector_store = get_vector_store()
        
        query = "financial markets and economy"
        query_vector = embedder.embed([query])[0]
        results = vector_store.search(query_vector, limit=3)
        
        print(f"   • Query: '{query}'")
        print(f"   • Results found: {len(results)}")
        
        if results:
            print("   ✓ Vector search working\n")
            print("   Top result:")
            print(f"   - Score: {results[0]['score']:.4f}")
            print(f"   - Category: {results[0]['payload']['category']}")
            print(f"   - Chunk: {results[0]['payload']['content_chunk'][:80]}...")
        else:
            print("   ℹ No results (collection might be empty)")
            
    except Exception as e:
        print(f"   ✗ Vector search failed: {e}")
        return False
    
    print("\n=== Phase 2 Test Complete ===")
    return True


if __name__ == "__main__":
    success = test_phase2()
    sys.exit(0 if success else 1)
