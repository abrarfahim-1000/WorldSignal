"""Initialize the database and vector store for WorldSignal."""

import sys
from app.db.models import init_db
from app.rag.vector_store import get_vector_store


if __name__ == "__main__":
    try:
        init_db()
        created = get_vector_store().init_collection()
        print("Storage initialized" + (" (collection created)" if created else " (collection exists)"))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
