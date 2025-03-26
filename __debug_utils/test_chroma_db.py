#!/usr/bin/env python
"""
Test script to verify ChromaDB is working properly with persistence.
This script performs basic operations with ChromaDB:
1. Creates/opens a ChromaDB instance with persistence
2. Creates a collection and adds test documents
3. Checks if documents were added successfully
4. Closes and reopens ChromaDB to verify persistence
5. Performs a simple query to verify functionality
"""
import os
import sys
import logging
import time
from pathlib import Path

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ChromaDB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chroma_db_test")

def get_test_docs():
    """Generate some test documents"""
    return [
        {"id": "doc1", "content": "This is a test document about Python programming.", 
         "metadata": {"lang": "en", "type": "test"}},
        {"id": "doc2", "content": "ChromaDB is a vector database for embeddings and semantic search.", 
         "metadata": {"lang": "en", "type": "test"}},
        {"id": "doc3", "content": "React hooks are used for state management in functional components.", 
         "metadata": {"lang": "en", "type": "test"}},
    ]

def create_client(db_path):
    """Create a ChromaDB client with persistence"""
    logger.info(f"Creating ChromaDB client with path: {db_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize client with settings for persistence
    settings = Settings(
        persist_directory=str(db_path),
        anonymized_telemetry=False,
        is_persistent=True
    )
    
    return chromadb.Client(settings)

def test_chroma_db():
    """Main test function for ChromaDB"""
    # Setup test directory
    test_dir = Path("./test_chroma_db")
    collection_name = "test_collection"
    
    try:
        # Clean up any existing test data
        if test_dir.exists():
            logger.info(f"Cleaning up existing test directory: {test_dir}")
            import shutil
            shutil.rmtree(test_dir)
        
        # Step 1: Create client and collection
        logger.info("=== Step 1: Creating ChromaDB client and collection ===")
        client = create_client(test_dir)
        
        # List collections (should be empty)
        collections = client.list_collections()
        logger.info(f"Initial collections: {[c.name for c in collections]}")
        
        # Create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created collection: {collection_name}")
        
        # Step 2: Add documents
        logger.info("=== Step 2: Adding test documents ===")
        docs = get_test_docs()
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Add documents with embeddings
        collection.add(
            ids=[doc["id"] for doc in docs],
            documents=[doc["content"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs],
            embeddings=[embedding_model.encode(doc["content"]).tolist() for doc in docs]
        )
        
        # Verify documents were added
        count = collection.count()
        logger.info(f"Document count after adding: {count}")
        assert count == len(docs), f"Expected {len(docs)} documents, but got {count}"
        
        # Step 3: Query to verify it works
        logger.info("=== Step 3: Testing query functionality ===")
        results = collection.query(
            query_texts=["React state management"],
            n_results=2
        )
        
        logger.info(f"Query results: {results}")
        assert len(results['documents']) > 0, "Expected at least one result, but got none"
        
        # Step 4: Close and reopen to verify persistence
        logger.info("=== Step 4: Testing persistence across sessions ===")
        logger.info("Closing ChromaDB client")
        del client
        del collection
        
        # Wait briefly to ensure changes are flushed
        time.sleep(1)
        
        # Reopen client
        logger.info("Reopening ChromaDB client")
        new_client = create_client(test_dir)
        
        # Check if collection exists
        collections = new_client.list_collections()
        logger.info(f"Collections after reopening: {[c.name for c in collections]}")
        
        # Get collection
        new_collection = new_client.get_collection(name=collection_name)
        
        # Verify document count
        new_count = new_collection.count()
        logger.info(f"Document count after reopening: {new_count}")
        assert new_count == len(docs), f"Expected {len(docs)} documents after reopening, but got {new_count}"
        
        # Step 5: Query again
        logger.info("=== Step 5: Testing query functionality after reopening ===")
        new_results = new_collection.query(
            query_texts=["Python programming"],
            n_results=2
        )
        
        logger.info(f"Query results after reopening: {new_results}")
        assert len(new_results['documents']) > 0, "Expected at least one result after reopening, but got none"
        
        logger.info("=== All tests passed successfully! ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test data
        if test_dir.exists() and "keep" not in sys.argv:
            logger.info(f"Cleaning up test directory: {test_dir}")
            import shutil
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_chroma_db()
    sys.exit(0 if success else 1) 