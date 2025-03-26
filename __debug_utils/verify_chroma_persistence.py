#!/usr/bin/env python
"""
Simple script to verify ChromaDB persistence is working correctly.
This script focuses specifically on testing that data persists between sessions.
"""
import os
import sys
import shutil
import logging
import chromadb
from chromadb.config import Settings
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_chroma_persistence")

# Test settings
TEST_DIR = "./test_persistence"
COLLECTION_NAME = "persistence_test"
TEST_DOCS = [
    {"id": "test1", "content": "This is test document 1", "metadata": {"source": "test"}},
    {"id": "test2", "content": "This is test document 2", "metadata": {"source": "test"}},
    {"id": "test3", "content": "This is test document 3", "metadata": {"source": "test"}}
]

def create_db():
    """Create a test database and add documents"""
    # Create/clear test directory
    test_path = Path(TEST_DIR)
    if test_path.exists():
        shutil.rmtree(test_path)
    test_path.mkdir(exist_ok=True)
    
    logger.info(f"ChromaDB version: {chromadb.__version__}")
    logger.info(f"Creating test database at: {test_path.absolute()}")
    
    # Create client with persistence
    settings = Settings(
        persist_directory=str(test_path.absolute()),
        anonymized_telemetry=False,
        is_persistent=True
    )
    
    client = chromadb.Client(settings)
    logger.info(f"Created ChromaDB client: {type(client).__name__}")
    
    # Create collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"Created collection: {COLLECTION_NAME}")
    
    # Add documents
    collection.add(
        ids=[doc["id"] for doc in TEST_DOCS],
        documents=[doc["content"] for doc in TEST_DOCS],
        metadatas=[doc["metadata"] for doc in TEST_DOCS]
    )
    
    # Verify document count
    count = collection.count()
    logger.info(f"Added {count} documents to collection")
    
    return count

def verify_db():
    """Verify that documents persist in the database"""
    test_path = Path(TEST_DIR)
    if not test_path.exists():
        logger.error(f"Test directory not found: {test_path.absolute()}")
        return False
    
    logger.info(f"Opening existing database at: {test_path.absolute()}")
    
    # Create client with same settings
    settings = Settings(
        persist_directory=str(test_path.absolute()),
        anonymized_telemetry=False,
        is_persistent=True
    )
    
    # Open client
    client = chromadb.Client(settings)
    logger.info(f"Opened ChromaDB client: {type(client).__name__}")
    
    # List collections
    collections = client.list_collections()
    logger.info(f"Found collections: {[c.name for c in collections]}")
    
    # Get collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Retrieved collection: {COLLECTION_NAME}")
        
        # Check document count
        count = collection.count()
        logger.info(f"Collection has {count} documents")
        
        if count == len(TEST_DOCS):
            logger.info("✅ SUCCESS: Document count matches expected value!")
            return True
        else:
            logger.error(f"❌ FAILURE: Expected {len(TEST_DOCS)} documents, but found {count}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FAILURE: Could not retrieve collection: {e}")
        return False

def run_persistence_test():
    """Run full persistence test"""
    try:
        # Create database and add documents
        logger.info("=== PHASE 1: Creating database and adding documents ===")
        doc_count = create_db()
        if doc_count != len(TEST_DOCS):
            logger.error(f"Failed to add documents. Expected {len(TEST_DOCS)}, got {doc_count}")
            return False
            
        logger.info("\n=== PHASE 2: Verifying database persistence ===")
        # Verify persistence
        return verify_db()
    finally:
        # Cleanup
        if "keep" not in sys.argv:
            test_path = Path(TEST_DIR)
            if test_path.exists():
                shutil.rmtree(test_path)
                logger.info(f"Cleaned up test directory: {test_path}")

if __name__ == "__main__":
    success = run_persistence_test()
    print("\n=== TEST RESULT ===")
    if success:
        print("✅ SUCCESS: ChromaDB persistence is working correctly!")
    else:
        print("❌ FAILURE: ChromaDB persistence test failed!")
    
    sys.exit(0 if success else 1) 