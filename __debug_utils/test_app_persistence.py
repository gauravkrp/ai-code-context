#!/usr/bin/env python
"""
Test ChromaDB persistence using the app's actual ChromaStore implementation.
This verifies that the app's ChromaStore correctly persists data between sessions.
"""
import os
import sys
import shutil
import logging
from pathlib import Path

# Add parent directory to system path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app modules
from app.vector_store.chroma_store import ChromaStore
from app.config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_app_persistence")

# Test settings
TEST_DIR = Path("./test_app_persistence")
TEST_DOCS = [
    {
        "content": "This is a test document for ChromaDB persistence verification.",
        "metadata": {
            "file_path": "test_file_1.py",
            "language": "python",
            "size": 100,
            "timestamp": "2025-03-26"
        }
    },
    {
        "content": "Another test document with different content for vector search testing.",
        "metadata": {
            "file_path": "test_file_2.js",
            "language": "javascript",
            "size": 200,
            "timestamp": "2025-03-26"
        }
    },
    {
        "content": "React hooks are functions that let you 'hook into' React state and lifecycle features.",
        "metadata": {
            "file_path": "test_file_3.tsx",
            "language": "typescript",
            "size": 300,
            "timestamp": "2025-03-26"
        }
    }
]

def setup_test_environment():
    """Set up test environment with a temporary ChromaDB directory"""
    # Create test directory
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(exist_ok=True)
    
    # Back up original persistence directory setting
    original_dir = config.vector_store.persistence_dir
    
    # Override with test directory
    config.vector_store.persistence_dir = str(TEST_DIR)
    logger.info(f"Set test persistence directory: {config.vector_store.persistence_dir}")
    
    return original_dir

def restore_environment(original_dir):
    """Restore original environment settings"""
    # Restore original persistence directory
    config.vector_store.persistence_dir = original_dir
    logger.info(f"Restored original persistence directory: {config.vector_store.persistence_dir}")
    
    # Clean up test directory
    if TEST_DIR.exists() and "keep" not in sys.argv:
        shutil.rmtree(TEST_DIR)
        logger.info(f"Cleaned up test directory: {TEST_DIR}")

def create_and_populate_store():
    """Create a ChromaStore instance and populate it with test documents"""
    logger.info("Creating ChromaStore instance...")
    store = ChromaStore()
    
    logger.info(f"Adding {len(TEST_DOCS)} test documents...")
    store.add_documents(TEST_DOCS)
    
    # Verify documents were added
    results = store.search("test document", n_results=10)
    logger.info(f"Search returned {len(results)} results after adding documents")
    
    return len(results)

def verify_persistence():
    """Verify that documents persist by creating a new ChromaStore instance"""
    logger.info("Creating new ChromaStore instance to verify persistence...")
    store = ChromaStore()
    
    # Search for documents
    results = store.search("test document", n_results=10)
    logger.info(f"Search returned {len(results)} results after reopening ChromaStore")
    
    return len(results)

def run_test():
    """Run the persistence test with the app's ChromaStore"""
    original_dir = None
    try:
        # Setup test environment
        logger.info("=== Setting up test environment ===")
        original_dir = setup_test_environment()
        
        # Create and populate store
        logger.info("\n=== Phase 1: Creating and populating ChromaStore ===")
        doc_count = create_and_populate_store()
        if doc_count == 0:
            logger.error("❌ FAILURE: No documents found after adding them")
            return False
            
        # Verify persistence
        logger.info("\n=== Phase 2: Verifying persistence ===")
        result_count = verify_persistence()
        
        # Check results
        if result_count > 0:
            logger.info(f"✅ SUCCESS: Found {result_count} documents after reopening ChromaStore")
            return True
        else:
            logger.error("❌ FAILURE: No documents found after reopening ChromaStore")
            return False
            
    except Exception as e:
        logger.error(f"❌ ERROR: Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore environment
        if original_dir:
            restore_environment(original_dir)

if __name__ == "__main__":
    success = run_test()
    print("\n=== TEST RESULT ===")
    if success:
        print("✅ SUCCESS: ChromaStore persistence is working correctly!")
    else:
        print("❌ FAILURE: ChromaStore persistence test failed!")
    
    sys.exit(0 if success else 1) 