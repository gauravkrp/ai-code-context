"""
ChromaDB vector store implementation with enhanced code chunking.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time

from app.config.settings import config
from app.utils.logger import setup_logger
from app.utils.code_chunker import CodeChunk, CodeChunker

logger = setup_logger(__name__, "logs/vector_store.log")

class ChromaStore:
    """ChromaDB vector store implementation."""
    
    def __init__(self, repo_id=None):
        """Initialize the vector store."""
        # Store config
        self.config = config.vector_store
        
        # Create persistence directory if it doesn't exist
        persistence_dir = Path(self.config.persistence_dir)
        persistence_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Using ChromaDB persistence directory: {persistence_dir.absolute()}")
        
        # Use repository-specific collection if repo_id is provided
        if repo_id:
            # Sanitize repo_id for use in collection name
            # Replace all non-alphanumeric characters with underscores
            sanitized_repo_id = ''.join(c if c.isalnum() else '_' for c in repo_id)
            # Ensure the collection name starts and ends with alphanumeric characters
            collection_name = f"{self.config.collection_name}_{sanitized_repo_id}"
            # Truncate if too long (ChromaDB has a 63 character limit)
            if len(collection_name) > 63:
                collection_name = collection_name[:63]
                # Ensure it ends with an alphanumeric character
                while not collection_name[-1].isalnum():
                    collection_name = collection_name[:-1]
        else:
            collection_name = self.config.collection_name
            
        self.collection_name = collection_name
        logger.info(f"Using collection name: {collection_name}")
        
        try:
            # Use Client with SQLite settings for reliable persistence
            logger.info(f"Initializing ChromaDB Client with SQLite and path: {persistence_dir.absolute()}")
            
            settings = Settings(
                persist_directory=str(persistence_dir.absolute()),
                anonymized_telemetry=False,
                is_persistent=True,
                allow_reset=True
            )
            
            self.client = chromadb.Client(settings)
            
            # List existing collections
            collections = self.client.list_collections()
            logger.info(f"Found existing collections: {[c.name for c in collections]}")
            
            # Get or create collection with explicit distance function
            logger.info(f"Getting or creating collection: {collection_name}")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Log collection count
            count = self.collection.count()
            logger.info(f"Collection '{collection_name}' contains {count} documents")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            logger.exception("Exception details:")
            raise
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize chunker
        self.chunker = CodeChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = None,
        incremental: bool = False
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add (either Document objects or dictionaries)
            batch_size: Optional batch size for processing
            incremental: If True, only add documents that don't exist in the store
        """
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
            
        try:
            # Use configured batch size if not specified
            batch_size = batch_size or self.config.batch_size
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"Starting indexing of {len(documents)} documents in {total_batches} batches")
            start_time = time.time()
            
            # If incremental, get existing document IDs to avoid re-adding
            existing_ids = set()
            if incremental:
                try:
                    # Get count of documents in collection
                    count = self.collection.count()
                    if count > 0:
                        # Get existing IDs (this might need to be batched for very large collections)
                        results = self.collection.get(limit=count, include=["metadatas"])
                        if results and "ids" in results:
                            existing_ids = set(results["ids"])
                        logger.info(f"Found {len(existing_ids)} existing documents for incremental update")
                except Exception as e:
                    logger.warning(f"Error checking existing documents for incremental update: {e}")
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch_num = (i // batch_size) + 1
                batch = documents[i:i + batch_size]
                
                # Log batch start with progress percentage
                progress = (batch_num / total_batches) * 100
                logger.info(f"Processing batch {batch_num}/{total_batches} ({progress:.1f}%) - {len(batch)} documents")
                
                # Process each document
                processed_batch = []
                batch_ids = []
                for j, doc in enumerate(batch):
                    # Handle both Document objects and dictionaries
                    if hasattr(doc, 'id') and hasattr(doc, 'text') and hasattr(doc, 'metadata'):
                        # It's a Document object
                        doc_id = str(doc.id)
                        content = doc.text
                        metadata = doc.metadata
                    else:
                        # It's a dictionary
                        processed_doc = doc.copy()
                        
                        # Ensure required metadata exists
                        metadata = processed_doc.get('metadata', {})
                        if not metadata:
                            # Create basic metadata from document fields
                            metadata = {
                                'file_path': processed_doc.get('file_path', ''),
                                'language': processed_doc.get('language', 'unknown'),
                                'size': processed_doc.get('size', 0),
                                'timestamp': processed_doc.get('timestamp', '')
                            }
                        
                        # Create a deterministic but unique ID based on content and file path
                        # This helps with deduplication and makes IDs consistent across runs
                        file_path = metadata.get('file_path', '')
                        # Replace special characters with '_' to make safe IDs
                        safe_path = ''.join(c if c.isalnum() else '_' for c in file_path)
                        # Use a positive hash value
                        content_hash = abs(hash(processed_doc['content'][:100])) if processed_doc.get('content') else 0
                        doc_id = f"doc_{safe_path}_{content_hash}_{i+j}"
                        content = processed_doc['content']
                    
                    # Convert any list or dict values in metadata to strings
                    metadata = {
                        k: str(v) if isinstance(v, (list, dict)) else v
                        for k, v in metadata.items()
                    }
                    
                    # Skip if this document ID already exists and we're doing incremental indexing
                    if incremental and doc_id in existing_ids:
                        logger.debug(f"Skipping existing document: {metadata.get('file_path', doc_id)}")
                        continue
                    
                    processed_batch.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata
                    })
                    batch_ids.append(doc_id)
            
                # Skip empty batches
                if not processed_batch:
                    logger.info(f"Batch {batch_num} has no new documents to add, skipping")
                    continue
                
                # Add batch to collection
                self.collection.add(
                    documents=[doc['content'] for doc in processed_batch],
                    metadatas=[doc['metadata'] for doc in processed_batch],
                    ids=batch_ids
                )
                
                # Verify documents were added
                current_count = self.collection.count()
                logger.info(f"Verified document count after batch: {current_count}")
                
                # Log batch completion
                logger.info(f"Completed batch {batch_num}/{total_batches} ({progress:.1f}%)")
            
            # Log overall completion with timing
            elapsed_time = time.time() - start_time
            docs_per_second = len(documents) / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"Successfully indexed {len(documents)} documents in {elapsed_time:.2f} seconds ({docs_per_second:.2f} docs/sec)")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            logger.error("Indexing failed - please check the error and try again")
            raise
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with metadata
        """
        try:
            # Log search details
            logger.info(f"Searching for query: '{query}' with n_results={n_results}")
            logger.info(f"Filter criteria: {filter_criteria}")
            
            # Log collection stats
            collection_count = self.collection.count()
            logger.info(f"Collection '{self.collection_name}' contains {collection_count} documents")
            
            # Create query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare where clause for filtering
            where = filter_criteria if filter_criteria else None
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection_count) if collection_count > 0 else n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            if 'documents' in results and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    doc_data = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    formatted_results.append(doc_data)
                
                # Log sample results
                logger.info(f"Top result similarity: {1 - formatted_results[0]['distance'] if formatted_results[0]['distance'] is not None else 'N/A'}")
                for i, result in enumerate(formatted_results[:2]):  # Log top 2 results
                    metadata_str = ', '.join([f"{k}: {v}" for k, v in result['metadata'].items() if k in ['file_path', 'language']])
                    content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                    logger.info(f"Result {i+1}: {metadata_str} - Content: {content_preview}")
            else:
                logger.warning(f"No documents found in search results for query: '{query}'")
            
            logger.info(f"Found {len(formatted_results)} results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            logger.exception("Search exception details:")
            return []
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.collection.delete(where={})
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}") 