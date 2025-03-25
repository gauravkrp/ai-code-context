"""
GitHub repository indexer module.

This module coordinates the indexing process for a GitHub repository.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import os
import time

from tqdm import tqdm

from app.github.repository import GitHubRepository
from app.utils.text_processing import split_code_to_chunks, create_document_metadata
from app.vector_store.chroma_store import ChromaVectorStore
from app.config.settings import config

logger = logging.getLogger(__name__)

class RepositoryIndexer:
    """Class for indexing GitHub repository code."""
    
    def __init__(
        self, 
        repository: GitHubRepository = None,
        vector_store: ChromaVectorStore = None,
        file_extensions: List[str] = None
    ):
        """
        Initialize the repository indexer.
        
        Args:
            repository: GitHubRepository instance. If None, a new one is created.
            vector_store: Vector store instance. If None, a new one is created.
            file_extensions: List of file extensions to index (e.g., ['.py', '.js']).
                            If None, all files are indexed.
        """
        self.repository = repository or GitHubRepository()
        self.vector_store = vector_store or ChromaVectorStore()
        self.file_extensions = file_extensions
        
        # Set of already indexed document IDs
        self.indexed_docs = set()
        
        # Statistics
        self.total_files = 0
        self.total_chunks = 0
        self.start_time = None
    
    def index_repository(self, clear_existing: bool = False) -> Tuple[int, int]:
        """
        Index the entire repository.
        
        Args:
            clear_existing: If True, clears the existing vector store data.
            
        Returns:
            Tuple of (number of files indexed, number of chunks created).
        """
        # Start timer
        self.start_time = time.time()
        
        # Reset statistics
        self.total_files = 0
        self.total_chunks = 0
        
        logger.info(f"Starting indexing of repository: {self.repository.repository_name}")
        
        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing vector store data")
            self.vector_store.clear_collection()
        
        try:
            # Clone the repository
            self.repository.clone_repository()
            
            # Get list of files
            file_list = self.repository.get_file_list(self.file_extensions)
            self.total_files = len(file_list)
            
            logger.info(f"Found {self.total_files} files to index")
            
            # Process files in batches for efficiency
            self._process_files()
            
            elapsed_time = time.time() - self.start_time
            logger.info(f"Indexing completed in {elapsed_time:.2f} seconds")
            logger.info(f"Indexed {self.total_files} files and created {self.total_chunks} chunks")
            
            # Clean up temporary files
            self.repository.cleanup()
            
            return self.total_files, self.total_chunks
            
        except Exception as e:
            logger.error(f"Error indexing repository: {e}")
            # Clean up temporary files even if an error occurs
            self.repository.cleanup()
            raise
    
    def _process_files(self, batch_size: int = 100):
        """
        Process files from the repository and add them to the vector store.
        
        Args:
            batch_size: Number of chunks to process in each batch.
        """
        all_chunks = []
        
        # Iterate through files using the repository iterator
        for file_path, content in tqdm(self.repository.iter_files(self.file_extensions), 
                                      desc="Processing files"):
            try:
                # Skip empty files
                if not content.strip():
                    continue
                
                # Skip binary files or files that are too large
                if len(content) > 1000000 or '\0' in content:  # Skip files > 1MB or containing null bytes
                    logger.warning(f"Skipping file {file_path}: too large or binary")
                    continue
                
                # Split the file into chunks
                chunks = split_code_to_chunks(file_path, content)
                
                if chunks:
                    all_chunks.extend(chunks)
                    self.total_chunks += len(chunks)
                
                # Process in batches to avoid memory issues
                if len(all_chunks) >= batch_size:
                    self._index_chunks(all_chunks)
                    all_chunks = []
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Index any remaining chunks
        if all_chunks:
            self._index_chunks(all_chunks)
    
    def _index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Index a batch of chunks in the vector store.
        
        Args:
            chunks: List of document chunks to index.
        """
        try:
            # Filter out any chunks that have already been indexed
            new_chunks = [chunk for chunk in chunks if chunk['id'] not in self.indexed_docs]
            
            if not new_chunks:
                return
            
            # Add chunks to vector store
            self.vector_store.add_documents(new_chunks)
            
            # Update the set of indexed documents
            for chunk in new_chunks:
                self.indexed_docs.add(chunk['id'])
                
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise 