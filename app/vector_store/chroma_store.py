"""
Vector database module using ChromaDB for storing and retrieving code embeddings.
"""
import os
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np

from app.config.settings import config

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """ChromaDB vector store for code embeddings."""
    
    def __init__(
        self, 
        persistence_directory: str = None, 
        collection_name: str = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            persistence_directory: Directory to persist ChromaDB data.
              If None, uses the directory from config.
            collection_name: Name of the ChromaDB collection to use.
              If None, uses the name from config.
            embedding_model: Name of the SentenceTransformer model to use for embeddings.
        """
        self.persistence_directory = persistence_directory or config.vector_store.persistence_directory
        self.collection_name = collection_name or config.vector_store.collection_name
        self.embedding_model_name = embedding_model
        
        # Ensure persistence directory exists
        os.makedirs(self.persistence_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.persistence_directory)
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Create embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB Vector Store with collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with the following structure:
                {
                    'id': str,              # Unique identifier
                    'content': str,         # Text content to embed
                    'metadata': Dict        # Additional metadata
                }
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return
        
        # Extract document components
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Add documents to ChromaDB collection
        try:
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        n_results: int = 5, 
        where: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store for relevant documents.
        
        Args:
            query: The search query.
            n_results: Number of results to return.
            where: Filter condition for metadata fields.
            
        Returns:
            List of result dictionaries with structure:
                {
                    'id': str,              # Document ID
                    'content': str,         # Document content
                    'metadata': Dict,       # Document metadata
                    'similarity': float     # Similarity score
                }
        """
        try:
            search_results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            # Format the results
            results = []
            for i in range(len(search_results['ids'][0])):
                results.append({
                    'id': search_results['ids'][0][i],
                    'content': search_results['documents'][0][i],
                    'metadata': search_results['metadatas'][0][i],
                    'similarity': search_results['distances'][0][i] if 'distances' in search_results else None
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The document ID.
            
        Returns:
            Document dictionary or None if not found.
        """
        try:
            result = self.collection.get(ids=[document_id])
            
            if result and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return None
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            document_id: The document ID to delete.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Remove all documents from the collection.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.collection.delete()
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False 