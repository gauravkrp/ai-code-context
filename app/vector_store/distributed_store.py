"""
Distributed vector store implementation with sharding and load balancing.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import asyncio
from app.vector_store.chroma_store import ChromaStore
from app.utils.code_embeddings import CodeEmbedder, EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class ShardConfig:
    """Configuration for vector store sharding."""
    num_shards: int = 4
    shard_size: int = 10000
    replication_factor: int = 2
    load_balance_threshold: float = 0.8

class DistributedStore:
    """Distributed vector store with sharding and load balancing."""
    
    def __init__(
        self,
        base_dir: str,
        shard_config: Optional[ShardConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """Initialize the distributed store."""
        self.base_dir = Path(base_dir)
        self.shard_config = shard_config or ShardConfig()
        self.embedding_config = embedding_config or EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize components
        self.embedder = CodeEmbedder(self.embedding_config)
        self.shards: List[ChromaStore] = []
        self.shard_metadata: List[Dict[str, Any]] = []
        
        # Create shards
        self._initialize_shards()
        
        # Start load balancing
        asyncio.create_task(self._load_balance_shards())
    
    def _initialize_shards(self):
        """Initialize vector store shards."""
        try:
            # Create shard directories
            for i in range(self.shard_config.num_shards):
                shard_dir = self.base_dir / f"shard_{i}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize ChromaDB store for shard
                store = ChromaStore(
                    persistence_dir=str(shard_dir),
                    collection_name=f"shard_{i}"
                )
                self.shards.append(store)
                
                # Initialize shard metadata
                self.shard_metadata.append({
                    "id": i,
                    "size": 0,
                    "load": 0.0,
                    "documents": set()
                })
            
            logger.info(f"Initialized {self.shard_config.num_shards} shards")
            
        except Exception as e:
            logger.error(f"Error initializing shards: {str(e)}")
            raise
    
    def _get_shard_id(self, document_id: str) -> int:
        """Get shard ID for a document using consistent hashing."""
        hash_value = int(hashlib.md5(document_id.encode()).hexdigest(), 16)
        return hash_value % self.shard_config.num_shards
    
    def _get_replica_shards(self, primary_shard_id: int) -> List[int]:
        """Get replica shard IDs for a primary shard."""
        replica_shards = []
        for i in range(self.shard_config.replication_factor):
            replica_id = (primary_shard_id + i + 1) % self.shard_config.num_shards
            replica_shards.append(replica_id)
        return replica_shards
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the distributed store."""
        try:
            # Generate embeddings
            embeddings = self.embedder.embed_code(
                [doc["content"] for doc in documents],
                metadata
            )
            
            # Group documents by shard
            shard_documents: Dict[int, List[Dict[str, Any]]] = {}
            shard_embeddings: Dict[int, np.ndarray] = {}
            shard_metadata: Dict[int, List[Dict[str, Any]]] = {}
            
            for i, doc in enumerate(documents):
                shard_id = self._get_shard_id(doc["id"])
                
                if shard_id not in shard_documents:
                    shard_documents[shard_id] = []
                    shard_embeddings[shard_id] = []
                    shard_metadata[shard_id] = []
                
                shard_documents[shard_id].append(doc)
                shard_embeddings[shard_id].append(embeddings[i])
                if metadata:
                    shard_metadata[shard_id].append(metadata[i])
            
            # Add documents to shards in parallel
            tasks = []
            for shard_id, docs in shard_documents.items():
                # Add to primary shard
                tasks.append(
                    self._add_to_shard(
                        shard_id,
                        docs,
                        np.array(shard_embeddings[shard_id]),
                        shard_metadata.get(shard_id)
                    )
                )
                
                # Add to replica shards
                for replica_id in self._get_replica_shards(shard_id):
                    tasks.append(
                        self._add_to_shard(
                            replica_id,
                            docs,
                            np.array(shard_embeddings[shard_id]),
                            shard_metadata.get(shard_id)
                        )
                    )
            
            await asyncio.gather(*tasks)
            
            # Update shard metadata
            for shard_id in shard_documents:
                self.shard_metadata[shard_id]["size"] += len(shard_documents[shard_id])
                self.shard_metadata[shard_id]["load"] = (
                    self.shard_metadata[shard_id]["size"] / self.shard_config.shard_size
                )
            
            logger.info(f"Added {len(documents)} documents to distributed store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    async def _add_to_shard(
        self,
        shard_id: int,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to a specific shard."""
        try:
            await asyncio.to_thread(
                self.shards[shard_id].add_documents,
                documents,
                embeddings,
                metadata
            )
            
            # Update shard metadata
            for doc in documents:
                self.shard_metadata[shard_id]["documents"].add(doc["id"])
            
        except Exception as e:
            logger.error(f"Error adding documents to shard {shard_id}: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search across all shards."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_code([query])[0]
            
            # Search all shards in parallel
            tasks = []
            for shard in self.shards:
                tasks.append(
                    asyncio.to_thread(
                        shard.search,
                        query_embedding,
                        n_results,
                        filter_criteria
                    )
                )
            
            results = await asyncio.gather(*tasks)
            
            # Combine and sort results
            all_results = []
            for shard_results in results:
                all_results.extend(shard_results)
            
            # Sort by similarity and take top n_results
            all_results.sort(key=lambda x: x["distance"])
            return all_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error searching distributed store: {str(e)}")
            raise
    
    async def _load_balance_shards(self):
        """Load balance shards if needed."""
        while True:
            try:
                # Check shard loads
                overloaded_shards = [
                    shard_id
                    for shard_id, metadata in enumerate(self.shard_metadata)
                    if metadata["load"] > self.shard_config.load_balance_threshold
                ]
                
                if overloaded_shards:
                    logger.info(f"Found {len(overloaded_shards)} overloaded shards")
                    await self._redistribute_shards(overloaded_shards)
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in load balancing: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _redistribute_shards(self, overloaded_shards: List[int]):
        """Redistribute documents from overloaded shards."""
        try:
            for shard_id in overloaded_shards:
                # Find target shard with lowest load
                target_shard_id = min(
                    range(len(self.shard_metadata)),
                    key=lambda i: self.shard_metadata[i]["load"]
                )
                
                # Get documents to redistribute
                documents = await asyncio.to_thread(
                    self.shards[shard_id].get_documents
                )
                
                if not documents:
                    continue
                
                # Redistribute documents
                await self.add_documents(documents)
                
                # Update metadata
                self.shard_metadata[shard_id]["size"] = 0
                self.shard_metadata[shard_id]["load"] = 0.0
                self.shard_metadata[shard_id]["documents"].clear()
                
                logger.info(
                    f"Redistributed {len(documents)} documents from shard {shard_id} "
                    f"to shard {target_shard_id}"
                )
            
        except Exception as e:
            logger.error(f"Error redistributing shards: {str(e)}")
            raise
    
    def get_shard_stats(self) -> Dict[str, Any]:
        """Get statistics about shard distribution."""
        return {
            "num_shards": self.shard_config.num_shards,
            "shard_sizes": [metadata["size"] for metadata in self.shard_metadata],
            "shard_loads": [metadata["load"] for metadata in self.shard_metadata],
            "total_documents": sum(metadata["size"] for metadata in self.shard_metadata)
        } 