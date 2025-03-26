"""
Advanced Retrieval Augmented Generation (RAG) for code repositories.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from app.vector_store.chroma_store import ChromaStore
from app.vector_store.distributed_store import DistributedStore
from app.utils.llm import LLMClient
from app.utils.code_embeddings import CodeEmbedder, EmbeddingConfig
from app.rag.query_optimizer import QueryOptimizer, QueryOptimizationConfig
from app.config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """A turn in a conversation."""
    query: str
    response: str
    context: List[Dict[str, Any]]
    timestamp: str

class AdvancedRAG:
    """Advanced Retrieval Augmented Generation for code repositories."""
    
    def __init__(
        self,
        vector_store: Optional[Any] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize the Advanced RAG system."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector store
        if vector_store:
            self.vector_store = vector_store
        elif config.vector_store.use_distributed:
            self.vector_store = DistributedStore(
                base_dir=config.vector_store.persistence_dir,
            )
        else:
            self.vector_store = ChromaStore()
        
        # Initialize LLM client
        self.llm_client = llm_client or LLMClient()
        
        # Initialize query optimizer
        self.query_optimizer = QueryOptimizer(
            QueryOptimizationConfig(
                use_query_expansion=config.rag.query_reformulation,
                use_bm25=True,
                use_code_specific_terms=True,
                use_hybrid_search=True,
                intent_detection=True
            )
        )
        
        # Initialize code embedder for query embedding
        self.code_embedder = CodeEmbedder(
            config.embedding
        )
        
        self.logger.info("Initialized Advanced RAG system")
    
    def query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a query with advanced RAG techniques.
        
        Args:
            query: The user's query
            conversation_history: Optional conversation history
            
        Returns:
            Dict with answer, relevant code snippets, and metadata
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Optimize query
            query_analysis = self.query_optimizer.optimize_query(
                query,
                None,  # No previous context initially
                conversation_history
            )
            
            optimized_query = query_analysis["optimized_query"]
            search_params = query_analysis["search_params"]
            
            # Increase search results count
            search_params["n_results"] = max(search_params.get("n_results", 5), 15)
            
            self.logger.debug(f"Optimized query: {optimized_query}")
            self.logger.info(f"Search parameters: n_results={search_params['n_results']}, filters={search_params.get('filter_criteria')}")
            
            # Retrieve relevant documents
            if "query_embedding" in query_analysis and config.rag.query_reformulation:
                # Embedding available, but can't pass directly - use optimized query
                results = self.vector_store.search(
                    query=optimized_query,
                    n_results=search_params["n_results"],
                    filter_criteria=search_params["filter_criteria"]
                )
            else:
                # Use text-based search
                results = self.vector_store.search(
                    query=optimized_query,
                    n_results=search_params["n_results"],
                    filter_criteria=search_params["filter_criteria"]
                )
            
            # Rerank results if needed
            if search_params["rerank"]:
                results = self.query_optimizer.rerank_results(results, query_analysis)
            
            # Extract code snippets and metadata
            code_snippets = []
            for result in results:
                snippet = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "relevance": 1.0 - result.get("distance", 0)
                }
                
                # Add language info if available
                if "language" in result["metadata"]:
                    snippet["language"] = result["metadata"]["language"]
                elif "file_path" in result["metadata"]:
                    ext = result["metadata"]["file_path"].split(".")[-1].lower()
                    # Map extension to language
                    lang_map = {
                        "py": "python",
                        "js": "javascript",
                        "ts": "typescript",
                        "jsx": "javascript",
                        "tsx": "typescript",
                        "java": "java",
                        "kt": "kotlin",
                        "kts": "kotlin",
                        "cpp": "cpp",
                        "c": "c",
                        "h": "cpp",
                        "hpp": "cpp",
                        "go": "go",
                        "rs": "rust"
                    }
                    snippet["language"] = lang_map.get(ext, "text")
                
                code_snippets.append(snippet)
            
            # Prepare context for LLM
            context = self._prepare_context(code_snippets, query_analysis["intent"])
            
            # Generate response with LLM
            response = self.llm_client.query(
                query=query,
                code_context=context,
                metadata={
                    "intent": query_analysis["intent"],
                    "code_terms": query_analysis["code_terms"]
                }
            )
            
            # Prepare final result
            result = {
                "answer": response,
                "code_snippets": code_snippets,
                "query_analysis": {
                    "intent": query_analysis["intent"],
                    "optimized_query": optimized_query
                }
            }
            
            # Update conversation history if available
            if conversation_history is not None and config.rag.conversation_history:
                conversation_history.append({
                    "query": query,
                    "answer": response,
                    "context_snippets": len(code_snippets)
                })
                
                # Trim history if needed
                if len(conversation_history) > config.rag.max_history_turns:
                    conversation_history = conversation_history[-config.rag.max_history_turns:]
            
            self.logger.info(f"Successfully processed query with {len(code_snippets)} code snippets")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _prepare_context(
        self,
        code_snippets: List[Dict[str, Any]],
        intent: str
    ) -> str:
        """
        Prepare context for the LLM from code snippets.
        
        Args:
            code_snippets: List of code snippets with metadata
            intent: Query intent
            
        Returns:
            Formatted context string
        """
        # Sort snippets by relevance
        sorted_snippets = sorted(
            code_snippets,
            key=lambda x: x["relevance"],
            reverse=True
        )
        
        # Limit number of snippets based on context window
        context_limit = min(config.rag.context_window, len(sorted_snippets))
        selected_snippets = sorted_snippets[:context_limit]
        
        # Format context
        context_parts = []
        
        for i, snippet in enumerate(selected_snippets):
            # Format metadata
            metadata_str = ""
            if "file_path" in snippet["metadata"]:
                metadata_str += f"File: {snippet['metadata']['file_path']}\n"
            if "language" in snippet:
                metadata_str += f"Language: {snippet['language']}\n"
            if "start_line" in snippet["metadata"] and "end_line" in snippet["metadata"]:
                metadata_str += f"Lines: {snippet['metadata']['start_line']}-{snippet['metadata']['end_line']}\n"
            
            # Format snippet
            context_parts.append(
                f"--- Code Snippet {i+1} ---\n"
                f"{metadata_str}\n"
                f"```{snippet.get('language', '')}\n"
                f"{snippet['content']}\n"
                f"```\n"
            )
        
        # Customize prefix based on intent while ensuring clear natural language responses
        if intent == "explain":
            prefix = "Based on the code snippets below, provide a clear, conversational explanation that directly answers the user's question. Focus on explaining concepts, patterns, or techniques found in the code in simple language accessible to someone learning programming.\n\n"
        elif intent == "bug":
            prefix = "Based on the code snippets below, analyze for potential bugs or issues. Start with a clear, direct answer about what problems exist, then explain why they're problematic and how they might be fixed, using natural language accessible to someone learning programming.\n\n"
        elif intent == "feature":
            prefix = "Based on the code snippets below, explain how a new feature could be implemented. Start with a direct answer to the user's question, then provide a clear explanation of the approach, using natural language accessible to someone learning programming.\n\n"
        elif intent == "usage":
            prefix = "Based on the code snippets below, explain how to use this code. Provide a clear, step-by-step explanation in natural language that's accessible to someone learning programming. Start with a direct answer to the user's question.\n\n"
        else:
            prefix = "Based on the code snippets below, provide a concise, clear explanation in natural language that directly answers the user's question. Focus on explaining concepts, patterns, or techniques found in the code rather than describing the code line by line. Start with a direct answer to the user's question.\n\n"
        
        return prefix + "\n".join(context_parts)
    
    async def reformulate_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Reformulate a query to make it more effective.
        
        Args:
            query: Original user query
            conversation_history: Optional conversation history
            
        Returns:
            Reformulated query
        """
        if not config.rag.query_reformulation:
            return query
            
        try:
            # Use LLM to reformulate the query
            prompt = self._create_reformulation_prompt(query, conversation_history)
            response = self.llm_client.query(prompt, "", {})
            
            # Extract reformulated query
            reformulated_query = response.strip()
            
            # Validate and use reformulated query
            if reformulated_query and len(reformulated_query) > 5:
                self.logger.info(f"Reformulated query: '{query}' -> '{reformulated_query}'")
                return reformulated_query
            else:
                self.logger.warning(f"Query reformulation failed, using original query")
                return query
                
        except Exception as e:
            self.logger.error(f"Error in query reformulation: {str(e)}")
            return query
    
    def _create_reformulation_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Create a prompt for query reformulation."""
        prompt = "Please reformulate the following query to make it more effective for searching code:\n\n"
        prompt += f"Query: {query}\n\n"
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt += "Previous conversation:\n"
            # Add the last 2 turns
            history = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
            for turn in history:
                prompt += f"User: {turn['query']}\n"
                prompt += f"Assistant: {turn['answer'][:100]}...\n"
            prompt += "\n"
        
        prompt += "Reformulated query:"
        return prompt 