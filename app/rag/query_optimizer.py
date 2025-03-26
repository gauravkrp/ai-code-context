"""
Advanced query optimization for code-specific queries.
"""
import logging
import re
import string
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer
from app.utils.code_embeddings import CodeEmbedder, EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class QueryOptimizationConfig:
    """Configuration for query optimization."""
    use_query_expansion: bool = True
    use_bm25: bool = True
    use_code_specific_terms: bool = True
    use_hybrid_search: bool = True
    intent_detection: bool = True
    expansion_terms: int = 3
    code_boost_factor: float = 1.5
    max_query_length: int = 1024
    min_token_length: int = 2

class QueryIntent:
    """Types of query intents for code-specific queries."""
    EXPLAIN = "explain"
    BUG = "bug"
    FEATURE = "feature"
    USAGE = "usage"
    IMPLEMENTATION = "implementation"
    DOCUMENT = "document"
    GENERAL = "general"

class QueryOptimizer:
    """Advanced query optimizer for code-specific queries."""
    
    # Code-specific terminology patterns
    CODE_PATTERNS = {
        "function_ref": r'\b\w+\((?:\w+)?\)',
        "variable_ref": r'\b[a-zA-Z_]\w*\b',
        "class_ref": r'\b[A-Z]\w+\b',
        "path_ref": r'[\w\/\\\.]+\.\w+',
        "code_snippet": r'`[^`]+`',
        "error_message": r'error|exception|traceback|failed|bug',
        "method_call": r'\.\w+\((?:\w+)?\)',
        "file_ref": r'\b\w+\.\w+\b'
    }
    
    # Language-specific terminology
    LANGUAGE_TERMS = {
        "python": ["def", "class", "import", "from", "with", "as", "except", "try", "finally"],
        "javascript": ["function", "const", "let", "var", "import", "export", "async", "await"],
        "typescript": ["interface", "type", "enum", "namespace", "declare", "extends", "implements"],
        "java": ["public", "private", "protected", "class", "interface", "extends", "implements"],
        "kotlin": ["fun", "val", "var", "companion", "object", "override", "suspend", "internal"]
    }
    
    # Term boost factors
    TERM_BOOST = {
        "function_ref": 2.0,
        "class_ref": 1.8,
        "method_call": 1.7,
        "path_ref": 1.6,
        "file_ref": 1.5,
        "error_message": 1.4,
        "code_snippet": 1.3,
        "variable_ref": 1.2
    }
    
    # Query intent patterns
    INTENT_PATTERNS = {
        QueryIntent.EXPLAIN: [
            r'explain', r'how does', r'what is', r'how is', r'what does', r'understand', r'describe'
        ],
        QueryIntent.BUG: [
            r'bug', r'error', r'issue', r'fix', r'crash', r'problem', r'exception', r'incorrect', r'wrong'
        ],
        QueryIntent.FEATURE: [
            r'feature', r'implement', r'add', r'enhance', r'improve', r'create', r'build', r'develop'
        ],
        QueryIntent.USAGE: [
            r'how to', r'usage', r'example', r'call', r'invoke', r'use', r'example', r'parameter', r'argument'
        ],
        QueryIntent.IMPLEMENTATION: [
            r'implementation', r'design', r'architecture', r'pattern', r'structure', r'class diagram'
        ],
        QueryIntent.DOCUMENT: [
            r'document', r'documentation', r'comment', r'explanation', r'doc', r'help', r'guide'
        ]
    }
    
    def __init__(self, config: QueryOptimizationConfig):
        """Initialize the query optimizer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize vectorizer for BM25
        self.vectorizer = CountVectorizer(
            min_df=1,
            stop_words='english',
            token_pattern=r'[a-zA-Z0-9_\.\-]+',
            lowercase=True
        )
        
        # Initialize code embedder for hybrid search
        if self.config.use_hybrid_search:
            self.code_embedder = CodeEmbedder(
                EmbeddingConfig(
                    model_name="codebert",
                    device="cpu",
                    batch_size=1,
                    max_length=self.config.max_query_length
                )
            )
    
    def optimize_query(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize a query for code retrieval.
        
        Args:
            query: The original user query
            context: Optional code context from previous searches
            conversation_history: Optional conversation history
            
        Returns:
            Dict with optimized query and search parameters
        """
        self.logger.info(f"Optimizing query: {query}")
        
        # Preprocess the query
        cleaned_query = self._preprocess_query(query)
        
        # Detect query intent
        intent = self._detect_intent(cleaned_query)
        self.logger.debug(f"Detected intent: {intent}")
        
        # Extract code-specific terms
        code_terms = self._extract_code_terms(cleaned_query)
        self.logger.debug(f"Extracted code terms: {code_terms}")
        
        # Expand query with relevant terms
        expanded_query = self._expand_query(
            cleaned_query, 
            code_terms, 
            intent, 
            context, 
            conversation_history
        )
        
        # Prepare hybrid search parameters
        search_params = self._prepare_search_params(intent, code_terms, context)
        
        # Prepare result
        result = {
            "original_query": query,
            "optimized_query": expanded_query,
            "intent": intent,
            "code_terms": code_terms,
            "search_params": search_params
        }
        
        # Prepare embeddings for hybrid search if enabled
        if self.config.use_hybrid_search:
            result["query_embedding"] = self._generate_query_embedding(
                expanded_query, intent, code_terms
            )
        
        self.logger.info(f"Query optimized: {expanded_query}")
        return result
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and clean the query."""
        # Truncate if too long
        if len(query) > self.config.max_query_length:
            query = query[:self.config.max_query_length]
        
        # Remove punctuation except for dots (for method calls) and underscores
        translator = str.maketrans('', '', string.punctuation.replace('.', '').replace('_', ''))
        query = query.translate(translator)
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        if not self.config.intent_detection:
            return QueryIntent.GENERAL
        
        # Check each intent pattern
        intent_scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.finditer(pattern, query.lower())
                score += sum(1 for _ in matches)
            intent_scores[intent] = score
        
        # Get the intent with the highest score
        if max(intent_scores.values(), default=0) > 0:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return QueryIntent.GENERAL
    
    def _extract_code_terms(self, query: str) -> Dict[str, List[str]]:
        """Extract code-specific terms from the query."""
        result = {}
        
        if not self.config.use_code_specific_terms:
            return result
        
        # Extract code patterns
        for pattern_type, pattern in self.CODE_PATTERNS.items():
            matches = re.finditer(pattern, query)
            result[pattern_type] = [match.group() for match in matches]
        
        return result
    
    def _expand_query(
        self,
        query: str,
        code_terms: Dict[str, List[str]],
        intent: str,
        context: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Expand the query with relevant terms."""
        if not self.config.use_query_expansion:
            return query
        
        expanded_terms = []
        
        # Add language-specific terms based on intent
        if intent in [QueryIntent.IMPLEMENTATION, QueryIntent.FEATURE]:
            # Detect language from context if available
            language = self._detect_language_from_context(context)
            if language and language in self.LANGUAGE_TERMS:
                relevant_terms = self.LANGUAGE_TERMS[language]
                for term in relevant_terms:
                    if term.lower() in query.lower():
                        expanded_terms.append(term)
        
        # Add code-specific terms with boost factors
        for term_type, terms in code_terms.items():
            boost_factor = self.TERM_BOOST.get(term_type, 1.0)
            if boost_factor > 1.0:
                for term in terms:
                    # Remove parentheses from function calls
                    clean_term = re.sub(r'\([^)]*\)', '', term).strip()
                    if clean_term and len(clean_term) >= self.config.min_token_length:
                        expanded_terms.append(clean_term)
        
        # Add terms from conversation history if available
        if conversation_history and len(conversation_history) > 0:
            last_exchanges = conversation_history[-2:]  # Last 2 exchanges
            for exchange in last_exchanges:
                if "answer" in exchange:
                    # Extract keywords from answer
                    answer = exchange["answer"]
                    keywords = self._extract_keywords(answer, 2)
                    expanded_terms.extend(keywords)
        
        # Build expanded query
        expanded_query = query
        if expanded_terms:
            # Remove duplicates and limit number of expansion terms
            expanded_terms = list(set(expanded_terms))[:self.config.expansion_terms]
            expanded_query = f"{query} {' '.join(expanded_terms)}"
        
        return expanded_query
    
    def _detect_language_from_context(
        self, context: Optional[List[Dict[str, Any]]]
    ) -> Optional[str]:
        """Detect programming language from context."""
        if not context:
            return None
        
        # Count languages in context
        language_counts = {}
        for doc in context:
            if "metadata" in doc and "language" in doc["metadata"]:
                lang = doc["metadata"]["language"].lower()
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Return the most common language
        if language_counts:
            return max(language_counts.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _extract_keywords(self, text: str, n: int = 3) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction by removing stopwords and taking most frequent
        if not text:
            return []
        
        # Tokenize and count
        words = re.findall(r'\b[a-zA-Z_]\w+\b', text.lower())
        
        # Remove common English stopwords
        stopwords = {
            'the', 'and', 'a', 'to', 'of', 'is', 'in', 'that', 'it', 'with',
            'as', 'for', 'this', 'was', 'be', 'on', 'are', 'by', 'an', 'not'
        }
        words = [w for w in words if w not in stopwords and len(w) >= self.config.min_token_length]
        
        # Count and sort
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Return top N keywords
        return [w for w, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n]]
    
    def _prepare_search_params(
        self,
        intent: str,
        code_terms: Dict[str, List[str]],
        context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Prepare search parameters based on query analysis."""
        search_params = {
            "n_results": 5,
            "filter_criteria": None,
            "rerank": False
        }
        
        # Adjust parameters based on intent
        if intent == QueryIntent.BUG:
            search_params["n_results"] = 8
            search_params["rerank"] = True
        elif intent == QueryIntent.IMPLEMENTATION:
            search_params["n_results"] = 10
            search_params["rerank"] = True
        elif intent == QueryIntent.EXPLAIN:
            search_params["n_results"] = 5
        
        # Add filters based on code terms
        if code_terms.get("file_ref") or code_terms.get("path_ref"):
            file_refs = code_terms.get("file_ref", []) + code_terms.get("path_ref", [])
            if file_refs:
                search_params["filter_criteria"] = {
                    "file_path": {"$in": file_refs}
                }
        
        # Detect language from context
        language = self._detect_language_from_context(context)
        if language:
            search_params.setdefault("filter_criteria", {})
            search_params["filter_criteria"]["language"] = language
        
        return search_params
    
    def _generate_query_embedding(
        self,
        query: str,
        intent: str,
        code_terms: Dict[str, List[str]]
    ) -> np.ndarray:
        """Generate an embedding for the query optimized for code search."""
        # Prioritize code terms in the query
        code_query = query
        
        # Add specific code terms with higher priority
        high_priority_terms = []
        for term_type in ["function_ref", "class_ref", "method_call", "file_ref"]:
            if term_type in code_terms and code_terms[term_type]:
                high_priority_terms.extend(code_terms[term_type])
        
        if high_priority_terms:
            # Double the high priority terms at the beginning for emphasis
            high_priority_str = " ".join(high_priority_terms)
            code_query = f"{high_priority_str} {high_priority_str} {query}"
        
        # Generate embedding
        embedding = self.code_embedder.embed_code([code_query])[0]
        
        return embedding
    
    def rerank_results(
        self,
        results: List[Dict[str, Any]],
        query_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rerank results using BM25 and other factors."""
        if not results or not self.config.use_bm25:
            return results
        
        # Extract code terms for boosting
        code_terms = query_analysis.get("code_terms", {})
        intent = query_analysis.get("intent", QueryIntent.GENERAL)
        
        # Extract documents and create corpus
        documents = [doc["content"] for doc in results]
        
        # Fit vectorizer
        try:
            X = self.vectorizer.fit_transform(documents)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Create query vector
            optimized_query = query_analysis["optimized_query"]
            query_vec = self.vectorizer.transform([optimized_query])
            
            # Calculate BM25 scores
            k1 = 1.5
            b = 0.75
            
            # TF component
            tf = X.toarray()
            
            # Document length
            doc_len = tf.sum(axis=1)
            avg_doc_len = doc_len.mean()
            
            # IDF component
            N = len(documents)
            df = np.asarray((X > 0).sum(axis=0)).squeeze()
            idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
            
            # BM25 score calculation
            bm25_scores = []
            for i, doc in enumerate(tf):
                score = 0
                for j, term in enumerate(query_vec.indices):
                    # Calculate BM25 score
                    term_freq = doc[term]
                    term_score = idf[term] * ((term_freq * (k1 + 1)) / 
                                              (term_freq + k1 * (1 - b + b * doc_len[i] / avg_doc_len)))
                    
                    # Boost score for code terms
                    term_text = feature_names[term]
                    for term_type, terms in code_terms.items():
                        if any(term_text in t for t in terms):
                            boost = self.TERM_BOOST.get(term_type, 1.0)
                            term_score *= boost
                    
                    score += term_score
                bm25_scores.append(score)
            
            # Add BM25 scores to results
            for i, result in enumerate(results):
                result["bm25_score"] = float(bm25_scores[i])
            
            # Combine BM25 with vector similarity
            for result in results:
                vector_score = 1.0 - result["distance"]  # Convert distance to similarity
                bm25_score = result["bm25_score"] / max(bm25_scores) if max(bm25_scores) > 0 else 0
                
                # Weighted combination
                if intent in [QueryIntent.BUG, QueryIntent.IMPLEMENTATION]:
                    # For bugs and implementation, prioritize BM25
                    result["combined_score"] = 0.6 * bm25_score + 0.4 * vector_score
                else:
                    # For other intents, balance more
                    result["combined_score"] = 0.5 * bm25_score + 0.5 * vector_score
            
            # Rerank results
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error in BM25 reranking: {str(e)}")
        
        return results 