"""
Main application for code indexing and querying with enhanced RAG capabilities.
"""
import argparse
import logging
from typing import List, Optional
import time

from app.config.settings import config
from app.github.repo_scanner import GitHubScanner
from app.vector_store.chroma_store import ChromaStore
from app.utils.llm import LLMClient
from app.rag.advanced_rag import AdvancedRAG
from app.rag.code_explainer import CodeExplainer
from app.analytics.monitor import AnalyticsMonitor

# Set up logging
logging.basicConfig(
    level=config.logging.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{config.logging.log_dir}/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def index_repository(repo_url: str, branch: Optional[str] = None) -> None:
    """Index a GitHub repository with enhanced code chunking."""
    try:
        logger.info(f"Starting repository scan: {repo_url}")
        
        # Initialize components
        scanner = GitHubScanner(config.github.access_token)
        store = ChromaStore()
        monitor = AnalyticsMonitor()
        
        # Scan repository
        documents = scanner.scan_repository(repo_url, branch)
        logger.info(f"Found {len(documents)} documents to process")
        
        # Add documents to vector store
        store.add_documents(documents)
        logger.info("Successfully indexed repository")
        
        # Record metrics
        monitor.record_system_metrics("indexing", {
            "documents_processed": len(documents),
            "repository": repo_url,
            "branch": branch or "main"
        })
        
    except Exception as e:
        logger.error(f"Error indexing repository: {str(e)}")
        raise

def query_codebase(
    query: str,
    conversation_history: Optional[List[dict]] = None,
    generate_docs: bool = False,
    explain_code: bool = False
) -> dict:
    """Query the codebase with enhanced RAG capabilities."""
    try:
        logger.info(f"Processing query: {query}")
        
        # Initialize components
        store = ChromaStore()
        llm_client = LLMClient()
        rag = AdvancedRAG(store, llm_client)
        monitor = AnalyticsMonitor()
        
        # Initialize explainer only if needed
        explainer = None
        if explain_code:
            explainer = CodeExplainer(store, llm_client)
        
        # Start timing the query
        start_time = time.time()
        
        # Process query with RAG
        response = rag.query(query, conversation_history)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Generate code explanations if needed
        if explain_code and response.get("code_snippets") and explainer:
            explanations = []
            for snippet in response["code_snippets"]:
                explanation = explainer.explain_code(
                    snippet["content"],
                    snippet.get("language", "python")
                )
                explanations.append(explanation)
            response["explanations"] = explanations
        
        # Calculate metrics
        num_results = len(response.get("code_snippets", []))
        
        # Calculate average similarity if results exist
        avg_similarity = 0.0
        if num_results > 0:
            avg_similarity = sum(snippet.get("relevance", 0) for snippet in response.get("code_snippets", [])) / num_results
        
        # Record metrics
        monitor.record_query_metrics(
            query=query,
            response_time=response_time,
            num_results=num_results,
            avg_similarity=avg_similarity
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="AI Code Context - Enhanced Code Indexing and Querying")
    parser.add_argument("command", choices=["index", "query"], help="Command to execute")
    parser.add_argument("--repo", help="GitHub repository URL (overrides GITHUB_REPOSITORY from .env)")
    parser.add_argument("--branch", help="GitHub branch name (overrides GITHUB_BRANCH from .env)")
    parser.add_argument("--query", help="Query string")
    parser.add_argument("--history", help="JSON string of conversation history")
    parser.add_argument("--generate-docs", action="store_true", help="Generate documentation")
    parser.add_argument("--explain", action="store_true", help="Explain code")
    parser.add_argument("--show-snippets", action="store_true", help="Show code snippets in the output")
    
    args = parser.parse_args()
    
    try:
        if args.command == "index":
            # Use repository from command line or config
            repo_url = args.repo or config.github.repository
            if not repo_url:
                raise ValueError("Repository URL is required. Either provide --repo or set GITHUB_REPOSITORY in .env")
            
            # Use branch from command line or config
            branch = args.branch or config.github.branch
            
            index_repository(repo_url, branch)
            logger.info("Indexing completed successfully")
            
        elif args.command == "query":
            if not args.query:
                raise ValueError("Query string is required")
            
            # Parse conversation history if provided
            history = None
            if args.history:
                import json
                history = json.loads(args.history)
            
            # Process query
            response = query_codebase(
                args.query,
                history,
                args.generate_docs,
                args.explain
            )
            
            # Print response
            print("\nResponse:")
            print(response["answer"])
            
            # Only show code snippets if requested
            if args.show_snippets and response.get("code_snippets"):
                print("\nRelevant Code Snippets:")
                for snippet in response["code_snippets"]:
                    print(f"\nFile: {snippet['metadata']['file_path']}")
                    print("```")
                    print(snippet["content"])
                    print("```")
            
            if args.explain and response.get("explanations"):
                print("\nCode Explanations:")
                for explanation in response["explanations"]:
                    # Handle CodeExplanation object properly
                    if hasattr(explanation, 'explanation'):
                        # It's a CodeExplanation object
                        print(f"\n{explanation.explanation}")
                        if hasattr(explanation, 'best_practices') and explanation.best_practices:
                            print("\nBest Practices:")
                            for practice in explanation.best_practices:
                                print(f"- {practice}")
                    else:
                        # Fallback for dictionary format
                        print(f"\n{explanation.get('summary', 'No summary available')}")
                        if explanation.get("improvements"):
                            print("\nSuggested Improvements:")
                            for improvement in explanation["improvements"]:
                                print(f"- {improvement}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 