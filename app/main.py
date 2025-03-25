"""
Main application module for the GitHub Code Search and Q&A system.

This module provides the entry point for both the indexing and query components.
"""
import logging
import argparse
import sys
import os
from typing import List, Dict, Any, Optional

from app.config.settings import config
from app.github.repository import GitHubRepository
from app.github.indexer import RepositoryIndexer
from app.vector_store.chroma_store import ChromaVectorStore
from app.utils.llm import LLMHandler
from app.slack.bot import SlackBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description='GitHub Code Search and Q&A System')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index a GitHub repository')
    index_parser.add_argument('--repo', help='GitHub repository to index in the format "owner/repo"')
    index_parser.add_argument('--clear', action='store_true', help='Clear existing index before indexing')
    index_parser.add_argument('--extensions', nargs='+', help='File extensions to index (e.g., .py .js)')
    
    # Start Slack bot command
    slack_parser = subparsers.add_parser('slack', help='Start the Slack bot')
    slack_parser.add_argument('--socket-mode', action='store_true', help='Use Socket Mode for Slack (requires app token)')
    
    # Direct query command
    query_parser = subparsers.add_parser('query', help='Directly query the code database')
    query_parser.add_argument('question', help='The question to ask about the codebase')
    
    # Web server command
    server_parser = subparsers.add_parser('server', help='Start the web server API')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    
    return parser

def index_repository(repo_name: str = None, clear_existing: bool = False, extensions: List[str] = None) -> None:
    """
    Index a GitHub repository.
    
    Args:
        repo_name: Name of the repository to index in the format "owner/repo".
        clear_existing: Whether to clear the existing index.
        extensions: List of file extensions to index.
    """
    try:
        # Use provided repo name or from config
        repository_name = repo_name or config.github.repository
        
        if not repository_name:
            logger.error("Repository name not provided and not found in configuration")
            return
        
        # Create repository and indexer instances
        repository = GitHubRepository(repository=repository_name)
        vector_store = ChromaVectorStore()
        
        # Format file extensions if provided
        file_exts = None
        if extensions:
            file_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        # Create and run the indexer
        indexer = RepositoryIndexer(
            repository=repository,
            vector_store=vector_store,
            file_extensions=file_exts
        )
        
        files_count, chunks_count = indexer.index_repository(clear_existing=clear_existing)
        
        logger.info(f"Successfully indexed {files_count} files and created {chunks_count} chunks")
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        sys.exit(1)

def start_slack_bot(use_socket_mode: bool = False) -> None:
    """
    Start the Slack bot.
    
    Args:
        use_socket_mode: Whether to use Socket Mode.
    """
    try:
        # Initialize LLM handler and vector store
        llm_handler = LLMHandler()
        vector_store = ChromaVectorStore()
        
        # Check if the vector store has data
        if vector_store.collection.count() == 0:
            logger.warning("Vector store is empty. Please index a repository first.")
            sys.exit(1)
        
        # Create and start Slack bot
        slack_bot = SlackBot(
            llm_handler=llm_handler,
            vector_store=vector_store
        )
        
        logger.info(f"Starting Slack bot in {'Socket Mode' if use_socket_mode else 'HTTP mode'}")
        slack_bot.start(use_socket_mode=use_socket_mode)
        
    except Exception as e:
        logger.error(f"Error starting Slack bot: {e}")
        sys.exit(1)

def direct_query(question: str) -> None:
    """
    Execute a direct query against the code database.
    
    Args:
        question: The question to ask.
    """
    try:
        # Initialize LLM handler and vector store
        llm_handler = LLMHandler()
        vector_store = ChromaVectorStore()
        
        # Check if the vector store has data
        if vector_store.collection.count() == 0:
            logger.warning("Vector store is empty. Please index a repository first.")
            sys.exit(1)
        
        logger.info(f"Querying: {question}")
        
        # Search the vector store
        search_results = vector_store.search(question, n_results=5)
        
        if not search_results:
            print("No relevant code found for your question.")
            return
        
        # Query the LLM
        response = llm_handler.query(question, search_results)
        
        # Print the response
        print("\n=== Answer ===\n")
        print(response)
        print("\n=== Sources ===\n")
        
        for i, result in enumerate(search_results[:3]):
            metadata = result.get('metadata', {})
            file_path = metadata.get('file_path', 'Unknown file')
            start_line = metadata.get('start_line', 1)
            end_line = metadata.get('end_line', start_line)
            
            print(f"{i+1}. {file_path} (lines {start_line}-{end_line})")
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        sys.exit(1)

def start_web_server(host: str = '0.0.0.0', port: int = 8000) -> None:
    """
    Start the web server API.
    
    Args:
        host: Host to bind the server to.
        port: Port to bind the server to.
    """
    try:
        # Import web server components only when needed
        import uvicorn
        from fastapi import FastAPI, HTTPException, Depends
        from pydantic import BaseModel
        
        # Initialize API components
        llm_handler = LLMHandler()
        vector_store = ChromaVectorStore()
        
        # Check if the vector store has data
        if vector_store.collection.count() == 0:
            logger.warning("Vector store is empty. Please index a repository first.")
        
        # Define API request/response models
        class QueryRequest(BaseModel):
            question: str
            num_results: int = 5
        
        class QueryResponse(BaseModel):
            answer: str
            sources: List[Dict[str, Any]]
        
        # Create FastAPI app
        app = FastAPI(title="GitHub Code Search API")
        
        @app.post("/api/query", response_model=QueryResponse)
        async def query_api(request: QueryRequest):
            """API endpoint for querying the code database."""
            try:
                # Search the vector store
                search_results = vector_store.search(request.question, n_results=request.num_results)
                
                if not search_results:
                    return {"answer": "No relevant code found for your question.", "sources": []}
                
                # Query the LLM
                response = llm_handler.query(request.question, search_results)
                
                # Format sources for response
                sources = []
                for result in search_results:
                    metadata = result.get('metadata', {})
                    sources.append({
                        "file_path": metadata.get('file_path', 'Unknown'),
                        "start_line": metadata.get('start_line', 1),
                        "end_line": metadata.get('end_line', 1),
                        "similarity": result.get('similarity')
                    })
                
                return {"answer": response, "sources": sources}
                
            except Exception as e:
                logger.error(f"API error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Start the server
        logger.info(f"Starting API server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except Exception as e:
        logger.error(f"Error starting web server: {e}")
        sys.exit(1)

def main():
    """Main entry point for the application."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    if args.command == 'index':
        index_repository(
            repo_name=args.repo,
            clear_existing=args.clear,
            extensions=args.extensions
        )
    elif args.command == 'slack':
        start_slack_bot(use_socket_mode=args.socket_mode)
    elif args.command == 'query':
        direct_query(args.question)
    elif args.command == 'server':
        start_web_server(host=args.host, port=args.port)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 