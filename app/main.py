"""
Main application for code indexing and querying with enhanced RAG capabilities.
"""
import argparse
import logging
from typing import List, Optional
import time
import uuid
import json
import traceback
from datetime import datetime

from app.config.settings import config
from app.github.repo_scanner import GitHubScanner
from app.vector_store.chroma_store import ChromaStore
from app.utils.llm import LLMClient
from app.rag.advanced_rag import AdvancedRAG
from app.rag.code_explainer import CodeExplainer
from app.analytics.monitor import AnalyticsMonitor
from app.db.database import get_db_session, close_db_session
from app.db.models import Repository, VectorStoreLink

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
    parser.add_argument("command", choices=["index", "query", "chat"], help="Command to execute")
    parser.add_argument("--repo", help="GitHub repository URL (overrides GITHUB_REPOSITORY from .env)")
    parser.add_argument("--branch", help="GitHub branch name (overrides GITHUB_BRANCH from .env)")
    parser.add_argument("--query", help="Query string")
    parser.add_argument("--history", help="JSON string of conversation history")
    parser.add_argument("--generate-docs", action="store_true", help="Generate documentation")
    parser.add_argument("--explain", action="store_true", help="Include explanations of code snippets")
    parser.add_argument("--show-snippets", action="store_true", help="Show code snippets in response")
    parser.add_argument("--history-file", help="File to save/load chat history")
    parser.add_argument("--force-full", action="store_true", help="Force full re-indexing instead of incremental update")
    parser.add_argument("--list-repos", action="store_true", help="List indexed repositories")
    args = parser.parse_args()

    # Handle listing repositories
    if args.list_repos and args.command == "index":
        list_indexed_repositories()
        return

    # Handle index command
    if args.command == "index":
        index_command(args)
    
    # Handle query command
    elif args.command == "query":
        if not args.query:
            parser.error("--query is required for query command")
        query_command(args)
    
    # Handle chat command
    elif args.command == "chat":
        chat_command(args)

def list_indexed_repositories():
    """List all indexed repositories."""
    db = get_db_session()
    try:
        repos = db.query(Repository).all()
        
        if not repos:
            print("No repositories indexed yet.")
            return
            
        print("\nIndexed Repositories:")
        print("─" * 80)
        print(f"{'ID':<36} | {'Name':<20} | {'Last Indexed':<20} | {'Status':<10}")
        print("─" * 80)
        
        for repo in repos:
            status = repo.repo_metadata.get("indexing_status", "unknown") if repo.repo_metadata else "unknown"
            last_indexed = repo.last_indexed.strftime("%Y-%m-%d %H:%M") if repo.last_indexed else "Never"
            print(f"{str(repo.id):<36} | {repo.name:<20} | {last_indexed:<20} | {status:<10}")
            
        print("─" * 80)
    finally:
        close_db_session(db)

def index_command(args):
    """Handle the index command."""
    from app.github.repo_scanner import GitHubScanner
    from app.vector_store.chroma_store import ChromaStore
    from app.db.database import get_db_session, close_db_session
    from app.db.models import Repository, User
    
    # Get repository info from args or config
    repo_url = args.repo or config.github.repository
    branch = args.branch or config.github.branch
    
    if not repo_url:
        print("Error: No repository specified. Use --repo option or set GITHUB_REPOSITORY in .env")
        return
        
    print(f"Indexing repository: {repo_url} (branch: {branch})")
    
    # Connect to database
    db = get_db_session()
    
    try:
        # Check if repository exists in database
        repo_parts = repo_url.split('/')
        repo_name = repo_parts[-1] if len(repo_parts) >= 1 else repo_url
        
        # Get or create admin user
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                id=uuid.uuid4(),
                username="admin",
                email="admin@example.com",
                password_hash="admin",  # This is just for CLI usage, would be secure in production
                is_admin=True
            )
            db.add(admin_user)
            db.commit()
        
        # Look for existing repository with matching URL and branch
        repository = db.query(Repository).filter(
            Repository.url == repo_url,
            Repository.branch == branch
        ).first()
        
        incremental = not args.force_full
        
        if repository:
            print(f"Repository already exists in database (ID: {repository.id})")
            
            if repository.last_indexed and incremental:
                print(f"Last indexed: {repository.last_indexed}")
                print("Performing incremental update...")
            else:
                print("Performing full re-indexing...")
                
        else:
            # Create new repository
            repository = Repository(
                id=uuid.uuid4(),
                owner_id=admin_user.id,
                name=repo_name,
                url=repo_url,
                branch=branch,
                is_public=True,
                repo_metadata={
                    "indexing_status": "pending",
                    "created_via": "cli"
                }
            )
            db.add(repository)
            db.commit()
            print(f"Created new repository entry (ID: {repository.id})")
            incremental = False  # Force full indexing for new repositories
        
        # Update repository status
        repository.repo_metadata = {
            **(repository.repo_metadata or {}),
            "indexing_status": "in_progress",
            "last_error": None
        }
        db.commit()
        
        # Initialize components
        scanner = GitHubScanner(config.github.access_token)
        store = ChromaStore(repo_id=f"{repository.owner_id}_{repository.id}")
        
        # Get last indexed time for incremental indexing
        last_indexed = repository.last_indexed if incremental else None
        
        # Scan repository for documents
        print("Scanning repository files...")
        documents = scanner.scan_repository(
            repo_url, 
            branch,
            since=last_indexed
        )
        
        if last_indexed and incremental:
            print(f"Found {len(documents)} documents modified since {last_indexed}")
        else:
            print(f"Performing full indexing, found {len(documents)} documents")
        
        if not documents:
            print("No new or modified documents found.")
            repository.repo_metadata = {
                **(repository.repo_metadata or {}),
                "indexing_status": "completed",
                "last_error": None
            }
            db.commit()
            return
            
        # Add documents to vector store
        print("Adding documents to vector store...")
        store.add_documents(documents, incremental=incremental)
        
        # Create links between repository and vector store entries
        for doc in documents:
            vector_link = db.query(VectorStoreLink).filter(
                VectorStoreLink.entity_id == repository.id,
                VectorStoreLink.vector_id == str(doc.id)
            ).first()
            
            if not vector_link:
                vector_link = VectorStoreLink(
                    id=uuid.uuid4(),
                    entity_type="repository_file",
                    entity_id=repository.id,
                    vector_id=str(doc.id),
                    collection_name=store.collection_name,
                    link_metadata={
                        "file_path": doc.metadata.get("file_path", ""),
                        "language": doc.metadata.get("language", ""),
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
                db.add(vector_link)
        
        # Update repository status
        repository.last_indexed = datetime.utcnow()
        repository.repo_metadata = {
            **(repository.repo_metadata or {}),
            "indexing_status": "completed",
            "last_error": None,
            "document_count": len(documents),
            "incremental": incremental
        }
        db.commit()
        
        print(f"Repository indexed successfully: {len(documents)} documents processed")
        
    except Exception as e:
        print(f"Error indexing repository: {str(e)}")
        traceback.print_exc()
        
        # Update repository status to indicate failure
        try:
            if repository:
                repository.repo_metadata = {
                    **(repository.repo_metadata or {}),
                    "indexing_status": "failed",
                    "last_error": str(e)
                }
                db.commit()
        except Exception as db_error:
            print(f"Error updating repository status: {str(db_error)}")
    
    finally:
        close_db_session(db)

def query_command(args):
    """Handle the query command."""
    from app.rag.advanced_rag import AdvancedRAG
    from app.vector_store.chroma_store import ChromaStore
    from app.utils.llm import LLMClient
    from app.db.database import get_db_session, close_db_session
    from app.db.models import Repository
    
    # Get repository info
    repo_url = args.repo or config.github.repository
    
    # Parse conversation history
    history = []
    if args.history:
        try:
            history = json.loads(args.history)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in history argument")
            return
    
    # Get query options
    show_snippets = args.show_snippets
    explain = args.explain
    generate_docs = args.generate_docs
    
    db = get_db_session()
    try:
        # Find repository in database
        repository = None
        if repo_url:
            repository = db.query(Repository).filter(
                Repository.url == repo_url
            ).first()
            
        if repository:
            print(f"Using repository: {repository.name} (ID: {repository.id})")
            
            # Initialize components with repository-specific collection
            store = ChromaStore(repo_id=f"{repository.owner_id}_{repository.id}")
            llm_client = LLMClient()
            rag = AdvancedRAG(store, llm_client)
            
            # Process query
            print(f"\nQuery: {args.query}\n")
            response = rag.query(
                args.query, 
                history,
                show_snippets=show_snippets,
                explain=explain,
                generate_docs=generate_docs
            )
            
            # Print response
            print(response["answer"])
            
            # Print snippets if requested
            if show_snippets and response.get("code_snippets"):
                print("\nCode Snippets:")
                for i, snippet in enumerate(response["code_snippets"]):
                    print(f"\n--- Snippet {i+1} ---")
                    print(f"File: {snippet.file_path}")
                    print(f"Language: {snippet.language}")
                    print("\n```")
                    print(snippet.code)
                    print("```")
                    
                    # Print explanation if requested
                    if explain and response.get("explanations") and i < len(response["explanations"]):
                        explanation = response["explanations"][i]
                        print("\nExplanation:")
                        print(explanation.explanation)
                        
                        if explanation.best_practices:
                            print("\nBest Practices:")
                            for practice in explanation.best_practices:
                                print(f"- {practice}")
        else:
            print("Error: Repository not found in database. Please index it first with the 'index' command.")
    finally:
        close_db_session(db)

def chat_command(args):
    """Handle the chat command."""
    from app.db.database import get_db_session, close_db_session
    from app.db.models import Repository
    
    # Get repository info from args or fall back to config
    repo_url = args.repo or config.github.repository
    
    if not repo_url:
        print("Error: No repository specified. Use --repo option or set GITHUB_REPOSITORY in .env")
        return
    
    # Get database session to find repository
    db = get_db_session()
    try:
        # Find repository in database
        repository = None
        if repo_url:
            repository = db.query(Repository).filter(
                Repository.url == repo_url
            ).first()
            
        if repository:
            print(f"Starting chat session for repository: {repository.name} (ID: {repository.id})")
            
            # Run chat mode with proper repository context
            run_chat_mode(
                show_snippets=args.show_snippets,
                explain_code=args.explain,
                generate_docs=args.generate_docs,
                history_file=args.history_file,
                repository_id=repository.id,
                owner_id=repository.owner_id
            )
        else:
            print("Error: Repository not found in database. Please index it first with the 'index' command.")
    finally:
        close_db_session(db)

def run_chat_mode(show_snippets: bool = False, explain_code: bool = False, 
                  generate_docs: bool = False, history_file: Optional[str] = None,
                  repository_id: Optional[uuid.UUID] = None, owner_id: Optional[uuid.UUID] = None):
    """
    Run an interactive chat session with the codebase.
    
    Args:
        show_snippets: Whether to show code snippets in responses
        explain_code: Whether to include code explanations
        generate_docs: Whether to generate documentation
        history_file: File to save and load chat history
        repository_id: ID of the repository to query
        owner_id: ID of the repository owner
    """
    import json
    from datetime import datetime
    from app.vector_store.chroma_store import ChromaStore
    from app.utils.llm import LLMClient
    from app.rag.advanced_rag import AdvancedRAG
    
    # Initialize conversation history
    conversation_history = []
    
    # Load history from file if provided
    if history_file:
        try:
            with open(history_file, 'r') as f:
                conversation_history = json.load(f)
                if conversation_history:
                    print(f"Loaded {len(conversation_history)} previous messages.")
        except (FileNotFoundError, json.JSONDecodeError):
            # Create a new history file
            with open(history_file, 'w') as f:
                json.dump([], f)
    
    # Initialize components with repository-specific collection if repository_id provided
    store = None
    llm_client = None
    rag = None
    
    if repository_id and owner_id:
        # Using repository-specific collection
        store = ChromaStore(repo_id=f"{owner_id}_{repository_id}")
        llm_client = LLMClient()
        rag = AdvancedRAG(store, llm_client)
    else:
        # Using default collection
        store = ChromaStore()
        llm_client = LLMClient()
        rag = AdvancedRAG(store, llm_client)
    
    # Welcome message
    print("\n===== AI Code Context Chat Mode =====")
    print("Ask questions about the codebase. Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'clear' to clear the conversation history.")
    print("Type 'help' to see additional commands.")
    print("=======================================\n")
    
    while True:
        try:
            # Get user input
            user_query = input("\n> ")
            
            # Check for exit commands
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("Exiting chat mode.")
                break
                
            # Check for clear command
            if user_query.lower() == 'clear':
                conversation_history = []
                print("Conversation history cleared.")
                continue
                
            # Check for help command
            if user_query.lower() == 'help':
                print("\nAvailable commands:")
                print("  exit, quit, q - Exit chat mode")
                print("  clear - Clear conversation history")
                print("  help - Show this help message")
                print("  snippets on/off - Toggle code snippet display")
                print("  explain on/off - Toggle code explanations")
                continue
                
            # Check for settings commands
            if user_query.lower() == 'snippets on':
                show_snippets = True
                print("Code snippets enabled.")
                continue
            elif user_query.lower() == 'snippets off':
                show_snippets = False
                print("Code snippets disabled.")
                continue
            elif user_query.lower() == 'explain on':
                explain_code = True
                print("Code explanations enabled.")
                continue
            elif user_query.lower() == 'explain off':
                explain_code = False
                print("Code explanations disabled.")
                continue
                
            # Skip empty queries
            if not user_query.strip():
                continue
                
            # Process the query
            print("Processing query...")
            
            # Use RAG system directly if available
            if rag:
                response = rag.query(
                    user_query,
                    conversation_history,
                    show_snippets=show_snippets,
                    explain=explain_code,
                    generate_docs=generate_docs
                )
            else:
                # Fall back to query_codebase
                response = query_codebase(
                    user_query,
                    conversation_history,
                    generate_docs,
                    explain_code
                )
            
            # Print response
            print("\nResponse:")
            print(response["answer"])
            
            # Add to conversation history
            conversation_turn = {
                "query": user_query,
                "answer": response["answer"],
                "timestamp": datetime.now().isoformat()
            }
            conversation_history.append(conversation_turn)
            
            # Save history to file if provided
            if history_file:
                with open(history_file, 'w') as f:
                    json.dump(conversation_history, f)
            
            # Only show code snippets if requested
            if show_snippets and response.get("code_snippets"):
                print("\nRelevant Code Snippets:")
                for i, snippet in enumerate(response["code_snippets"]):
                    print(f"\n--- Snippet {i+1} ---")
                    
                    # Handle both object and dict format
                    if hasattr(snippet, 'file_path'):
                        print(f"File: {snippet.file_path}")
                        print(f"Language: {snippet.language}")
                        print("\n```")
                        print(snippet.code)
                    else:
                        print(f"File: {snippet['metadata']['file_path']}")
                        print("```")
                        print(snippet["content"])
                    
                    print("```")
            
            if explain_code and response.get("explanations"):
                print("\nCode Explanations:")
                for i, explanation in enumerate(response["explanations"]):
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
                                
        except KeyboardInterrupt:
            print("\nExiting chat mode.")
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 