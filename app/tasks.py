"""
Celery tasks for asynchronous operations.
"""
import logging
import os
from typing import Dict, Any, Optional
from celery import Celery
import uuid
from datetime import datetime

from app.config.settings import config
from app.github.repo_scanner import GitHubScanner
from app.vector_store.chroma_store import ChromaStore
from app.db.database import get_db_session, close_db_session
from app.db.models import Repository, VectorStoreLink

# Configure Celery
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("ai_code_context", broker=redis_url, backend=redis_url)

# Configure task routes
celery_app.conf.task_routes = {
    "app.tasks.index_repository": {"queue": "indexing"},
    "app.tasks.analyze_code": {"queue": "analysis"},
}

# Configure logging
logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="app.tasks.index_repository")
def index_repository(
    self,
    repository_id: str,
    repo_url: str,
    branch: Optional[str] = None,
    incremental: bool = True
) -> Dict[str, Any]:
    """
    Index a GitHub repository asynchronously.
    
    Args:
        repository_id: ID of the repository in the database
        repo_url: GitHub repository URL
        branch: Branch to index (default: repository default branch)
        incremental: If True, only index files changed since last indexing
        
    Returns:
        Dict with task status and repository data
    """
    logger.info(f"Starting repository indexing: {repo_url} (branch: {branch})")
    
    try:
        # Initialize components
        scanner = GitHubScanner(config.github.access_token)
        
        # Get database session
        db = get_db_session()
        
        # Update repository status
        repository = db.query(Repository).filter(Repository.id == uuid.UUID(repository_id)).first()
        
        if not repository:
            raise ValueError(f"Repository not found: {repository_id}")
        
        repository.repo_metadata = {
            **(repository.repo_metadata or {}),
            "indexing_status": "in_progress",
            "last_error": None,
        }
        db.commit()
        
        # Get last indexed time for incremental indexing
        last_indexed = repository.last_indexed if incremental else None
        
        # Initialize store with repository-specific collection
        store = ChromaStore(repo_id=f"{repository.owner_id}_{repository_id}")
        
        # Scan repository for documents
        documents = scanner.scan_repository(
            repo_url, 
            branch,
            since=last_indexed if incremental and last_indexed else None
        )
        
        # Log info about documents
        if last_indexed and incremental:
            logger.info(f"Found {len(documents)} documents modified since {last_indexed}")
        else:
            logger.info(f"Performing full indexing, found {len(documents)} documents")
        
        # Add documents to vector store
        store.add_documents(documents, incremental=incremental)
        
        # Create links between repository and vector store entries
        for doc in documents:
            # Create VectorStoreLink entries
            vector_link = VectorStoreLink(
                id=uuid.uuid4(),
                entity_type="repository_file",
                entity_id=repository.id,
                vector_id=str(doc.id),
                collection_name=store.collection_name,
                metadata={
                    "file_path": doc.metadata.get("file_path", ""),
                    "language": doc.metadata.get("language", ""),
                    "indexed_at": datetime.utcnow().isoformat()
                }
            )
            db.add(vector_link)
        
        # Update repository status
        repository.last_indexed = db.func.now()
        repository.repo_metadata = {
            **(repository.repo_metadata or {}),
            "indexing_status": "completed",
            "last_error": None,
            "document_count": len(documents),
            "incremental": incremental
        }
        db.commit()
        
        logger.info(f"Repository indexed successfully: {repo_url}")
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "repository_id": repository_id,
            "incremental": incremental
        }
        
    except Exception as e:
        logger.error(f"Error indexing repository: {str(e)}")
        
        # Update repository status to indicate failure
        try:
            repository = db.query(Repository).filter(Repository.id == uuid.UUID(repository_id)).first()
            if repository:
                repository.repo_metadata = {
                    **(repository.repo_metadata or {}),
                    "indexing_status": "failed",
                    "last_error": str(e)
                }
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating repository status: {str(db_error)}")
        
        # Re-raise as Celery task failure
        self.retry(exc=e, countdown=60, max_retries=3)
        
    finally:
        # Close database session
        close_db_session(db)

@celery_app.task(bind=True, name="app.tasks.analyze_code")
def analyze_code(
    self,
    code: str,
    language: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze code for insights, patterns, etc.
    
    Args:
        code: Code to analyze
        language: Programming language
        metadata: Additional metadata
        
    Returns:
        Dict with analysis results
    """
    from app.rag.code_explainer import CodeExplainer
    from app.utils.llm import LLMClient
    
    logger.info(f"Starting code analysis for {language} code")
    
    try:
        # Initialize components
        store = ChromaStore()
        llm_client = LLMClient()
        explainer = CodeExplainer(store, llm_client)
        
        # Analyze code
        explanation = explainer.explain_code(code, language)
        
        # Find similar code patterns
        similar_code = explainer.find_similar_code(code, language, top_k=5)
        
        # Suggest improvements
        improvements = explainer.suggest_improvements(code, language)
        
        return {
            "status": "success",
            "explanation": {
                "content": explanation.explanation,
                "key_concepts": explanation.key_concepts,
                "dependencies": explanation.dependencies,
                "complexity": explanation.complexity,
                "best_practices": explanation.best_practices
            },
            "similar_code": similar_code,
            "improvements": improvements
        }
        
    except Exception as e:
        logger.error(f"Error analyzing code: {str(e)}")
        self.retry(exc=e, countdown=30, max_retries=2)

# Make Celery worker load all tasks when starting
celery_app.autodiscover_tasks(["app"]) 