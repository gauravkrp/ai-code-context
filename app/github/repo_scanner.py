"""
GitHub repository scanner for code indexing.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Union
from pathlib import Path
import os
import tempfile
from github import Github, GithubException
import git
import fnmatch
from datetime import datetime
import uuid
import traceback

from app.config.settings import config
from app.utils.logger import setup_logger
from app.utils.code_chunker import CodeChunk, CodeChunker

logger = setup_logger(__name__, "logs/github_scanner.log")

class GitHubScanner:
    """GitHub repository scanner for code indexing."""
    
    def __init__(self, access_token: Optional[str] = None):
        """
        Initialize the GitHub scanner.
        
        Args:
            access_token: Optional GitHub access token. If not provided, uses token from config.
        """
        self.access_token = access_token or config.github.access_token
        self.github = Github(self.access_token)
        self.chunker = CodeChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
        self.ignore_patterns: Set[str] = set()
        # Load ignore patterns from .codeignore
        self.load_ignore_patterns()
        logger.info("Initialized GitHub Scanner")
    
    def load_ignore_patterns(self, ignore_file: str = ".codeignore") -> None:
        """
        Load ignore patterns from a .codeignore file.
        
        Args:
            ignore_file: Path to the ignore file (default: .codeignore)
        """
        try:
            # Get the absolute path to the .codeignore file in the workspace
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ignore_path = os.path.join(workspace_root, ignore_file)
            
            if os.path.exists(ignore_path):
                with open(ignore_path, 'r') as f:
                    # Read patterns and filter out comments and empty lines
                    patterns = [
                        line.strip() for line in f
                        if line.strip() and not line.startswith('#')
                    ]
                self.ignore_patterns.update(patterns)
                logger.info(f"Loaded {len(patterns)} ignore patterns from {ignore_path}")
            else:
                logger.warning(f"Could not find {ignore_file} at {ignore_path}")
        except Exception as e:
            logger.error(f"Error loading ignore patterns: {e}")
    
    def should_ignore(self, file_path: str) -> bool:
        """
        Check if a file should be ignored based on patterns.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if file should be ignored
        """
        # Convert to relative path for pattern matching
        relative_path = str(file_path)
        
        # Check if any pattern matches the path
        for pattern in self.ignore_patterns:
            # Try exact match
            if fnmatch.fnmatch(relative_path, pattern):
                logger.debug(f"Ignoring {relative_path} (matches pattern: {pattern})")
                return True
            # Try with leading slash
            if fnmatch.fnmatch(f"/{relative_path}", pattern):
                logger.debug(f"Ignoring {relative_path} (matches pattern with leading slash: {pattern})")
                return True
            # Try with leading **/
            if fnmatch.fnmatch(f"**/{relative_path}", pattern):
                logger.debug(f"Ignoring {relative_path} (matches pattern with **/: {pattern})")
                return True
        return False
    
    def scan_repository(
        self,
        repo_url: str,
        branch: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan a GitHub repository and extract code documents.
        
        Args:
            repo_url: GitHub repository URL in the format "owner/repo"
            branch: Optional branch name
            since: Optional datetime to get only files changed since this time
            
        Returns:
            List of Document objects
        """
        try:
            # Extract owner and repo name
            if "/" not in repo_url:
                raise ValueError(f"Invalid repo URL format: {repo_url}. Expected format: owner/repo")
                
            # Handle URLs like 'https://github.com/owner/repo'
            if repo_url.startswith("http"):
                parts = repo_url.split("/")
                if len(parts) < 2:
                    raise ValueError(f"Invalid repo URL: {repo_url}")
                owner = parts[-2]
                repo_name = parts[-1]
            else:
                # Handle format 'owner/repo'
                parts = repo_url.split("/")
                owner = parts[0]
                repo_name = parts[1]
                
            logger.info(f"Scanning GitHub repository: {owner}/{repo_name} (branch: {branch if branch else 'main'})")
            
            # Access the repository
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            
            # Use specified branch or default branch
            if branch:
                repo_branch = branch
            else:
                repo_branch = repo.default_branch
                logger.info(f"Using default branch: {repo_branch}")
            
            # Get files recursively
            all_files = []
            
            # Handle incremental indexing if timestamp provided
            if since:
                logger.info(f"Getting commits since {since}")
                # Get commits since the 'since' timestamp
                commits = repo.get_commits(sha=repo_branch, since=since)
                
                # Get changed files from each commit
                changed_files = set()
                for commit in commits:
                    for file in commit.files:
                        if file.status in ["added", "modified"]:
                            changed_files.add(file.filename)
                
                if not changed_files:
                    logger.info("No files changed since last indexing")
                    return []
                    
                logger.info(f"Found {len(changed_files)} changed files since {since}")
                
                # Get each changed file
                for file_path in changed_files:
                    try:
                        # Skip files in excluded directories or with unsupported extensions
                        if self.should_ignore(file_path):
                            continue
                            
                        file_content = repo.get_contents(file_path, ref=repo_branch)
                        
                        # Skip if file is too large
                        if hasattr(file_content, "size") and file_content.size > config.github.max_file_size:
                            logger.warning(f"Skipping large file: {file_path} ({file_content.size} bytes)")
                            continue
                            
                        # Handle binary files
                        if hasattr(file_content, "content") and not self._is_text_content(file_content.content):
                            logger.info(f"Skipping binary file: {file_path}")
                            continue
                            
                        # Add to list of files to process
                        all_files.append(file_content)
                    except Exception as e:
                        logger.error(f"Error getting file {file_path}: {str(e)}")
            else:
                # Full repository scan
                logger.info("Performing full repository scan")
                self._get_repository_contents(repo, "", repo_branch, all_files)
            
            logger.info(f"Found {len(all_files)} files to process")
            
            # Process files into documents
            documents = []
            for file in all_files:
                try:
                    file_path = file.path
                    file_content = file.decoded_content.decode('utf-8')
                    file_sha = file.sha
                    
                    # Get file metadata
                    extension = os.path.splitext(file_path)[1].lstrip('.')
                    language = self._detect_language(extension)
                    
                    # Create document with metadata
                    document = {
                        'file_path': file_path,
                        'content': file_content,
                        'language': language,
                        'extension': extension,
                        'file_sha': file_sha,
                        'size': len(file_content),
                        'processed_at': datetime.utcnow().isoformat(),
                        'source': f"{owner}/{repo_name}",
                        'branch': repo_branch
                    }
                    
                    # Only include files that pass all filters
                    if self._should_include_document(document):
                        documents.append(document)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file.path}: {str(e)}")
            
            logger.info(f"Created {len(documents)} documents from repository")
            return documents
            
        except Exception as e:
            logger.error(f"Error scanning repository: {str(e)}")
            raise
    
    def _scan_directory(
        self,
        directory: str,
        file_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Scan a directory for code files.
        
        Args:
            directory: Directory to scan
            file_types: List of file extensions to include
            
        Returns:
            List[Dict[str, Any]]: List of code documents
        """
        documents = []
        dir_path = Path(directory)
        
        try:
            # Normalize file types to include dot prefix
            normalized_types = [f".{ext}" if not ext.startswith(".") else ext for ext in file_types]
            logger.info(f"Scanning for file types: {normalized_types}")
            
            # Walk through directory
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    
                    # Check if file should be ignored
                    if self.should_ignore(str(file_path)):
                        logger.debug(f"Ignoring file: {file_path}")
                        continue
                    
                    # Log each file being checked
                    logger.debug(f"Checking file: {file_path}")
                    
                    # Check file extension
                    if file_path.suffix.lower() in normalized_types:
                        try:
                            # Read file content
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Create document
                            doc = {
                                'file_path': str(file_path.relative_to(dir_path)),
                                'content': content,
                                'language': self._detect_language(file_path.suffix),
                                'size': os.path.getsize(file_path)
                            }
                            
                            documents.append(doc)
                            logger.info(f"Found document: {doc['file_path']} ({doc['language']})")
                            
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                            continue
                    else:
                        logger.debug(f"Skipping file {file_path} - extension not in supported types")
            
            logger.info(f"Scan complete. Found {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            return []
    
    def _detect_language(self, extension: str) -> str:
        """
        Detect programming language from file extension.
        
        Args:
            extension: File extension
            
        Returns:
            str: Programming language name
        """
        language_map = {
            # Programming Languages
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'matlab',
            '.sh': 'shell',
            
            # Web Technologies
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            
            # Data Formats
            '.json': 'json',
            '.json5': 'json5',
            '.jsonc': 'jsonc',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.xml': 'xml',
            
            # Documentation
            '.md': 'markdown',
            '.mdown': 'markdown',
            '.markdown': 'markdown',
            '.rst': 'rst',
            '.txt': 'text',
            
            # Configuration
            '.env': 'env',
            '.config': 'config',
            '.conf': 'config',
            '.ini': 'ini',
            '.properties': 'properties',
            
            # Database
            '.sql': 'sql',
            '.prisma': 'prisma',
            '.graphql': 'graphql',
            
            # Docker & Container
            'dockerfile': 'dockerfile',
            'docker-compose.yml': 'yaml',
            
            # CI/CD
            '.github/workflows/*.yml': 'yaml',
            'gitlab-ci.yml': 'yaml',
            '.gitlab-ci.yml': 'yaml',
            
            # Testing
            '.test.js': 'javascript',
            '.test.ts': 'typescript',
            '.spec.js': 'javascript',
            '.spec.ts': 'typescript',
            
            # Package Management
            'package.json': 'json',
            
            # Misc
            '.svg': 'svg',
            '.proto': 'protobuf'
        }
        
        # Handle special cases for exact filenames
        if extension in language_map:
            return language_map[extension]
            
        # Handle file extensions
        return language_map.get(extension.lower(), 'unknown') 

    def _should_include_document(self, document: Dict[str, Any]) -> bool:
        """
        Check if a document should be included in the index.
        
        Args:
            document: Document to check
            
        Returns:
            bool: True if document should be included
        """
        # Skip if file is too large
        if document.get('size', 0) > config.github.max_file_size:
            logger.warning(f"Skipping large file: {document['file_path']} ({document['size']} bytes)")
            return False
            
        # Skip if file should be ignored
        if self.should_ignore(document['file_path']):
            logger.debug(f"Ignoring file: {document['file_path']}")
            return False
            
        # Skip if language is unknown
        if document.get('language') == 'unknown':
            logger.debug(f"Skipping file with unknown language: {document['file_path']}")
            return False
            
        return True

    def _is_text_content(self, content: Union[bytes, str]) -> bool:
        """
        Check if the content is text-based rather than binary.
        
        Args:
            content: Raw file content in bytes or string
            
        Returns:
            bool: True if content appears to be text, False if binary
        """
        # If content is already a string, it's text
        if isinstance(content, str):
            return True
            
        # If content is bytes, try to decode it
        try:
            content.decode('utf-8')
            return True
        except UnicodeDecodeError:
            # If we can't decode as UTF-8, it's likely binary
            return False

    def _get_repository_contents(
        self,
        repo,
        path: str,
        branch: str,
        all_files: List[Any]
    ) -> None:
        """
        Recursively get repository contents.
        
        Args:
            repo: GitHub repository object
            path: Current path in repository
            branch: Branch name
            all_files: List to store file contents
        """
        try:
            logger.info(f"Getting contents for path: {path}")
            contents = repo.get_contents(path, ref=branch)
            logger.info(f"Found {len(contents)} items in path: {path}")
            
            for content in contents:
                # Check if directory or file should be ignored
                if self.should_ignore(content.path):
                    logger.debug(f"Ignoring path: {content.path}")
                    continue
                    
                if content.type == "dir":
                    logger.info(f"Processing directory: {content.path}")
                    # Recursively process directories
                    self._get_repository_contents(repo, content.path, branch, all_files)
                else:
                    logger.debug(f"Processing file: {content.path}")
                    # Skip if file is too large
                    if hasattr(content, "size") and content.size > config.github.max_file_size:
                        logger.warning(f"Skipping large file: {content.path} ({content.size} bytes)")
                        continue
                        
                    # Handle binary files
                    if hasattr(content, "content") and not self._is_text_content(content.content):
                        logger.info(f"Skipping binary file: {content.path}")
                        continue
                        
                    # Add to list of files to process
                    all_files.append(content)
                    logger.debug(f"Added file to process: {content.path}")
                    
        except Exception as e:
            logger.error(f"Error getting contents for path {path}: {str(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}") 