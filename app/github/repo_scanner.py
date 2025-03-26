"""
GitHub repository scanner for code indexing.
"""
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import os
import tempfile
from github import Github, GithubException
import git
import fnmatch

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
        logger.info("Initialized GitHub Scanner")
    
    def load_ignore_patterns(self, ignore_file: str = ".codeignore") -> None:
        """
        Load ignore patterns from a .codeignore file.
        
        Args:
            ignore_file: Path to the ignore file (default: .codeignore)
        """
        try:
            if os.path.exists(ignore_file):
                with open(ignore_file, 'r') as f:
                    # Read patterns and filter out comments and empty lines
                    patterns = [
                        line.strip() for line in f
                        if line.strip() and not line.startswith('#')
                    ]
                self.ignore_patterns.update(patterns)
                logger.info(f"Loaded {len(patterns)} ignore patterns from {ignore_file}")
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
        # Check both exact filename and path patterns
        return any(
            fnmatch.fnmatch(relative_path, pattern) or 
            fnmatch.fnmatch(os.path.basename(relative_path), pattern)
            for pattern in self.ignore_patterns
        )
    
    def scan_repository(
        self, 
        repo_name: str,
        branch: str = "main",
        file_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan a GitHub repository and extract code documents.
        
        Args:
            repo_name: Repository name in format 'owner/repo'
            branch: Branch to scan (default: main)
            file_types: List of file extensions to include (default: all supported)
            
        Returns:
            List[Dict[str, Any]]: List of code documents with metadata
        """
        try:
            # Get repository
            repo = self.github.get_repo(repo_name)
            logger.info(f"Scanning repository: {repo_name}")
            
            # Create temporary directory for cloning
            with tempfile.TemporaryDirectory() as temp_dir:
                # Clone repository using git.Repo
                repo_path = os.path.join(temp_dir, repo.name)
                clone_url = f"https://{self.access_token}@github.com/{repo_name}.git"
                git.Repo.clone_from(clone_url, repo_path, branch=branch)
                logger.info(f"Cloned repository to {repo_path}")
                
                # Load ignore patterns from both local and repository .codeignore
                local_ignore = os.path.join(os.getcwd(), ".codeignore")
                repo_ignore = os.path.join(repo_path, ".codeignore")
                
                if os.path.exists(local_ignore):
                    self.load_ignore_patterns(local_ignore)
                if os.path.exists(repo_ignore):
                    self.load_ignore_patterns(repo_ignore)
                
                # Scan files
                documents = self._scan_directory(
                    repo_path,
                    file_types or config.github.supported_file_types
                )
                
                logger.info(f"Found {len(documents)} documents in repository")
                return documents
                
        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error scanning repository: {e}")
            return []
    
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