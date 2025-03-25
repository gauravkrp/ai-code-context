"""
GitHub repository module for cloning and accessing code repositories.
"""
import os
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple

from github import Github, GithubException
import git
from tqdm import tqdm

from app.config.settings import config

logger = logging.getLogger(__name__)

class GitHubRepository:
    """Class for interacting with GitHub repositories."""
    
    def __init__(self, access_token: str = None, repository: str = None):
        """
        Initialize the GitHub repository handler.
        
        Args:
            access_token: GitHub access token. If None, uses the token from config.
            repository: Repository name in the format "owner/repo". If None, uses the repo from config.
        """
        self.access_token = access_token or config.github.access_token
        self.repository_name = repository or config.github.repository
        self.github_client = Github(self.access_token)
        self.local_path = None
        self._temp_dir = None
        
    def clone_repository(self) -> str:
        """
        Clone the repository to a temporary directory.
        
        Returns:
            str: Path to the cloned repository.
        """
        try:
            # Create a temporary directory
            self._temp_dir = tempfile.TemporaryDirectory()
            self.local_path = self._temp_dir.name
            
            # Clone the repository
            logger.info(f"Cloning repository {self.repository_name} to {self.local_path}")
            clone_url = f"https://{self.access_token}@github.com/{self.repository_name}.git"
            git.Repo.clone_from(clone_url, self.local_path)
            
            return self.local_path
            
        except GithubException as e:
            logger.error(f"GitHub error: {e}")
            raise
        except git.GitCommandError as e:
            logger.error(f"Git command error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get the content of a file from the repository.
        
        Args:
            file_path: Path to the file in the repository.
            
        Returns:
            str: Content of the file or None if the file doesn't exist.
        """
        if not self.local_path:
            logger.error("Repository has not been cloned. Call clone_repository() first.")
            return None
        
        full_path = Path(self.local_path) / file_path
        if not full_path.exists():
            logger.warning(f"File {file_path} does not exist in the repository.")
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def get_file_list(self, file_extensions: List[str] = None) -> List[str]:
        """
        Get a list of all files in the repository, optionally filtered by extension.
        
        Args:
            file_extensions: List of file extensions to include, e.g. ['.py', '.js'].
                            If None, all files are included.
        
        Returns:
            List[str]: List of file paths relative to the repository root.
        """
        if not self.local_path:
            logger.error("Repository has not been cloned. Call clone_repository() first.")
            return []
        
        all_files = []
        repo_path = Path(self.local_path)
        
        for root, _, files in os.walk(repo_path):
            for file in files:
                # Skip .git directory
                if '.git' in root:
                    continue
                    
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.local_path)
                
                # Filter by extension if specified
                if file_extensions:
                    if any(relative_path.endswith(ext) for ext in file_extensions):
                        all_files.append(relative_path)
                else:
                    all_files.append(relative_path)
        
        return all_files
    
    def iter_files(self, file_extensions: List[str] = None) -> Generator[Tuple[str, str], None, None]:
        """
        Iterate through files in the repository, yielding (file_path, content) tuples.
        
        Args:
            file_extensions: List of file extensions to include, e.g. ['.py', '.js'].
                            If None, all files are included.
        
        Yields:
            Tuple[str, str]: (file_path, file_content) for each file.
        """
        file_list = self.get_file_list(file_extensions)
        
        for file_path in tqdm(file_list, desc="Processing files"):
            content = self.get_file_content(file_path)
            if content is not None:
                yield file_path, content
    
    def cleanup(self):
        """Remove the temporary directory with the cloned repository."""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self.local_path = None
            logger.info("Temporary repository directory cleaned up.") 