"""
Enhanced code chunking module with better code structure awareness.
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from app.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/code_chunker.log")

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # function, class, module, etc.
    dependencies: List[str] = None
    docstring: Optional[str] = None

class CodeChunker:
    """Enhanced code chunking with structure awareness."""
    
    # Language-specific patterns
    PATTERNS = {
        'python': {
            'function': r'def\s+(\w+)\s*\((.*?)\):',
            'class': r'class\s+(\w+)\s*(?:\((.*?)\))?:',
            'docstring': r'"""(.*?)"""|\'\'\'(.*?)\'\'\'',
        },
        'javascript': {
            'function': r'function\s+(\w+)\s*\((.*?)\)|const\s+(\w+)\s*=\s*\((.*?)\)\s*=>',
            'class': r'class\s+(\w+)\s*(?:extends\s+(\w+))?',
            'docstring': r'/\*\*(.*?)\*/|//\s*(.*?)$',
        },
        'typescript': {
            'function': r'function\s+(\w+)\s*\((.*?)\)|const\s+(\w+)\s*=\s*\((.*?)\)\s*=>',
            'class': r'class\s+(\w+)\s*(?:extends\s+(\w+))?',
            'docstring': r'/\*\*(.*?)\*/|//\s*(.*?)$',
        }
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the code chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized CodeChunker with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def detect_language(self, file_path: str) -> str:
        """Detect the programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
        }
        return language_map.get(ext, 'unknown')
    
    def extract_docstring(self, content: str, language: str) -> Optional[str]:
        """Extract docstring from code content."""
        if language not in self.PATTERNS:
            return None
            
        pattern = self.PATTERNS[language]['docstring']
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1) or match.group(2)
        return None
    
    def extract_dependencies(self, content: str, language: str) -> List[str]:
        """Extract dependencies/imports from code content."""
        dependencies = []
        
        if language == 'python':
            # Match import statements
            import_pattern = r'^(?:from\s+(\w+(?:\.\w+)*)\s+import|import\s+(\w+(?:\.\w+)*))'
            for line in content.split('\n'):
                match = re.match(import_pattern, line.strip())
                if match:
                    dep = match.group(1) or match.group(2)
                    if dep:
                        dependencies.append(dep)
        
        elif language in ['javascript', 'typescript']:
            # Match require/import statements
            import_pattern = r'(?:require\([\'"]([^\'"]+)[\'"]\)|import\s+.*?from\s+[\'"]([^\'"]+)[\'"])'
            matches = re.finditer(import_pattern, content)
            for match in matches:
                dep = match.group(1) or match.group(2)
                if dep:
                    dependencies.append(dep)
        
        return dependencies
    
    def chunk_code(self, content: str, file_path: str) -> List[CodeChunk]:
        """
        Chunk code content with structure awareness.
        
        Args:
            content: The code content to chunk
            file_path: Path to the source file
            
        Returns:
            List[CodeChunk]: List of code chunks with metadata
        """
        language = self.detect_language(file_path)
        logger.info(f"Processing file: {file_path} (language: {language})")
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_line = 0
        
        while current_line < len(lines):
            line = lines[current_line]
            
            # Check if we've reached a chunk boundary
            if len('\n'.join(current_chunk)) >= self.chunk_size:
                # Create a chunk with the current content
                chunk_content = '\n'.join(current_chunk)
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_line - len(current_chunk),
                    end_line=current_line,
                    language=language,
                    chunk_type='module',
                    dependencies=self.extract_dependencies(chunk_content, language),
                    docstring=self.extract_docstring(chunk_content, language)
                )
                chunks.append(chunk)
                logger.debug(f"Created chunk: {chunk.chunk_type} at lines {chunk.start_line}-{chunk.end_line}")
                
                # Keep some overlap for context
                overlap_lines = []
                for i in range(min(self.chunk_overlap, len(current_chunk))):
                    overlap_lines.append(current_chunk[-(i+1)])
                current_chunk = list(reversed(overlap_lines))
            
            current_chunk.append(line)
            current_line += 1
        
        # Add the last chunk if there's any content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk = CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=current_line - len(current_chunk),
                end_line=current_line,
                language=language,
                chunk_type='module',
                dependencies=self.extract_dependencies(chunk_content, language),
                docstring=self.extract_docstring(chunk_content, language)
            )
            chunks.append(chunk)
            logger.debug(f"Created final chunk: {chunk.chunk_type} at lines {chunk.start_line}-{chunk.end_line}")
        
        logger.info(f"Created {len(chunks)} chunks for {file_path}")
        return chunks 