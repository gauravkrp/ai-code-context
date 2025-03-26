"""
Text processing utilities for chunking code files and preparing for vectorization.
"""
import re
from typing import List, Dict, Any, Tuple
import logging

from app.config.settings import config

logger = logging.getLogger(__name__)

def split_code_to_chunks(file_path: str, content: str) -> List[Dict[str, Any]]:
    """
    Split a code file into chunks for indexing.
    
    Args:
        file_path: Path to the file within the repository.
        content: Content of the file.
        
    Returns:
        List of dictionaries with structure:
        {
            'id': str,              # Unique identifier for the chunk
            'file_path': str,       # Path to the file 
            'content': str,         # The chunk content
            'start_line': int,      # Starting line number
            'end_line': int,        # Ending line number
            'metadata': Dict        # Additional metadata
        }
    """
    if not content.strip():
        return []
    
    # Split the content into lines
    lines = content.split('\n')
    
    # Calculate the number of chunks based on chunk size and overlap
    total_lines = len(lines)
    chunk_size = config.chunking.chunk_size
    chunk_overlap = config.chunking.chunk_overlap
    
    # Handle small files
    if total_lines <= chunk_size:
        return [create_chunk(file_path, content, 1, total_lines)]
    
    chunks = []
    position = 0
    
    while position < total_lines:
        # Calculate the end position for this chunk
        end_position = min(position + chunk_size, total_lines)
        # Extract the lines for this chunk
        chunk_lines = lines[position:end_position]
        # Join the lines to form the chunk content
        chunk_content = '\n'.join(chunk_lines)
        
        # Create the chunk dictionary
        chunk = create_chunk(
            file_path, 
            chunk_content, 
            position + 1,  # 1-indexed line numbers
            end_position
        )
        chunks.append(chunk)
        
        # Move to the next chunk position, accounting for overlap
        position += chunk_size - chunk_overlap
        
        # Avoid creating chunks with less than 10 lines
        if position >= total_lines - 10:
            break
    
    # Handle any remaining lines
    if position < total_lines:
        remaining_lines = lines[position:]
        remaining_content = '\n'.join(remaining_lines)
        chunk = create_chunk(
            file_path, 
            remaining_content, 
            position + 1, 
            total_lines
        )
        chunks.append(chunk)
    
    return chunks

def create_chunk(file_path: str, content: str, start_line: int, end_line: int) -> Dict[str, Any]:
    """
    Create a chunk dictionary with metadata.
    
    Args:
        file_path: Path to the file within the repository.
        content: Content of the chunk.
        start_line: Starting line number.
        end_line: Ending line number.
        
    Returns:
        Dictionary representing the chunk.
    """
    # Create a unique ID for the chunk
    chunk_id = f"{file_path}:{start_line}-{end_line}"
    
    # Extract file extension for language detection
    extension = file_path.split('.')[-1] if '.' in file_path else ''
    
    # Basic metadata
    metadata = {
        'file_path': file_path,
        'start_line': start_line,
        'end_line': end_line,
        'extension': extension,
        'lines_count': end_line - start_line + 1,
    }
    
    # Create the chunk dictionary
    return {
        'id': chunk_id,
        'file_path': file_path,
        'content': content,
        'start_line': start_line,
        'end_line': end_line,
        'metadata': metadata
    }

def extract_code_elements(content: str) -> Dict[str, List[str]]:
    """
    Extract key code elements (functions, classes, variables) from code content.
    
    Args:
        content: The code content to parse.
        
    Returns:
        Dictionary of extracted elements by type.
    """
    # This is a simplified implementation that could be expanded with more sophisticated parsing
    elements = {
        'functions': [],
        'classes': [],
        'variables': []
    }
    
    # Simple regex patterns to identify code elements (not perfect but useful for metadata)
    # Function pattern (works for Python, JavaScript, etc.)
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)|function\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    # Class pattern
    class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    # Variable declaration (simplified)
    var_pattern = r'(?:let|const|var|my|our)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    
    # Extract functions
    for match in re.finditer(function_pattern, content):
        func_name = match.group(1) or match.group(2)
        if func_name and func_name not in elements['functions']:
            elements['functions'].append(func_name)
    
    # Extract classes
    for match in re.finditer(class_pattern, content):
        class_name = match.group(1)
        if class_name and class_name not in elements['classes']:
            elements['classes'].append(class_name)
    
    # Extract variables
    for match in re.finditer(var_pattern, content):
        var_name = match.group(1)
        if var_name and var_name not in elements['variables']:
            elements['variables'].append(var_name)
    
    return elements

def create_document_metadata(file_path: str, content: str) -> Dict[str, Any]:
    """
    Create comprehensive metadata for a file.
    
    Args:
        file_path: Path to the file within the repository.
        content: Content of the file.
        
    Returns:
        Dictionary of metadata.
    """
    # Extract file extension
    extension = file_path.split('.')[-1] if '.' in file_path else ''
    
    # Get line count
    line_count = len(content.split('\n'))
    
    # Extract code elements
    code_elements = extract_code_elements(content)
    
    # Determine file type/language based on extension
    file_type_map = {
        'py': 'Python',
        'js': 'JavaScript',
        'ts': 'TypeScript',
        'tsx': 'TypeScript React',
        'jsx': 'JavaScript React',
        'java': 'Java',
        'c': 'C',
        'cpp': 'C++',
        'go': 'Go',
        'rb': 'Ruby',
        'php': 'PHP',
        'html': 'HTML',
        'css': 'CSS',
        'md': 'Markdown',
        'json': 'JSON',
        'yml': 'YAML',
        'yaml': 'YAML',
        'sh': 'Shell',
        'bat': 'Batch',
        'ps1': 'PowerShell',
    }
    
    language = file_type_map.get(extension.lower(), 'Unknown')
    
    # Convert lists to strings in code elements
    def convert_to_string(value):
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        return str(value)
    
    # Create metadata dictionary with string values
    metadata = {
        'file_path': file_path,
        'extension': extension,
        'language': language,
        'line_count': line_count,
        'functions': convert_to_string(code_elements['functions']),
        'classes': convert_to_string(code_elements['classes']),
        'variables': convert_to_string(code_elements['variables']),
    }
    
    return metadata 