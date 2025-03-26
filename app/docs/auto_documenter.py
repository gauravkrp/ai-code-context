"""
Auto-documentation system for generating comprehensive code documentation.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from app.utils.logger import setup_logger
from app.utils.llm import LLMClient
from app.vector_store.chroma_store import ChromaStore
from app.config.settings import config

logger = setup_logger(__name__, "logs/auto_documenter.log")

@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    subsections: List['DocumentationSection'] = None

class AutoDocumenter:
    """System for generating comprehensive code documentation."""
    
    def __init__(self, vector_store: ChromaStore, llm_client: LLMClient):
        """
        Initialize the auto-documenter.
        
        Args:
            vector_store: Vector store for code retrieval
            llm_client: LLM client for documentation generation
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        logger.info("Initialized AutoDocumenter")
    
    def generate_module_docs(self, module_path: str) -> DocumentationSection:
        """
        Generate documentation for a module.
        
        Args:
            module_path: Path to the module
            
        Returns:
            DocumentationSection: Module documentation
        """
        # Get module content
        with open(module_path, 'r') as f:
            content = f.read()
        
        # Create prompt for module documentation
        prompt = f"""Generate comprehensive documentation for the following module.

Module Path: {module_path}
Content:
{content}

Generate documentation that includes:
1. Module overview and purpose
2. Key components and their relationships
3. Usage examples
4. Dependencies and requirements
5. Configuration options
6. Common patterns and best practices

Format the response as JSON with the following structure:
{{
    "title": "Module name",
    "content": "Detailed documentation",
    "subsections": [
        {{
            "title": "Section title",
            "content": "Section content",
            "subsections": []
        }}
    ]
}}"""
        
        # Get documentation from LLM
        response = self.llm_client.query(
            query=prompt,
            code_context=content,
            metadata={"task": "module_documentation", "module_path": module_path}
        )
        
        # Create documentation section
        docs = DocumentationSection(
            title=response["title"],
            content=response["content"],
            subsections=[
                DocumentationSection(**section)
                for section in response["subsections"]
            ]
        )
        
        logger.info(f"Generated documentation for module: {module_path}")
        return docs
    
    def generate_api_docs(self, module_path: str) -> DocumentationSection:
        """
        Generate API documentation for a module.
        
        Args:
            module_path: Path to the module
            
        Returns:
            DocumentationSection: API documentation
        """
        # Get module content
        with open(module_path, 'r') as f:
            content = f.read()
        
        # Create prompt for API documentation
        prompt = f"""Generate comprehensive API documentation for the following module.

Module Path: {module_path}
Content:
{content}

Generate API documentation that includes:
1. Public interface overview
2. Class and function documentation
3. Parameter descriptions
4. Return value descriptions
5. Usage examples
6. Error handling
7. Performance considerations

Format the response as JSON with the following structure:
{{
    "title": "API Documentation",
    "content": "Overview of the API",
    "subsections": [
        {{
            "title": "Class/Function name",
            "content": "Detailed documentation",
            "subsections": []
        }}
    ]
}}"""
        
        # Get documentation from LLM
        response = self.llm_client.query(
            query=prompt,
            code_context=content,
            metadata={"task": "api_documentation", "module_path": module_path}
        )
        
        # Create documentation section
        docs = DocumentationSection(
            title=response["title"],
            content=response["content"],
            subsections=[
                DocumentationSection(**section)
                for section in response["subsections"]
            ]
        )
        
        logger.info(f"Generated API documentation for module: {module_path}")
        return docs
    
    def generate_readme(self, project_root: str) -> str:
        """
        Generate a comprehensive README for the project.
        
        Args:
            project_root: Path to the project root
            
        Returns:
            str: Generated README content
        """
        # Get project structure
        project_files = []
        for path in Path(project_root).rglob('*.py'):
            if not any(part.startswith('.') for part in path.parts):
                project_files.append(str(path))
        
        # Create prompt for README generation
        prompt = f"""Generate a comprehensive README for the following project.

Project Files:
{project_files}

Generate a README that includes:
1. Project overview and purpose
2. Installation instructions
3. Usage examples
4. Configuration options
5. Project structure
6. Contributing guidelines
7. License information
8. Dependencies and requirements

Format the response as a markdown document."""
        
        # Get README content from LLM
        readme = self.llm_client.query(
            query=prompt,
            code_context=None,
            metadata={"task": "readme_generation", "project_root": project_root}
        )
        
        logger.info("Generated project README")
        return readme
    
    def generate_docs_site(self, project_root: str, output_dir: str):
        """
        Generate a complete documentation site.
        
        Args:
            project_root: Path to the project root
            output_dir: Directory to output documentation
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate documentation for each module
        for path in Path(project_root).rglob('*.py'):
            if not any(part.startswith('.') for part in path.parts):
                # Generate module documentation
                module_docs = self.generate_module_docs(str(path))
                
                # Generate API documentation
                api_docs = self.generate_api_docs(str(path))
                
                # Create documentation file
                doc_path = output_path / f"{path.stem}.md"
                with open(doc_path, 'w') as f:
                    f.write(f"# {module_docs.title}\n\n")
                    f.write(module_docs.content)
                    f.write("\n\n# API Reference\n\n")
                    f.write(api_docs.content)
        
        # Generate README
        readme = self.generate_readme(project_root)
        with open(output_path / "README.md", 'w') as f:
            f.write(readme)
        
        logger.info(f"Generated documentation site in: {output_dir}") 