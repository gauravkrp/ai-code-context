"""
Specialized code explanation features for better code understanding.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.utils.logger import setup_logger
from app.utils.llm import LLMClient
from app.vector_store.chroma_store import ChromaStore
from app.config.settings import config

logger = setup_logger(__name__, "logs/code_explainer.log")

@dataclass
class CodeExplanation:
    """Represents a detailed code explanation."""
    code_snippet: str
    explanation: str
    key_concepts: List[str]
    dependencies: List[str]
    complexity: str
    best_practices: List[str]

class CodeExplainer:
    """Specialized code explanation system."""
    
    def __init__(self, vector_store: ChromaStore, llm_client: LLMClient):
        """
        Initialize the code explainer.
        
        Args:
            vector_store: Vector store for code retrieval
            llm_client: LLM client for code analysis
        """
        self.vector_store = vector_store
        self.llm_client = llm_client
        logger.info("Initialized CodeExplainer")
    
    def explain_code(self, code: str, language: str) -> CodeExplanation:
        """
        Generate a detailed explanation of the code.
        
        Args:
            code: Code to explain
            language: Programming language
            
        Returns:
            CodeExplanation: Detailed code explanation
        """
        # Create prompt for code explanation
        prompt = f"""Analyze the following {language} code and provide a detailed explanation.

Code:
{code}

Provide a comprehensive explanation that includes:
1. Overall purpose and functionality
2. Key concepts and patterns used
3. Dependencies and relationships
4. Complexity analysis
5. Best practices and potential improvements

Format the response as JSON with the following structure:
{{
    "explanation": "Detailed explanation",
    "key_concepts": ["List of key concepts"],
    "dependencies": ["List of dependencies"],
    "complexity": "Complexity analysis",
    "best_practices": ["List of best practices"]
}}"""
        
        # Get explanation from LLM as JSON format
        response = self.llm_client.query(
            query=prompt,
            code_context=code,
            metadata={"task": "code_explanation", "language": language},
            response_format="json"
        )
        
        # Create explanation with the parsed JSON response
        try:
            explanation = CodeExplanation(
                code_snippet=code,
                explanation=response.get("explanation", "No explanation provided"),
                key_concepts=response.get("key_concepts", []),
                dependencies=response.get("dependencies", []),
                complexity=response.get("complexity", "Medium"),
                best_practices=response.get("best_practices", [])
            )
        except (AttributeError, TypeError):
            # Fallback if response is not a dictionary
            explanation = CodeExplanation(
                code_snippet=code,
                explanation=str(response) if response else "No explanation provided",
                key_concepts=["Code analysis"],
                dependencies=["Language runtime"],
                complexity="Medium",
                best_practices=["Follow language best practices"]
            )
        
        logger.info("Generated code explanation")
        return explanation
    
    def explain_function(self, function_code: str, language: str) -> Dict[str, Any]:
        """
        Generate a detailed explanation of a function.
        
        Args:
            function_code: Function code to explain
            language: Programming language
            
        Returns:
            Dict[str, Any]: Function explanation
        """
        # Create prompt for function explanation
        prompt = f"""Analyze the following {language} function and provide a detailed explanation.

Function:
{function_code}

Provide a comprehensive explanation that includes:
1. Function purpose and behavior
2. Parameters and return values
3. Side effects and state changes
4. Error handling
5. Performance considerations
6. Usage examples

Format the response as JSON with the following structure:
{{
    "purpose": "Function purpose",
    "parameters": ["Parameter descriptions"],
    "returns": "Return value description",
    "side_effects": ["List of side effects"],
    "error_handling": "Error handling approach",
    "performance": "Performance considerations",
    "examples": ["Usage examples"]
}}"""
        
        # Get explanation from LLM in JSON format
        response = self.llm_client.query(
            query=prompt,
            code_context=function_code,
            metadata={"task": "function_explanation", "language": language},
            response_format="json"
        )
        
        logger.info("Generated function explanation")
        
        # If response is not a dictionary, create a basic one
        if not isinstance(response, dict):
            response = {
                "purpose": str(response),
                "parameters": [],
                "returns": "",
                "side_effects": [],
                "error_handling": "",
                "performance": "",
                "examples": []
            }
            
        return response
    
    def suggest_improvements(self, code: str, language: str) -> List[Dict[str, Any]]:
        """
        Suggest improvements for the code.
        
        Args:
            code: Code to analyze
            language: Programming language
            
        Returns:
            List[Dict[str, Any]]: List of suggested improvements
        """
        # Create prompt for improvement suggestions
        prompt = f"""Analyze the following {language} code and suggest improvements.

Code:
{code}

Suggest improvements that include:
1. Code organization and structure
2. Performance optimizations
3. Error handling
4. Documentation
5. Testing
6. Security considerations

Format the response as JSON with the following structure:
{{
    "improvements": [
        {{
            "type": "Type of improvement",
            "description": "Detailed description",
            "priority": "High/Medium/Low",
            "example": "Example of improved code"
        }}
    ]
}}"""
        
        # Get suggestions from LLM in JSON format
        response = self.llm_client.query(
            query=prompt,
            code_context=code,
            metadata={"task": "improvement_suggestions", "language": language},
            response_format="json"
        )
        
        logger.info("Generated improvement suggestions")
        
        # Return improvements if available, otherwise create a default improvement
        if isinstance(response, dict) and "improvements" in response and isinstance(response["improvements"], list):
            return response["improvements"]
        else:
            return [{
                "type": "General improvements",
                "description": str(response),
                "priority": "Medium",
                "example": code
            }]
    
    def find_similar_code(self, code: str, language: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar code patterns in the codebase.
        
        Args:
            code: Code to find similar patterns for
            language: Programming language
            top_k: Number of similar patterns to return
            
        Returns:
            List[Dict[str, Any]]: List of similar code patterns
        """
        # Search for similar code patterns
        results = self.vector_store.search(
            query=code,
            n_results=top_k,
            filter_criteria={"language": language}
        )
        
        # Process and format results
        similar_patterns = []
        for result in results:
            pattern = {
                "code": result["content"],
                "file_path": result["metadata"]["file_path"],
                "similarity": 1 - result["distance"],  # Convert distance to similarity
                "context": result["metadata"]
            }
            similar_patterns.append(pattern)
        
        logger.info(f"Found {len(similar_patterns)} similar code patterns")
        return similar_patterns 