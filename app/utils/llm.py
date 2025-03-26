"""
LLM client for code querying.
"""
import logging
from typing import Dict, Any, Optional
import openai
from anthropic import Anthropic

from app.config.settings import config
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/llm.log")

class LLMClient:
    """LLM client for code querying."""
    
    def __init__(self):
        """Initialize the LLM client."""
        # Initialize OpenAI client
        if config.llm.openai_api_key:
            openai.api_key = config.llm.openai_api_key
            self.use_openai = True
        else:
            self.use_openai = False
        
        # Initialize Anthropic client
        if config.llm.anthropic_api_key:
            self.anthropic = Anthropic(api_key=config.llm.anthropic_api_key)
            self.use_anthropic = True
        else:
            self.use_anthropic = False
        
        if not self.use_openai and not self.use_anthropic:
            raise ValueError("No LLM API keys provided")
        
        logger.info(f"Initialized LLM client with OpenAI: {self.use_openai}, Anthropic: {self.use_anthropic}")
    
    def query(
        self,
        query: str,
        code_context: str,
        metadata: Optional[Dict[str, Any]] = None,
        response_format: str = "text"
    ) -> Any:
        """
        Query the LLM with code context.
        
        Args:
            query: User's question
            code_context: Code context
            metadata: Optional metadata about the code
            response_format: Expected response format ("text" or "json")
            
        Returns:
            Any: LLM response (string for text, dict/list for json)
        """
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(query, code_context, metadata)
            
            # Get response from appropriate LLM
            if self.use_openai:
                response = self._query_openai(prompt)
            else:
                response = self._query_anthropic(prompt)
            
            logger.info(f"Generated response for query: {query[:100]}...")
            
            # Parse JSON if needed
            if response_format == "json":
                try:
                    import json
                    # Try to find JSON in the response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        return json.loads(json_str)
                    else:
                        logger.warning(f"Could not extract JSON from response: {response[:100]}...")
                        # Return a basic dict with the response text
                        return {"text": response}
                except Exception as e:
                    logger.error(f"Error parsing JSON response: {e}")
                    # Return a basic dict with the response text
                    return {"text": response}
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return f"Error generating response: {str(e)}"
    
    def _prepare_prompt(
        self,
        query: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare the prompt for the LLM.
        
        Args:
            query: User's question
            context: Code context
            metadata: Optional metadata about the code
            
        Returns:
            str: Formatted prompt
        """
        # Build prompt with metadata if available
        prompt_parts = [
            "You are a helpful coding assistant who excels at explaining code concepts in clear, natural language. Your goal is to help users understand programming concepts, not just provide code.",
            f"\nQuestion: {query}",
            "\nCode Context:",
            context
        ]
        
        if metadata:
            prompt_parts.extend([
                "\nCode Metadata:",
                f"File: {metadata.get('file_path', 'Unknown')}",
                f"Language: {metadata.get('language', 'Unknown')}",
                f"Lines: {metadata.get('start_line', 1)}-{metadata.get('end_line', 1)}"
            ])
        
        prompt_parts.extend([
            "\nPlease provide a clear, conversational explanation in natural language that directly answers the user's question. Focus on concepts and understanding rather than just describing the code. Use simple language that would be accessible to someone learning programming.",
            "If the answer cannot be determined from the provided code, say so."
        ])
        
        return "\n".join(prompt_parts)
    
    def _query_openai(self, prompt: str) -> str:
        """
        Query OpenAI's API.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            str: OpenAI's response
        """
        try:
            system_message = "You are a helpful coding assistant who excels at explaining code concepts in clear, natural language. Your goal is to help users understand programming concepts, not just provide code."
            
            # For newer API versions, some parameters have been renamed
            if "gpt-4" in config.llm.model_name or "gpt-3.5" in config.llm.model_name:
                response = openai.chat.completions.create(
                    model=config.llm.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                )
            # For o3-mini and other models that don't support temperature
            elif "o3-mini" in config.llm.model_name:
                response = openai.chat.completions.create(
                    model=config.llm.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                # Fall back to using the standard parameters
                response = openai.chat.completions.create(
                    model=config.llm.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=config.llm.max_tokens,
                    temperature=config.llm.temperature
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            raise
    
    def _query_anthropic(self, prompt: str) -> str:
        """
        Query Anthropic's API.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            str: Anthropic's response
        """
        try:
            response = self.anthropic.messages.create(
                model=config.llm.model_name,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error querying Anthropic: {e}")
            raise 