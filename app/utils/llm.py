"""
LLM interaction module to handle different LLM providers.
"""
import logging
from typing import Dict, List, Optional, Any
import json
import httpx

from anthropic import Anthropic
from openai import OpenAI

from app.config.settings import config

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handler for interacting with different LLM providers."""
    
    def __init__(self):
        """Initialize the LLM handler."""
        self.initialized = False
        self.anthropic_client = None
        self.openai_client = None
        
        # Initialize the appropriate LLM client
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize the LLM clients based on configuration."""
        # Initialize Anthropic client if enabled
        if config.llm.use_claude and config.llm.anthropic_api_key:
            try:
                self.anthropic_client = Anthropic(api_key=config.llm.anthropic_api_key)
                logger.info("Anthropic Claude client initialized")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {e}")
        
        # Initialize OpenAI client if enabled
        if config.llm.use_openai and config.llm.openai_api_key:
            try:
                # Create a direct httpx client with no proxy configuration
                http_client = httpx.Client()
                
                # Initialize OpenAI with the custom HTTP client
                self.openai_client = OpenAI(
                    api_key=config.llm.openai_api_key,
                    http_client=http_client
                )
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
        
        # Check if at least one client is initialized
        if self.anthropic_client or self.openai_client:
            self.initialized = True
        else:
            logger.error("No LLM clients initialized. Please check your configuration.")
    
    def get_available_llm(self) -> str:
        """
        Get the name of the available LLM.
        
        Returns:
            str: Name of the available LLM client.
        """
        if config.llm.use_claude and self.anthropic_client:
            return "claude"
        elif config.llm.use_openai and self.openai_client:
            return "openai"
        else:
            return "none"
    
    def query(
        self, 
        prompt: str, 
        context: List[Dict[str, Any]] = None, 
        max_tokens: int = None
    ) -> str:
        """
        Query the LLM with a prompt and context.
        
        Args:
            prompt: User prompt/question to ask.
            context: List of context documents from vector search.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            str: Generated response from the LLM.
        """
        if not self.initialized:
            logger.error("LLM handler not initialized. Cannot process query.")
            return "Sorry, I cannot process your query at this time due to missing LLM configuration."
        
        # Construct the prompt with context
        formatted_prompt = self._format_prompt_with_context(prompt, context)
        
        # Use configured max tokens or default
        max_tokens_to_use = max_tokens or config.llm.max_tokens
        
        # Query the appropriate LLM
        if config.llm.use_claude and self.anthropic_client:
            return self._query_claude(formatted_prompt, max_tokens_to_use)
        elif config.llm.use_openai and self.openai_client:
            return self._query_openai(formatted_prompt, max_tokens_to_use)
        else:
            logger.error("No LLM client available for query.")
            return "Sorry, I cannot process your query at this time due to missing LLM configuration."
    
    def _format_prompt_with_context(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Format the prompt with context information.
        
        Args:
            prompt: User prompt/question to ask.
            context: List of context documents from vector search.
            
        Returns:
            str: Formatted prompt with context.
        """
        if not context:
            return f"""
            You are a helpful assistant that answers questions about code repositories.
            
            Question: {prompt}
            
            Answer the question as best you can. If you don't know the answer, just say so.
            """
        
        # Format context snippets
        context_text = ""
        for i, doc in enumerate(context):
            metadata = doc.get('metadata', {})
            file_path = metadata.get('file_path', 'Unknown file')
            start_line = metadata.get('start_line', 1)
            end_line = metadata.get('end_line', start_line)
            
            context_text += f"\n\nCONTEXT SNIPPET {i+1} (From {file_path}, lines {start_line}-{end_line}):\n```\n{doc['content']}\n```"
        
        return f"""
        You are a helpful assistant that answers questions about code repositories.
        
        I'll provide you with code snippets that might be relevant to the question.
        
        {context_text}
        
        Question: {prompt}
        
        When answering:
        1. Cite specific files and line numbers from the provided context.
        2. Be specific about where in the code the relevant information is found.
        3. If the context doesn't contain enough information to answer fully, acknowledge that.
        4. If a question is about a feature, bug, or implementation detail, explain with references to the specific code.
        
        Answer:
        """
    
    def _query_claude(self, prompt: str, max_tokens: int) -> str:
        """Query the Anthropic Claude model with a prompt."""
        try:
            response = self.anthropic_client.messages.create(
                model=config.llm.model_name,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error querying Claude: {e}")
            return f"Sorry, I encountered an error when processing your query: {str(e)}"
    
    def _query_openai(self, prompt: str, max_tokens: int) -> str:
        """Query the OpenAI model with a prompt."""
        try:
            response = self.openai_client.chat.completions.create(
                model=config.llm.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about code repositories."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            return f"Sorry, I encountered an error when processing your query: {str(e)}" 