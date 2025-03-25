"""
Slack bot module for interacting with the code search engine.
"""
import logging
import re
import os
from typing import Dict, Any, List

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.config.settings import config
from app.utils.llm import LLMHandler
from app.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

class SlackBot:
    """Slack bot for interacting with the code search engine."""
    
    def __init__(
        self, 
        bot_token: str = None, 
        signing_secret: str = None,
        llm_handler: LLMHandler = None,
        vector_store: ChromaVectorStore = None
    ):
        """
        Initialize the Slack bot.
        
        Args:
            bot_token: Slack bot token. If None, uses the token from config.
            signing_secret: Slack signing secret. If None, uses the secret from config.
            llm_handler: LLM handler instance.
            vector_store: Vector store instance.
        """
        self.bot_token = bot_token or config.slack.bot_token
        self.signing_secret = signing_secret or config.slack.signing_secret
        self.channel_id = config.slack.channel_id
        
        # Initialize LLM and Vector Store if not provided
        self.llm_handler = llm_handler or LLMHandler()
        self.vector_store = vector_store or ChromaVectorStore()
        
        # Initialize Slack app
        self.app = App(token=self.bot_token, signing_secret=self.signing_secret)
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("Slack bot initialized")
    
    def _register_handlers(self):
        """Register event handlers for Slack events."""
        # Handle messages in the configured channel
        @self.app.message("")
        def handle_message(message, say):
            """Handle incoming messages."""
            # Ignore bot messages
            if message.get("bot_id"):
                return
            
            # Only respond in the configured channel if specified
            if self.channel_id and message.get("channel") != self.channel_id:
                return
            
            # Process the message
            self._process_message(message, say)
    
    def _process_message(self, message: Dict[str, Any], say):
        """
        Process an incoming message and respond.
        
        Args:
            message: Slack message event.
            say: Function to send a response.
        """
        # Extract message text
        text = message.get("text", "").strip()
        
        if not text:
            return
        
        # Log the incoming message
        logger.info(f"Received message: {text}")
        
        try:
            # Send a thinking message
            say("Searching codebase for an answer...")
            
            # Search the vector store for relevant context
            search_results = self.vector_store.search(text, n_results=5)
            
            if not search_results:
                say("I couldn't find any relevant code to answer your question. The codebase might not contain information related to your query.")
                return
            
            # Query the LLM with the message and search results as context
            response = self.llm_handler.query(text, search_results)
            
            # Format the response for Slack
            formatted_response = self._format_response_for_slack(text, response, search_results)
            
            # Send the response
            say(formatted_response)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            say(f"Sorry, I encountered an error while processing your question: {str(e)}")
    
    def _format_response_for_slack(
        self, 
        question: str, 
        response: str, 
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format the response for Slack, including code blocks and file references.
        
        Args:
            question: The original question.
            response: The response from the LLM.
            search_results: The search results from the vector store.
            
        Returns:
            Formatted response for Slack.
        """
        # Format the response with markdown
        formatted_response = f"*Question:*\n>{question}\n\n*Answer:*\n{response}\n"
        
        # Add source references
        if search_results:
            formatted_response += "\n*Sources:*\n"
            for i, result in enumerate(search_results[:3]):  # Show top 3 sources
                metadata = result.get('metadata', {})
                file_path = metadata.get('file_path', 'Unknown file')
                start_line = metadata.get('start_line', 1)
                end_line = metadata.get('end_line', start_line)
                
                formatted_response += f"• `{file_path}` (lines {start_line}-{end_line})\n"
        
        return formatted_response
    
    def start(self, use_socket_mode: bool = False):
        """
        Start the Slack bot.
        
        Args:
            use_socket_mode: Whether to use Socket Mode or HTTP.
        """
        try:
            if use_socket_mode:
                if not os.environ.get("SLACK_APP_TOKEN"):
                    logger.error("SLACK_APP_TOKEN not found in environment. Socket Mode requires an App-level token.")
                    return
                
                handler = SocketModeHandler(self.app, os.environ["SLACK_APP_TOKEN"])
                logger.info("Starting Slack bot in Socket Mode...")
                handler.start()
            else:
                logger.info("Starting Slack bot with HTTP server...")
                self.app.start(port=int(os.environ.get("PORT", 3000)))
        except Exception as e:
            logger.error(f"Error starting Slack bot: {e}")
            raise 