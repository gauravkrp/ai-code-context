"""
Configuration module for the application.
Loads environment variables and provides configuration settings.
"""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class GitHubConfig(BaseModel):
    """GitHub configuration settings."""
    access_token: str = Field(default=os.getenv("GITHUB_ACCESS_TOKEN", ""))
    repository: str = Field(default=os.getenv("GITHUB_REPOSITORY", ""))

class VectorStoreConfig(BaseModel):
    """Vector store configuration settings for ChromaDB."""
    persistence_directory: str = Field(default=os.getenv("CHROMA_PERSISTENCE_DIR", "./chroma_db"))
    collection_name: str = Field(default=os.getenv("CHROMA_COLLECTION_NAME", "github-code-index"))

class LLMConfig(BaseModel):
    """LLM configuration settings."""
    anthropic_api_key: str = Field(default=os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    use_claude: bool = Field(default=os.getenv("USE_CLAUDE", "false").lower() == "true")
    use_openai: bool = Field(default=os.getenv("USE_OPENAI", "false").lower() == "true")
    model_name: str = Field(default=os.getenv("MODEL_NAME", "o3-mini"))
    max_tokens: int = Field(default=int(os.getenv("MAX_TOKENS_RESPONSE", "4000")))

class SlackConfig(BaseModel):
    """Slack configuration settings."""
    bot_token: str = Field(default=os.getenv("SLACK_BOT_TOKEN", ""))
    signing_secret: str = Field(default=os.getenv("SLACK_SIGNING_SECRET", ""))
    channel_id: str = Field(default=os.getenv("SLACK_CHANNEL_ID", ""))

class ChunkingConfig(BaseModel):
    """Text chunking configuration settings."""
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "200")))

class AppConfig(BaseModel):
    """Main application configuration."""
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

# Create a singleton configuration instance
config = AppConfig() 