"""
Application configuration settings.
"""
import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChunkingConfig:
    """Configuration for code chunking."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    min_chunk_size: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))

@dataclass
class GitHubConfig:
    """Configuration for GitHub integration."""
    access_token: str = os.getenv("GITHUB_ACCESS_TOKEN", "")
    repository: str = os.getenv("GITHUB_REPOSITORY", "")
    branch: str = os.getenv("GITHUB_BRANCH", "main")
    supported_file_types: List[str] = field(default_factory=lambda: os.getenv(
        "SUPPORTED_FILE_TYPES", 
        # Programming Languages
        "py,js,ts,jsx,tsx,java,cpp,c,h,hpp,cs,go,rs,rb,php,swift,kt,scala,r,m,sh,"
        # Web Technologies
        "html,htm,css,scss,sass,less,jsx,tsx,"
        # Data Formats (excluding binary/structured data)
        "json,json5,jsonc,yaml,yml,toml,xml,"
        # Documentation
        "md,mdown,markdown,rst,txt,"
        # Configuration
        "env,env.*,config,conf,ini,properties,"
        # Database
        "sql,prisma,graphql,"
        # Docker & Container
        "dockerfile,docker-compose.yml,"
        # CI/CD
        "github/workflows/*.yml,gitlab-ci.yml,.gitlab-ci.yml,"
        # Testing
        "test.js,test.ts,spec.js,spec.ts,"
        # Package Management
        "package.json,"
        # Misc
        "svg,graphql,proto"
    ).split(","))
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "1000000"))  # 1MB
    excluded_dirs: List[str] = field(default_factory=lambda: os.getenv(
        "EXCLUDED_DIRS", 
        # Build and dist directories
        "dist,build,.next,out,target,generated,generated-sources,"
        # Cache and temporary directories
        ".cache,coverage,.pytest_cache,.mypy_cache,.ruff_cache,.coverage,"
        # Dependencies and virtual environments
        "node_modules,.venv,venv,env,.env,"
        # Version control and IDE
        ".git,.idea,.vscode,.DS_Store,"
        # Test directories
        "test,tests,__tests__,__mocks__,"
        # Package manager specific
        "vendor,packages,examples"
    ).split(","))

@dataclass
class EmbeddingConfig:
    """Configuration for code embeddings."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device: str = os.getenv("EMBEDDING_DEVICE", "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
    normalize: bool = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    use_pooling: bool = os.getenv("USE_EMBEDDING_POOLING", "true").lower() == "true"
    pooling_strategy: str = os.getenv("POOLING_STRATEGY", "mean")
    use_ast: bool = os.getenv("USE_AST", "false").lower() == "true"
    hybrid_embedding: bool = os.getenv("HYBRID_EMBEDDING", "false").lower() == "true"
    cache_embeddings: bool = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
    cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", ".embedding_cache")

@dataclass
class VectorStoreConfig:
    """Vector store configuration settings."""
    persistence_dir: str = os.getenv("CHROMA_PERSISTENCE_DIR", "chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "code_chunks")  # Base name for collections
    batch_size: int = int(os.getenv("CHROMA_BATCH_SIZE", "100"))
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")  # Options: cosine, l2, ip
    use_distributed: bool = os.getenv("USE_DISTRIBUTED_STORE", "false").lower() == "true"
    use_gpu: bool = os.getenv("USE_GPU", "false").lower() == "true"
    num_workers: int = int(os.getenv("NUM_WORKERS", "4"))
    num_shards: int = int(os.getenv("NUM_SHARDS", "4"))
    shard_size: int = int(os.getenv("SHARD_SIZE", "10000"))
    replication_factor: int = int(os.getenv("REPLICATION_FACTOR", "2"))
    load_balance_threshold: float = float(os.getenv("LOAD_BALANCE_THRESHOLD", "0.8"))

@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    use_openai: bool = os.getenv("USE_OPENAI", "true").lower() == "true"
    use_anthropic: bool = os.getenv("USE_ANTHROPIC", "false").lower() == "true"

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = os.getenv("LOG_DIR", "logs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_bytes: int = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    query_reformulation: bool = os.getenv("QUERY_REFORMULATION", "true").lower() == "true"
    conversation_history: bool = os.getenv("CONVERSATION_HISTORY", "true").lower() == "true"
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "20"))

@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    output_dir: str = os.getenv("DOCS_OUTPUT_DIR", "docs")
    include_api_docs: bool = os.getenv("INCLUDE_API_DOCS", "true").lower() == "true"
    include_examples: bool = os.getenv("INCLUDE_EXAMPLES", "true").lower() == "true"
    include_diagrams: bool = os.getenv("INCLUDE_DIAGRAMS", "false").lower() == "true"

@dataclass
class AnalyticsConfig:
    """Configuration for analytics and monitoring."""
    metrics_dir: str = os.getenv("METRICS_DIR", "metrics")
    track_system_metrics: bool = os.getenv("TRACK_SYSTEM_METRICS", "true").lower() == "true"
    track_query_metrics: bool = os.getenv("TRACK_QUERY_METRICS", "true").lower() == "true"
    metrics_retention_days: int = int(os.getenv("METRICS_RETENTION_DAYS", "30"))

class DatabaseConfig:
    """Database configuration settings."""
    url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ai_code_context")
    echo: bool = os.getenv("DATABASE_ECHO", "false").lower() == "true"
    pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
    pool_recycle: int = int(os.getenv("DATABASE_POOL_RECYCLE", "1800"))  # 30 minutes

@dataclass
class Config:
    """Main configuration class."""
    chunking: ChunkingConfig
    github: GitHubConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    llm: LLMConfig
    logging: LoggingConfig
    rag: RAGConfig
    documentation: DocumentationConfig
    analytics: AnalyticsConfig
    database: DatabaseConfig
    
    def __init__(self):
        """Initialize configuration with all components."""
        # Create configuration objects
        self.chunking = ChunkingConfig()
        self.github = GitHubConfig()
        self.embedding = EmbeddingConfig()
        self.vector_store = VectorStoreConfig()
        self.llm = LLMConfig()
        self.logging = LoggingConfig()
        self.rag = RAGConfig()
        self.documentation = DocumentationConfig()
        self.analytics = AnalyticsConfig()
        self.database = DatabaseConfig()
        
        # Create necessary directories
        self._create_directories()
        
        # Validate configuration
        self._validate()
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.logging.log_dir,
            self.vector_store.persistence_dir,
            self.documentation.output_dir,
            self.analytics.metrics_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate(self):
        """Validate configuration settings."""
        # Validate API keys
        if not self.llm.openai_api_key and not self.llm.anthropic_api_key:
            print("WARNING: No LLM API keys provided. At least one of OPENAI_API_KEY or ANTHROPIC_API_KEY must be set in the environment.")
        
        # Validate chunking settings
        if self.chunking.chunk_size < self.chunking.min_chunk_size:
            raise ValueError(f"chunk_size ({self.chunking.chunk_size}) must be >= min_chunk_size ({self.chunking.min_chunk_size})")
        if self.chunking.chunk_size > self.chunking.max_chunk_size:
            raise ValueError(f"chunk_size ({self.chunking.chunk_size}) must be <= max_chunk_size ({self.chunking.max_chunk_size})")
        
        # Validate GitHub settings - make these warnings rather than errors
        if not self.github.access_token:
            print("WARNING: GitHub access token not provided. Set GITHUB_ACCESS_TOKEN in your environment to use GitHub features.")
        if not self.github.repository:
            print("WARNING: GitHub repository not specified. Set GITHUB_REPOSITORY in your environment to use GitHub features.")
        
        # Validate vector store settings
        if self.vector_store.use_distributed:
            if self.vector_store.num_shards < 1:
                raise ValueError("num_shards must be at least 1")
            if self.vector_store.replication_factor < 1:
                raise ValueError("replication_factor must be at least 1")
            if self.vector_store.load_balance_threshold <= 0 or self.vector_store.load_balance_threshold >= 1:
                raise ValueError("load_balance_threshold must be between 0 and 1")
        
        # Validate embedding settings
        if not self.embedding.model_name:
            raise ValueError("embedding model must be specified")
        if self.embedding.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.embedding.max_length < 1:
            raise ValueError("max_length must be at least 1")
        
        # Validate logging settings
        if self.logging.max_bytes < 1024:  # 1KB
            raise ValueError("max_bytes must be at least 1KB")
        if self.logging.backup_count < 1:
            raise ValueError("backup_count must be at least 1")
        
        # Validate RAG settings
        if self.rag.max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1")
        if self.rag.context_window < 1:
            raise ValueError("context_window must be at least 1")
        
        # Validate analytics settings
        if self.analytics.metrics_retention_days < 1:
            raise ValueError("metrics_retention_days must be at least 1")

# Create global configuration instance
config = Config() 