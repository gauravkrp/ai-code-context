# AI Code Context

A powerful application for indexing and querying code repositories using AI. This tool provides intelligent code search, explanation, and documentation capabilities using advanced RAG (Retrieval-Augmented Generation) techniques.

## Features

### Advanced Code Understanding
- **Specialized Code Embeddings**
  - CodeBERT and GraphCodeBERT integration
  - AST-based code embeddings
  - Cross-language code embeddings
  - Function-level and class-level embeddings
  - Language-specific parsers for multiple languages

- **Enhanced Code Analysis**
  - AST-based code structure analysis
  - Code dependency tracking
  - Code flow analysis
  - Code complexity metrics
  - Type inference and validation

### Intelligent RAG System
- **Advanced Query Processing**
  - Query intent detection (explain, bug, feature, usage, implementation)
  - Query reformulation for better search results
  - Dynamic context window sizing
  - Conversation history support
  - Context-aware responses

- **Hybrid Search**
  - Dense and sparse retriever combination
  - Code-specific reranking using BM25
  - Vector similarity search
  - Code knowledge graph integration
  - Multi-stage retrieval pipeline

### Documentation Generation
- **Comprehensive Documentation**
  - API documentation generation
  - Module documentation
  - Code examples
  - Project README generation
  - Documentation site generation
  - Code explanation capabilities

### Performance & Monitoring
- **Analytics System**
  - System metrics tracking
  - Query performance analytics
  - Usage statistics
  - Metrics retention management
  - Performance monitoring

- **Enhanced Logging**
  - Rotating log files
  - Component-specific logging
  - Detailed error tracking
  - Performance monitoring
  - Debug information

## Architecture

```
ai-code-context/
├── app/
│   ├── analytics/         # Analytics and monitoring system
│   │   ├── monitor.py     # System metrics tracking
│   │   └── metrics.py     # Performance metrics
│   ├── config/           # Configuration management
│   │   └── settings.py    # Application settings
│   ├── docs/             # Documentation generation
│   │   └── auto_documenter.py  # Auto-documentation system
│   ├── github/           # GitHub integration
│   │   ├── repo_scanner.py    # Repository scanning
│   │   └── indexer.py         # Code indexing
│   ├── rag/              # RAG system components
│   │   ├── advanced_rag.py    # Advanced RAG implementation
│   │   ├── query_optimizer.py # Query optimization
│   │   └── code_explainer.py  # Code explanation
│   ├── utils/            # Utility functions
│   │   ├── code_chunker.py    # Code chunking
│   │   ├── llm.py            # LLM integration
│   │   └── text_processing.py # Text processing
│   └── vector_store/     # Vector storage
│       └── chroma_store.py    # ChromaDB implementation
├── logs/                 # Application logs
├── metrics/              # Analytics metrics
├── docs/                 # Generated documentation
├── .env                  # Environment variables
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## System Flow

1. **Repository Indexing**
   ```
   GitHub Repository → Scanner → Code Chunker → Vector Store
   ```

2. **Query Processing**
   ```
   User Query → Query Optimizer → RAG System → LLM → Response
   ```

3. **Documentation Generation**
   ```
   Code → Auto Documenter → Documentation Site
   ```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-code-context.git
cd ai-code-context
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Getting Started

Here's a quick guide to get up and running:

1. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your GitHub token and OpenAI/Anthropic API key
   ```

2. **Index a repository**:
   ```bash
   python -m app.main index --repo owner/repo
   ```

3. **Query the codebase**:
   ```bash
   python -m app.main query --query "How does X work?"
   ```

That's it! You'll get a natural language explanation of the code based on your query. For more advanced usage, see the Usage section below.

## Configuration

Configure the application by creating a `.env` file in the project root directory. You can copy the `.env.example` file and modify it as needed.

### Required Environment Variables

- `GITHUB_ACCESS_TOKEN`: Your GitHub access token for repository access
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: At least one LLM API key is required

### GitHub Configuration

- `GITHUB_REPOSITORY`: Default repository to index in "owner/repo" format
- `GITHUB_BRANCH`: Default branch to index (defaults to "main")
- `SUPPORTED_FILE_TYPES`: Comma-separated list of file extensions to index (e.g., "py,js,ts,jsx,tsx")
- `EXCLUDED_DIRS`: Directories to exclude from indexing (e.g., "node_modules,.git,__pycache__")

### LLM Configuration

- `MODEL_NAME`: LLM model to use (default: "gpt-4")
- `USE_OPENAI`: Set to "true" to use OpenAI models
- `USE_ANTHROPIC`: Set to "true" to use Anthropic Claude models
- `TEMPERATURE`: Controls response randomness (0.0-1.0, default: 0.7)
- `MAX_TOKENS`: Maximum tokens in generated responses (default: 4000)

### Chunking Configuration

- `CHUNK_SIZE`: Size of code chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MIN_CHUNK_SIZE`: Minimum size for each chunk (default: 100)
- `MAX_CHUNK_SIZE`: Maximum size for each chunk (default: 2000)

### RAG System Configuration

- `QUERY_REFORMULATION`: Enable query reformulation (default: true)
- `CONVERSATION_HISTORY`: Enable conversation history (default: true)
- `MAX_HISTORY_TURNS`: Maximum conversation turns to remember (default: 5)
- `CONTEXT_WINDOW`: Number of code snippets to include in context (default: 3)

### Vector Store Configuration

- `CHROMA_PERSISTENCE_DIR`: Directory for ChromaDB persistence (default: "./chroma_db")
- `CHROMA_COLLECTION_NAME`: Collection name in ChromaDB (default: "code_chunks")
- `SIMILARITY_METRIC`: Similarity metric for vector search (default: "cosine")
- `USE_DISTRIBUTED_STORE`: Whether to use distributed storage (default: false)

### Logging and Analytics

- `LOG_LEVEL`: Logging verbosity (default: "INFO")
- `LOG_DIR`: Directory for log files (default: "logs")
- `TRACK_SYSTEM_METRICS`: Enable system metrics tracking (default: true)
- `TRACK_QUERY_METRICS`: Enable query metrics tracking (default: true)

For more advanced configuration options, see the `.env.example` file.

## Usage

### Indexing a Repository

The application needs to index a GitHub repository before it can answer questions about the code.

```bash
python -m app.main index --repo owner/repo --branch main
```

**Parameters:**
- `--repo`: The GitHub repository to index in the format "owner/repo" (overrides GITHUB_REPOSITORY from .env)
- `--branch`: The branch to index (overrides GITHUB_BRANCH from .env, defaults to "main")

Example:
```bash
python -m app.main index --repo microsoft/TypeScript --branch main
```

### Querying the Codebase

Once a repository is indexed, you can query it with natural language questions.

```bash
python -m app.main query --query "your question here" 
```

**Parameters:**
- `--query`: Your natural language question about the code (required)
- `--history`: JSON string of conversation history for contextual queries (optional)
- `--show-snippets`: Display code snippets in the output (optional, off by default)
- `--explain`: Generate detailed explanations of the code snippets (optional, off by default)
- `--generate-docs`: Generate documentation based on the query (optional, off by default)

**Example - Basic query:**
```bash
python -m app.main query --query "How are React hooks used for state management?"
```

**Example - Show code snippets:**
```bash
python -m app.main query --query "How are React hooks used for state management?" --show-snippets
```

**Example - With code explanations:**
```bash
python -m app.main query --query "How are React hooks used for state management?" --explain
```

**Example - With conversation history:**
```bash
python -m app.main query --query "How are they implemented?" --history '[{"query": "What are React hooks?", "answer": "React hooks are functions that..."}]'
```

### Understanding Output

The output is structured as follows:

1. **Response**: A natural language explanation answering your question
2. **Code Snippets** (optional, with `--show-snippets`): Relevant code from the repository
3. **Code Explanations** (optional, with `--explain`): Detailed explanation of each code snippet

### Advanced Usage

**Combining multiple flags:**
```bash
python -m app.main query --query "Explain the implementation of useState hook" --show-snippets --explain
```

**For documentation generation:**
```bash
python -m app.main query --query "Generate documentation for the repository" --generate-docs
```

## Logging

The application maintains separate log files for different components:
- `logs/app.log`: Main application logs
- `logs/github_scanner.log`: GitHub scanning logs
- `logs/vector_store.log`: Vector store operations
- `logs/llm.log`: LLM interactions
- `logs/auto_documenter.log`: Documentation generation logs

## Troubleshooting

### Common Issues

1. **LLM Service Unavailable**
   - Check your API keys in `.env`
   - Verify network connectivity
   - Check service status

2. **Vector Store Errors**
   - Verify ChromaDB installation
   - Check disk space
   - Verify permissions

3. **Documentation Generation Failures**
   - Check file permissions
   - Verify output directory exists
   - Check for syntax errors in code

### Performance Optimization

1. **Indexing Large Repositories**
   - Adjust chunk size and overlap
   - Use batch processing
   - Monitor memory usage

2. **Query Performance**
   - Enable caching
   - Optimize context window size
   - Use appropriate model size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- CodeBERT for code-specific embeddings

## Common Use Cases

Here are some examples of how to use the application for different purposes:

### Understanding a New Codebase

Index the repository and ask questions to quickly understand the codebase:

```bash
python -m app.main index --repo owner/repo
python -m app.main query --query "What is the high-level architecture of this project?"
python -m app.main query --query "What are the main components and how do they interact?"
```

### Finding How to Implement a Feature

```bash
python -m app.main query --query "How do I implement authentication in this codebase?"
python -m app.main query --query "What's the pattern for adding a new API endpoint?"
```

### Debugging Issues

```bash
python -m app.main query --query "Why might I be getting this error: [paste error message]"
python -m app.main query --query "What could cause this function to return null in these cases?"
```

### Learning Patterns and Techniques

```bash
python -m app.main query --query "How are React hooks used in this project?"
python -m app.main query --query "What design patterns are used for handling async operations?"
```

### Code Reviews and Quality

```bash
python -m app.main query --query "What areas of this codebase might need refactoring?"
python -m app.main query --query "Are there any potential security vulnerabilities in the authentication system?"
```

### Contribution Guide

```bash
python -m app.main query --query "What's the code style and contribution process for this project?"
python -m app.main query --query "How are tests structured and implemented in this project?"
``` 