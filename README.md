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
- `--force-full`: Force a full re-indexing instead of incremental update (optional)
- `--list-repos`: List all indexed repositories (optional)

**Examples:**

Index a repository for the first time (full indexing):
```bash
python -m app.main index --repo microsoft/TypeScript --branch main
```

Update an already indexed repository (incremental indexing):
```bash
python -m app.main index --repo microsoft/TypeScript
```

Force a complete re-indexing of a repository:
```bash
python -m app.main index --repo microsoft/TypeScript --force-full
```

List all indexed repositories:
```bash
python -m app.main index --list-repos
```

### Multiple Repository Support

AI Code Context now supports indexing and querying multiple repositories independently. Each repository is stored in its own vector store collection, ensuring that:

1. Indexing a new repository doesn't overwrite previously indexed repositories
2. You can switch between repositories without reindexing
3. Incremental updates only process files that have changed since the last indexing

When querying, the system automatically uses the correct repository:

```bash
# Query the first repository
python -m app.main query --repo owner/repo1 --query "How does feature X work?"

# Query a different repository
python -m app.main query --repo owner/repo2 --query "How does feature Y work?"
```

### Incremental Indexing

When you re-index an already indexed repository, the system will perform an incremental update by default:

1. Checks the repository's last indexed timestamp
2. Fetches only files that have been added or modified since that time
3. Updates the vector store with just the new/changed files

This significantly speeds up the indexing process for repositories that have already been indexed once. To force a complete re-indexing, use the `--force-full` flag.

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

### Interactive Chat Mode

For a more conversational experience, you can use the chat mode which maintains conversation history and allows for back-and-forth interaction.

```bash
python -m app.main chat
```

**Parameters:**
- `--show-snippets`: Display code snippets in responses (optional, off by default)
- `--explain`: Include detailed code explanations (optional, off by default) 
- `--generate-docs`: Generate documentation (optional, off by default)
- `--history-file`: Path to save/load chat history (optional)

**Example - Basic chat:**
```bash
python -m app.main chat
```

**Example - Chat with code snippets and saved history:**
```bash
python -m app.main chat --show-snippets --history-file ./chat_history.json
```

**Chat Commands:**
- `exit`, `quit`, or `q` - Exit chat mode
- `clear` - Clear conversation history
- `help` - Show available commands
- `snippets on/off` - Toggle code snippet display
- `explain on/off` - Toggle code explanation

The chat mode maintains context between questions, allowing for follow-up questions and a more natural conversation flow. The conversation history is saved to the specified file (if provided) so you can continue conversations across sessions.

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

### Interactive Learning Sessions

Use the chat mode for extended learning sessions about a codebase:

```bash
python -m app.main chat --history-file ./learning_session.json
```

Example chat session:
```
> What are the key components of this application?
[Response explaining components]

> How does the error handling work?
[Response explaining error handling, with context from previous question]

> Show me examples of error handling
[Response with examples, building on previous context]

> explain on
Code explanations enabled.

> snippets on
Code snippets enabled.

> How could I improve the error handling?
[Response with explanation and code snippets]
```

This approach allows for natural exploration of a codebase, with the AI maintaining context between questions. 

## Docker Setup

The application is fully containerized and can be run using Docker Compose:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Available Services

- **PostgreSQL**: Database for structured data storage
- **Redis**: Caching and task queue
- **Celery Worker**: Asynchronous task processing
- **FastAPI**: Backend API for the Next.js frontend

### Environment Variables for Docker

When using Docker, the environment variables are defined in `docker-compose.yml`. For local development, you can copy the `.env.example` file:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## API Usage

The application provides a REST API for integration with frontend applications:

### Authentication

```bash
# Register a new user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "email": "user@example.com", "password": "password"}'

# Login to get JWT token
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'
```

### Repository Management

```bash
# Create a new repository
curl -X POST http://localhost:8000/api/repositories \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"name": "My Repo", "url": "owner/repo", "branch": "main"}'

# List repositories
curl -X GET http://localhost:8000/api/repositories \
  -H "Authorization: Bearer <token>"
```

### Chat API

The application supports both conversation-based chat and stateless stream queries:

```bash
# Create a conversation
curl -X POST http://localhost:8000/api/chats/conversations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"title": "My Chat", "repository_id": "<repo_id>"}'

# Add a message to a conversation (streaming)
# Note: This requires SSE client support
curl -X POST http://localhost:8000/api/chats/conversations/<conversation_id>/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"content": "How is error handling implemented?"}'

# Stream a stateless query
curl -X POST http://localhost:8000/api/chats/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "Explain how testing works", "repository_id": "<repo_id>", "include_snippets": true}'
```

## Next.js Integration

To use the API with a Next.js frontend:

1. Use the Server-Sent Events (SSE) client to receive streaming responses
2. Connect to the appropriate endpoints for chat functionality
3. Handle authentication using the JWT token

Example Next.js code for streaming chat:

```javascript
import { useEffect, useState } from 'react';

function ChatComponent() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

  const sendMessage = async () => {
    const newMessage = { content: inputValue, isUser: true };
    setMessages(prev => [...prev, newMessage]);
    setInputValue('');

    // Create an EventSource for SSE
    const eventSource = new EventSource(
      `http://localhost:8000/api/chats/stream?query=${encodeURIComponent(inputValue)}`,
      { 
        headers: { 
          'Authorization': `Bearer ${localStorage.getItem('token')}` 
        } 
      }
    );

    let responseContent = '';

    eventSource.addEventListener('message', (event) => {
      const data = JSON.parse(event.data);
      
      if (!data.is_complete) {
        responseContent += data.content;
        setMessages(prev => [
          ...prev.slice(0, -1),
          newMessage,
          { content: responseContent, isUser: false, isComplete: false }
        ]);
      } else {
        setMessages(prev => [
          ...prev.slice(0, -1),
          newMessage,
          { 
            content: responseContent, 
            isUser: false, 
            isComplete: true,
            metadata: data.metadata 
          }
        ]);
        eventSource.close();
      }
    });
    
    eventSource.onerror = () => {
      eventSource.close();
    };
  };

  return (
    <div>
      {/* Chat messages */}
      <div>
        {messages.map((msg, i) => (
          <div key={i} className={msg.isUser ? 'user-message' : 'system-message'}>
            {msg.content}
            {msg.metadata?.code_snippets && (
              <div className="code-snippets">
                {msg.metadata.code_snippets.map((snippet, j) => (
                  <pre key={j}><code>{snippet.code}</code></pre>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      
      {/* Input */}
      <div>
        <input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}
```

### Repository Isolation and Collection Management

The application uses a sophisticated approach to manage multiple repositories:

#### Collection Naming Strategy

Each repository gets its own isolated ChromaDB collection using the following naming convention:
```
{collection_name}_{owner_id}_{repository_id}
```

For example:
```
code_chunks_f7e8a1b2_c9d0e3f4
```

This ensures:
1. Complete isolation between repositories
2. No data leakage or collisions
3. Clean organization of vector data

#### How Repository Management Works

When you use the application:

1. **First time indexing a repository**:
   - Creates a new Repository record in the database
   - Assigns a unique UUID to the repository
   - Creates a dedicated ChromaDB collection
   
2. **Switching between repositories**:
   - Simply use the `--repo` flag to specify which repository to use
   - No need to re-index when switching
   
3. **View available repositories**:
   ```bash
   python -m app.main index --list-repos
   ```

#### Benefits Over Global Collection

This approach provides several advantages:
- Avoids reindexing when changing repositories
- Allows having multiple repositories ready at the same time
- Keeps vector search results specific to a single codebase
- Supports different branches or versions of the same codebase 

### Default Repository Configuration

The application supports setting default repositories in your `.env` file, which are used when no `--repo` flag is provided:

```bash
# In .env
GITHUB_REPOSITORY=owner/repo
GITHUB_BRANCH=main
```

With this configuration:

```bash
# Uses owner/repo from the .env file
python -m app.main index

# Uses specific repository, overriding .env
python -m app.main index --repo different/repo

# Queries the default repository from .env
python -m app.main query --query "How does feature X work?"

# Starts chat session with the default repository
python -m app.main chat
```

This makes it convenient to work with a primary repository while still having the flexibility to switch when needed.

#### Command Priority Order

The application follows this order of precedence for repository selection:
1. Command-line parameter (`--repo` flag)
2. Environment variable (`GITHUB_REPOSITORY` in `.env`)
3. Error if no repository is specified 