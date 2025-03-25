# GitHub Codebase Scanner and Q&A Bot

A Python application that scans GitHub repositories, indexes the code, and provides intelligent answers to questions through a Slack bot or direct queries.

## Features

- **Repository Indexing**: Clone and index GitHub repositories
- **Semantic Search**: Find relevant code snippets using vector similarity search
- **LLM Integration**: Process queries using Claude or OpenAI models
- **Slack Integration**: Answer questions about the codebase directly in Slack
- **Command Line Interface**: Index repositories and query the codebase directly
- **RESTful API**: Query the codebase using a web API

## Setup

### Prerequisites

- Python 3.8 or later
- A GitHub access token with repo scope
- (Optional) A Slack bot token and signing secret
- An API key for at least one of the supported LLM providers (Claude or OpenAI)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd github-code-qa
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file by copying the example:
   ```
   cp .env.example .env
   ```

5. Edit the `.env` file to include your API keys and configuration.

## Usage

### Indexing a Repository

```
python -m app.main index --repo <owner/repo> --clear
```

Options:
- `--repo`: The GitHub repository to index (e.g., `owner/repo`)
- `--clear`: Clear existing index before indexing
- `--extensions`: Specify file extensions to index (e.g., `py js`)

### Starting the Slack Bot

```
python -m app.main slack
```

Options:
- `--socket-mode`: Use Slack's Socket Mode (requires app-level token)

### Direct Querying

```
python -m app.main query "How does the authentication system work?"
```

### Starting the Web API

```
python -m app.main server --port 8000
```

Options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)

## Cost Comparison (LLM Providers)

| Provider | Model | Cost Per Million Tokens (Input) | Cost Per Million Tokens (Output) | Notes |
|----------|-------|--------------------------------|----------------------------------|-------|
| OpenAI   | o3-mini | $0.15 | $0.60 | Good performance at lower cost |
| Claude   | claude-3-5-sonnet | $3.00 | $15.00 | Strong code understanding |
| DeepSeek | deepseek-chat | $0.30 - $0.50 | $0.30 - $0.50 | Good cost/performance ratio |

## Architecture

The application consists of several components:

1. **GitHub Repository Handler**: Clones repositories and extracts code files
2. **Text Processing**: Chunks code into appropriate segments for indexing
3. **Vector Store**: Uses ChromaDB to create and query code embeddings
4. **LLM Integration**: Handles communication with different LLM providers
5. **Slack Bot**: Provides a conversational interface in Slack
6. **Web API**: Offers RESTful endpoints for querying

## License

MIT 