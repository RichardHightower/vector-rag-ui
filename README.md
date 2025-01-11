# Vector RAG UI

A Streamlit-based user interface for document processing and retrieval using the vector-rag library.

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- OpenAI API key
- [Task](https://taskfile.dev/) - Task runner

## Quick Start

This project uses Task for managing common operations. View available tasks:
```bash
task -l
```

1. Set up the development environment:
```bash
task setup
```

2. Start the database:
```bash
task db:up
```

3. Run the application:
```bash
task run:app
```

## Available Tasks

- `task setup` - Set up development environment (installs dependencies)
- `task db:up` - Start the PostgreSQL database
- `task db:down` - Stop the PostgreSQL database
- `task run:app` - Start the Streamlit application
- `task run:example` - Run the example script
- `task clean` - Clean up generated files and virtual environment
- `task reset` - Clean everything and set up again

## Environment Setup

Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=your_db_name
CHUNK_SIZE=25
CHUNK_OVERLAP=3
```

## Features

- Create and manage projects
- Upload and process documents
- Search through documents using natural language queries
- View and explore search results
