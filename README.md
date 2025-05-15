# Smart Search Service

A powerful search service that combines query understanding, hybrid search (vector + keyword), and re-ranking capabilities.

## Features

- Query Understanding: Analyzes user queries to understand intent and extract relevant information
- Hybrid Search: Combines vector search (using pgvector) and keyword search (using OpenSearch)
- Re-ranking: Uses OpenAI to re-rank and justify search results
- Tool Repository: Manages tool metadata and embeddings

## Prerequisites

- Go 1.21 or later
- PostgreSQL 12 or later with pgvector extension
- OpenSearch 2.x
- OpenAI API key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-search.git
   cd smart-search
   ```

2. Install dependencies:
   ```bash
   go mod download
   ```

3. Set up PostgreSQL with pgvector:
   ```sql
   CREATE EXTENSION vector;
   CREATE TABLE tools (
       id TEXT PRIMARY KEY,
       name TEXT NOT NULL,
       description TEXT,
       category TEXT,
       tags TEXT[],
       input_schema JSONB,
       output_schema JSONB,
       version TEXT,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
       embedding vector(1536)
   );
   ```

4. Configure OpenSearch:
   - Create an index for tools
   - Set up appropriate mappings for text fields
   - Configure authentication

5. Update `config.json` with your settings:
   - OpenAI API key
   - PostgreSQL connection string
   - OpenSearch credentials
   - Server configuration

## Running the Service

1. Start the service:
   ```bash
   go run cmd/smartsearch/main.go
   ```

2. The service will be available at `http://localhost:8080`

## API Usage

### Search Endpoint

```http
POST /search
Content-Type: application/json

{
    "query": "Find tools for data analysis",
    "filters": {
        "category": "data",
        "tags": ["analysis", "visualization"]
    },
    "top_k": 10,
    "min_score": 0.7
}
```

Response:
```json
{
    "results": [
        {
            "tool": {
                "id": "tool-1",
                "name": "Data Analysis Tool",
                "description": "A powerful tool for data analysis",
                "category": "data",
                "tags": ["analysis", "visualization"],
                "input_schema": {},
                "output_schema": {},
                "version": "1.0.0"
            },
            "score": 0.95,
            "vector_score": 0.92,
            "keyword_score": 0.98,
            "reranked_score": 0.95,
            "confidence": 0.9,
            "justification": "This tool is highly relevant for data analysis tasks"
        }
    ],
    "total": 1,
    "time_ms": 150
}
```

## Architecture

The service consists of several components:

1. Query Understanding Layer
   - Analyzes user queries
   - Extracts intent and relevant information
   - Generates expanded terms and sub-queries

2. Hybrid Search
   - Vector search using pgvector
   - Keyword search using OpenSearch
   - Score fusion and result merging

3. Re-ranking
   - Uses OpenAI to evaluate result relevance
   - Provides justifications for rankings
   - Adjusts final scores

4. Tool Repository
   - Manages tool metadata
   - Handles embeddings generation
   - Maintains search indices

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
