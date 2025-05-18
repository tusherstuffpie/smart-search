package vector

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"smartsearch/pkg/config"
	"smartsearch/pkg/models"

	"github.com/lib/pq"
	pgvector "github.com/pgvector/pgvector-go"
)

type VectorStore struct {
	db           *sql.DB
	ollamaClient *OllamaClient
	targetDim    int
}

func NewVectorStore(db *sql.DB, ollamaURL, model string, cfg *config.Config) *VectorStore {
	return &VectorStore{
		db:           db,
		ollamaClient: NewOllamaClient(ollamaURL, model),
		targetDim:    cfg.Ollama.Options.Dimensionality,
	}
}

func (vs *VectorStore) getEmbedding(ctx context.Context, text string) ([]float32, error) {
	embedding, err := vs.ollamaClient.GetEmbedding(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %w", err)
	}
	log.Printf("Generated embedding with %d dimensions", len(embedding))

	// Reduce dimensions if needed
	if len(embedding) > vs.targetDim {
		embedding = reduceDimensions(embedding, vs.targetDim)
	}

	return embedding, nil
}
func (vs *VectorStore) Search(ctx context.Context, query string, k int) ([]models.SearchResult, error) {
	log.Print("query: ", query)
	// Get embedding for query
	embedding, err := vs.getEmbedding(ctx, query)
	if err != nil {
		return nil, err
	}

	// Convert embedding to pgvector format
	vec := pgvector.NewVector(embedding)
	log.Printf("Query vector created with %d dimensions", len(embedding))

	// Search for the k most similar tools by cosine distance
	rows, err := vs.db.QueryContext(ctx, `
		SELECT 
			t.id, t.name, t.description, t.category, t.tags,
			t.input_schema, t.output_schema, t.version,
			t.created_at, t.updated_at,
			(t.embedding <=> $1::vector) AS distance
		FROM tools t
		WHERE t.embedding IS NOT NULL
		ORDER BY distance ASC
		LIMIT $2
	`, vec, 5)
	if err != nil {
		log.Printf("Query error: %v", err)
		return nil, fmt.Errorf("failed to query vector store: %w", err)
	}
	defer rows.Close()

	var results []models.SearchResult
	for rows.Next() {
		var tool models.Tool
		var distance float64
		var inputSchemaJSON, outputSchemaJSON []byte
		err := rows.Scan(
			&tool.ID, &tool.Name, &tool.Description, &tool.Category, pq.Array(&tool.Tags),
			&inputSchemaJSON, &outputSchemaJSON, &tool.Version,
			&tool.CreatedAt, &tool.UpdatedAt,
			&distance,
		)
		if err != nil {
			log.Printf("Error scanning row: %v", err)
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		// Unmarshal JSON schemas
		if len(inputSchemaJSON) > 0 {
			if err := json.Unmarshal(inputSchemaJSON, &tool.InputSchema); err != nil {
				log.Printf("Error unmarshaling input schema: %v", err)
				return nil, fmt.Errorf("failed to unmarshal input schema: %w", err)
			}
		}
		if len(outputSchemaJSON) > 0 {
			if err := json.Unmarshal(outputSchemaJSON, &tool.OutputSchema); err != nil {
				log.Printf("Error unmarshaling output schema: %v", err)
				return nil, fmt.Errorf("failed to unmarshal output schema: %w", err)
			}
		}

		// Convert distance to similarity score (1 - distance)
		similarity := 1 - distance
		log.Printf("Found tool: %s with distance: %f, similarity: %f", tool.Name, distance, similarity)
		results = append(results, models.SearchResult{
			Tool:        tool,
			VectorScore: similarity,
			Score:       similarity,
		})
	}

	if err = rows.Err(); err != nil {
		log.Printf("Error iterating rows: %v", err)
		return nil, fmt.Errorf("error iterating rows: %w", err)
	}

	log.Printf("Total results found: %d", len(results))
	return results, nil
}

func (vs *VectorStore) IndexTool(ctx context.Context, tool models.Tool) error {
	// Get embedding for tool description
	embedding, err := vs.getEmbedding(ctx, tool.Description)
	if err != nil {
		return err
	}

	// Convert embedding to pgvector format
	vec := pgvector.NewVector(embedding)

	// Convert input_schema and output_schema to JSONB
	inputSchemaJSON, err := json.Marshal(tool.InputSchema)
	if err != nil {
		return fmt.Errorf("failed to marshal input schema: %w", err)
	}

	outputSchemaJSON, err := json.Marshal(tool.OutputSchema)
	if err != nil {
		return fmt.Errorf("failed to marshal output schema: %w", err)
	}

	// Insert or update tool with embedding
	_, err = vs.db.ExecContext(ctx, `
        INSERT INTO tools (
            id, name, description, category, tags,
            input_schema, output_schema, version,
            created_at, updated_at, embedding
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (id) DO UPDATE SET
            name = $2,
            description = $3,
            category = $4,
            tags = $5,
            input_schema = $6,
            output_schema = $7,
            version = $8,
            updated_at = $10,
            embedding = $11
    `, tool.ID, tool.Name, tool.Description, tool.Category, pq.Array(tool.Tags),
		inputSchemaJSON, outputSchemaJSON, tool.Version,
		tool.CreatedAt, tool.UpdatedAt, vec)
	if err != nil {
		return fmt.Errorf("failed to index tool: %w", err)
	}

	return nil
}