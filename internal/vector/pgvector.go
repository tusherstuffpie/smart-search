package vector

import (
	"context"
	"database/sql"
	"fmt"
	"smartsearch/pkg/models"

	pgvector "github.com/pgvector/pgvector-go"
	"github.com/sashabaranov/go-openai"
)

type VectorStore struct {
    db          *sql.DB
    openaiClient *openai.Client
    model       string
}

func NewVectorStore(db *sql.DB, openaiClient *openai.Client, model string) *VectorStore {
    return &VectorStore{
        db:          db,
        openaiClient: openaiClient,
        model:       model,
    }
}

func (vs *VectorStore) getEmbedding(ctx context.Context, text string) ([]float32, error) {
    resp, err := vs.openaiClient.CreateEmbeddings(ctx, openai.EmbeddingRequest{
        Input: []string{text},
        Model: openai.AdaEmbeddingV2,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create embedding: %w", err)
    }
    return resp.Data[0].Embedding, nil
}

func (vs *VectorStore) Search(ctx context.Context, query string, k int) ([]models.SearchResult, error) {
    // Get embedding for query
    embedding, err := vs.getEmbedding(ctx, query)
    if err != nil {
        return nil, err
    }

    // Convert embedding to pgvector format
    vec := pgvector.NewVector(embedding)

    // Search using cosine similarity
    rows, err := vs.db.QueryContext(ctx, `
        SELECT 
            t.id, t.name, t.description, t.category, t.tags,
            t.input_schema, t.output_schema, t.version,
            t.created_at, t.updated_at,
            1 - (t.embedding <=> $1) as similarity
        FROM tools t
        ORDER BY similarity DESC
        LIMIT $2
    `, vec, k)
    if err != nil {
        return nil, fmt.Errorf("failed to query vector store: %w", err)
    }
    defer rows.Close()

    var results []models.SearchResult
    for rows.Next() {
        var tool models.Tool
        var similarity float64
        err := rows.Scan(
            &tool.ID, &tool.Name, &tool.Description, &tool.Category, &tool.Tags,
            &tool.InputSchema, &tool.OutputSchema, &tool.Version,
            &tool.CreatedAt, &tool.UpdatedAt,
            &similarity,
        )
        if err != nil {
            return nil, fmt.Errorf("failed to scan row: %w", err)
        }

        results = append(results, models.SearchResult{
            Tool:        tool,
            VectorScore: similarity,
            Score:       similarity,
        })
    }

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
    `, tool.ID, tool.Name, tool.Description, tool.Category, tool.Tags,
        tool.InputSchema, tool.OutputSchema, tool.Version,
        tool.CreatedAt, tool.UpdatedAt, vec)
    if err != nil {
        return fmt.Errorf("failed to index tool: %w", err)
    }

    return nil
}