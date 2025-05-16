package seed

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"smartsearch/internal/vector"
	"smartsearch/pkg/config"
	"smartsearch/pkg/models"
	"smartsearch/pkg/search"
	"time"

	"github.com/opensearch-project/opensearch-go"
)

type SeedData struct {
	Tools []models.Tool `json:"tools"`
}

func SeedDatabase(
	ctx context.Context,
	db *sql.DB,
	opensearchClient *opensearch.Client,
	cfg *config.Config,
) error {
	// Initialize vector store and search client
	vectorStore := vector.NewVectorStore(db, cfg.Ollama.URL, cfg.Ollama.EmbedModel, cfg)
	searchClient := search.NewOpenSearchClient(opensearchClient, "tools")

	// Read seed data
	data, err := os.ReadFile(filepath.Join("seed", "data", "tools.json"))
	if err != nil {
		return fmt.Errorf("failed to read seed data: %w", err)
	}

	var seedData SeedData
	if err := json.Unmarshal(data, &seedData); err != nil {
		return fmt.Errorf("failed to unmarshal seed data: %w", err)
	}

	// Set timestamps for all tools
	now := time.Now()
	for i := range seedData.Tools {
		seedData.Tools[i].CreatedAt = now
		seedData.Tools[i].UpdatedAt = now
	}

	// Store tools in both PostgreSQL and OpenSearch
	for _, tool := range seedData.Tools {
		// Store in PostgreSQL with vector embedding
		if err := vectorStore.IndexTool(ctx, tool); err != nil {
			return fmt.Errorf("failed to index tool %s in PostgreSQL: %w", tool.ID, err)
		}

		// Store in OpenSearch
		if err := searchClient.IndexTool(ctx, tool); err != nil {
			return fmt.Errorf("failed to index tool %s in OpenSearch: %w", tool.ID, err)
		}
	}

	fmt.Printf("Successfully seeded %d tools\n", len(seedData.Tools))
	return nil
}
