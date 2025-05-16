package search

import (
	"context"
	"encoding/json"
	"fmt"
	"smartsearch/pkg/models"
	"strings"

	"github.com/opensearch-project/opensearch-go"
)

type OpenSearchClient struct {
	client *opensearch.Client
	index  string
}

func NewOpenSearchClient(client *opensearch.Client, index string) *OpenSearchClient {
	return &OpenSearchClient{
		client: client,
		index:  index,
	}
}

func (c *OpenSearchClient) Search(ctx context.Context, query string, k int) ([]models.SearchResult, error) {
	// Construct search query
	searchQuery := map[string]interface{}{
		"query": map[string]interface{}{
			"multi_match": map[string]interface{}{
				"query":     query,
				"fields":    []string{"name^3", "description^2", "category", "tags"},
				"type":      "best_fields",
				"tie_breaker": 0.3,
			},
		},
		"size": k,
	}

	// Convert query to JSON
	queryJSON, err := json.Marshal(searchQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search query: %w", err)
	}

	// Execute search
	res, err := c.client.Search(
		c.client.Search.WithContext(ctx),
		c.client.Search.WithIndex(c.index),
		c.client.Search.WithBody(strings.NewReader(string(queryJSON))),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search: %w", err)
	}
	defer res.Body.Close()

	// Parse response
	var searchResponse struct {
		Hits struct {
			Total struct {
				Value int `json:"value"`
			} `json:"total"`
			Hits []struct {
				Source models.Tool `json:"_source"`
				Score  float64    `json:"_score"`
			} `json:"hits"`
		} `json:"hits"`
	}

	if err := json.NewDecoder(res.Body).Decode(&searchResponse); err != nil {
		return nil, fmt.Errorf("failed to decode search response: %w", err)
	}

	// Convert to SearchResult slice
	results := make([]models.SearchResult, 0, len(searchResponse.Hits.Hits))
	for _, hit := range searchResponse.Hits.Hits {
		results = append(results, models.SearchResult{
			Tool:         hit.Source,
			Score:        hit.Score,
			KeywordScore: hit.Score,
		})
	}

	return results, nil
}

func (c *OpenSearchClient) IndexTool(ctx context.Context, tool models.Tool) error {
	// Convert tool to JSON
	toolJSON, err := json.Marshal(tool)
	if err != nil {
		return fmt.Errorf("failed to marshal tool: %w", err)
	}

	// Index document
	res, err := c.client.Index(
		c.index,
		strings.NewReader(string(toolJSON)),
		c.client.Index.WithContext(ctx),
		c.client.Index.WithDocumentID(tool.ID),
	)
	if err != nil {
		return fmt.Errorf("failed to index tool: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("error indexing tool: %s", res.String())
	}

	return nil
}

func (c *OpenSearchClient) DeleteTool(ctx context.Context, toolID string) error {
	res, err := c.client.Delete(
		c.index,
		toolID,
		c.client.Delete.WithContext(ctx),
	)
	if err != nil {
		return fmt.Errorf("failed to delete tool: %w", err)
	}
	defer res.Body.Close()

	if res.IsError() {
		return fmt.Errorf("error deleting tool: %s", res.String())
	}

	return nil
} 