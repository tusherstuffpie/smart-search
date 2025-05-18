package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
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

// CreateOrUpdateIndex creates or updates the index with proper settings
func (c *OpenSearchClient) CreateOrUpdateIndex(ctx context.Context) error {
	// Define index settings
	settings := map[string]interface{}{
		"settings": map[string]interface{}{
			"index": map[string]interface{}{
				"mapping": map[string]interface{}{
					"total_fields.limit": 2000, // Increase field limit
				},
			},
		},
		"mappings": map[string]interface{}{
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type": "text",
					"fields": map[string]interface{}{
						"keyword": map[string]interface{}{
							"type": "keyword",
						},
					},
				},
				"description": map[string]interface{}{
					"type": "text",
				},
				"category": map[string]interface{}{
					"type": "keyword",
				},
				"tags": map[string]interface{}{
					"type": "keyword",
				},
			},
		},
	}

	// Convert settings to JSON
	settingsJSON, err := json.Marshal(settings)
	if err != nil {
		return fmt.Errorf("failed to marshal index settings: %w", err)
	}

	// Check if index exists
	exists, err := c.client.Indices.Exists([]string{c.index})
	if err != nil {
		return fmt.Errorf("failed to check if index exists: %w", err)
	}

	if exists.StatusCode == 404 {
		// Create new index
		res, err := c.client.Indices.Create(
			c.index,
			c.client.Indices.Create.WithContext(ctx),
			c.client.Indices.Create.WithBody(strings.NewReader(string(settingsJSON))),
		)
		if err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
		defer res.Body.Close()

		if res.IsError() {
			return fmt.Errorf("error creating index: %s", res.String())
		}
	} else {
		// Update existing index settings
		res, err := c.client.Indices.PutSettings(
			strings.NewReader(string(settingsJSON)),
			c.client.Indices.PutSettings.WithContext(ctx),
			c.client.Indices.PutSettings.WithIndex(c.index),
		)
		if err != nil {
			return fmt.Errorf("failed to update index settings: %w", err)
		}
		defer res.Body.Close()

		if res.IsError() {
			return fmt.Errorf("error updating index settings: %s", res.String())
		}
	}

	return nil
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
		"track_total_hits": true,
	}

	// Convert query to JSON
	queryJSON, err := json.Marshal(searchQuery)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search query: %w", err)
	}

	log.Printf("OpenSearch query: %s", string(queryJSON))

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

	// Read and log the raw response
	bodyBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}
	log.Printf("OpenSearch raw response: %s", string(bodyBytes))

	// Create a new reader for the response body since we consumed it
	res.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

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

	log.Printf("OpenSearch found %d total hits", len(searchResponse.Hits.Hits))

	// Convert to SearchResult slice
	results := make([]models.SearchResult, 0, len(searchResponse.Hits.Hits))
	for _, hit := range searchResponse.Hits.Hits {
		log.Printf("Found tool: %s with score: %f", hit.Source.Name, hit.Score)
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

// VerifyIndexSettings checks if the index exists and has the correct settings
func (c *OpenSearchClient) VerifyIndexSettings(ctx context.Context) error {
	// Check if index exists
	exists, err := c.client.Indices.Exists([]string{c.index})
	if err != nil {
		return fmt.Errorf("failed to check if index exists: %w", err)
	}

	if exists.StatusCode == 404 {
		return fmt.Errorf("index %s does not exist", c.index)
	}

	// Get index settings
	res, err := c.client.Indices.GetSettings(
		c.client.Indices.GetSettings.WithContext(ctx),
		c.client.Indices.GetSettings.WithIndex(c.index),
	)
	if err != nil {
		return fmt.Errorf("failed to get index settings: %w", err)
	}
	defer res.Body.Close()

	// Read and log the settings
	bodyBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return fmt.Errorf("failed to read settings response: %w", err)
	}
	log.Printf("Index settings: %s", string(bodyBytes))

	// Get index mappings
	res, err = c.client.Indices.GetMapping(
		c.client.Indices.GetMapping.WithContext(ctx),
		c.client.Indices.GetMapping.WithIndex(c.index),
	)
	if err != nil {
		return fmt.Errorf("failed to get index mappings: %w", err)
	}
	defer res.Body.Close()

	// Read and log the mappings
	bodyBytes, err = io.ReadAll(res.Body)
	if err != nil {
		return fmt.Errorf("failed to read mappings response: %w", err)
	}
	log.Printf("Index mappings: %s", string(bodyBytes))

	return nil
} 