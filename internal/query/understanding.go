package query

import (
	"context"
	"encoding/json"
	"fmt"
	"smartsearch/internal/ollama"
	"smartsearch/pkg/models"
)

type QueryEngine struct {
	client *ollama.Client
}

func NewQueryEngine(url, model string) *QueryEngine {
	return &QueryEngine{
		client: ollama.NewClient(url, model),
	}
}

func (qe *QueryEngine) UnderstandQuery(ctx context.Context, query string) (*models.QueryUnderstanding, error) {
	// Prepare prompt for query understanding
	prompt := fmt.Sprintf(`Analyze this query for tool search and provide structured understanding:
Query: %s

Provide a JSON response with the following structure:
{
    "intent": "The main purpose or goal of the query",
    "expanded_terms": ["List of relevant terms that expand the query"],
    "sub_queries": ["Any sub-queries that should be considered"],
    "filters": {
        "category": "Optional category filter",
        "tags": ["Optional tag filters"],
        "other_filters": {}
    }
}`, query)

	// Get completion from Ollama
	messages := []ollama.ChatMessage{
		{Role: "system", Content: "You are a query understanding system for a tool search service."},
		{Role: "user", Content: prompt},
	}

	response, err := qe.client.GetChatCompletion(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("failed to get query understanding: %w", err)
	}

	// Parse response
	var understanding models.QueryUnderstanding
	if err := json.Unmarshal([]byte(response), &understanding); err != nil {
		return nil, fmt.Errorf("failed to parse query understanding: %w", err)
	}

	return &understanding, nil
}

func (qe *QueryEngine) ExpandQuery(ctx context.Context, query string) ([]string, error) {
	// Get query understanding
	understanding, err := qe.UnderstandQuery(ctx, query)
	if err != nil {
		return nil, err
	}

	// Combine original query with expanded terms
	expandedQueries := []string{query}
	expandedQueries = append(expandedQueries, understanding.ExpandedTerms...)
	expandedQueries = append(expandedQueries, understanding.SubQueries...)

	return expandedQueries, nil
} 