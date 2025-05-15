package query

import (
	"context"
	"encoding/json"
	"fmt"
	"smartsearch/pkg/models"

	"github.com/sashabaranov/go-openai"
)

type QueryEngine struct {
    client *openai.Client
    model  string
}

func NewQueryEngine(client *openai.Client, model string) *QueryEngine {
    return &QueryEngine{
        client: client,
        model:  model,
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

    // Get completion from OpenAI
    resp, err := qe.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
        Model: qe.model,
        Messages: []openai.ChatCompletionMessage{
            {Role: "system", Content: "You are a query understanding system for a tool search service."},
            {Role: "user", Content: prompt},
        },
        ResponseFormat: &openai.ChatCompletionResponseFormat{Type: "json_object"},
    })
    if err != nil {
        return nil, fmt.Errorf("failed to get query understanding: %w", err)
    }

    // Parse response
    var understanding models.QueryUnderstanding
    if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &understanding); err != nil {
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