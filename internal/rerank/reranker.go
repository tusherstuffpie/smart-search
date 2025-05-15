package rerank

import (
	"context"
	"encoding/json"
	"fmt"
	"smartsearch/pkg/models"
	"sort"

	"github.com/sashabaranov/go-openai"
)

type Reranker struct {
	client *openai.Client
	model  string
}

func NewReranker(client *openai.Client, model string) *Reranker {
	return &Reranker{
		client: client,
		model:  model,
	}
}

func (r *Reranker) Rerank(ctx context.Context, query string, results []models.SearchResult) ([]models.SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}

	// Prepare candidates for re-ranking
	candidates := make([]map[string]interface{}, len(results))
	for i, result := range results {
		candidates[i] = map[string]interface{}{
			"id":          result.Tool.ID,
			"name":        result.Tool.Name,
			"description": result.Tool.Description,
			"category":    result.Tool.Category,
			"tags":        result.Tool.Tags,
			"score":       result.Score,
		}
	}

	// Prepare prompt for re-ranking
	prompt := fmt.Sprintf(`Re-rank these search results for the query: "%s"

Results to re-rank:
%s

Provide a JSON response with the following structure:
{
    "rankings": [
        {
            "id": "tool_id",
            "score": 0.95,
            "justification": "Brief explanation of why this result is relevant"
        }
    ]
}`, query, mustMarshal(candidates))

	// Get completion from OpenAI
	resp, err := r.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: r.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: "You are a search result re-ranking system. Analyze the relevance of each result to the query and provide a new ranking with scores and justifications."},
			{Role: "user", Content: prompt},
		},
		ResponseFormat: &openai.ChatCompletionResponseFormat{Type: "json_object"},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get re-ranking: %w", err)
	}

	// Parse response
	var rankingResponse struct {
		Rankings []struct {
			ID           string  `json:"id"`
			Score        float64 `json:"score"`
			Justification string `json:"justification"`
		} `json:"rankings"`
	}

	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &rankingResponse); err != nil {
		return nil, fmt.Errorf("failed to parse re-ranking response: %w", err)
	}

	// Create map of new rankings
	newRankings := make(map[string]struct {
		Score        float64
		Justification string
	})
	for _, ranking := range rankingResponse.Rankings {
		newRankings[ranking.ID] = struct {
			Score        float64
			Justification string
		}{
			Score:        ranking.Score,
			Justification: ranking.Justification,
		}
	}

	// Update results with new rankings
	for i := range results {
		if newRanking, ok := newRankings[results[i].Tool.ID]; ok {
			results[i].RerankedScore = newRanking.Score
			results[i].Justification = newRanking.Justification
			results[i].Score = newRanking.Score // Update final score with re-ranked score
		}
	}

	// Sort results by new score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

func mustMarshal(v interface{}) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		panic(err)
	}
	return string(b)
} 