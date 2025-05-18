package query

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"smartsearch/internal/ollama"
	"smartsearch/pkg/models"
	"strings"
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
	// Get completion from Ollama
	messages := []ollama.ChatMessage{
		{
			Role: "system",
			Content: "You are a query understanding system for a tool search service. Always respond with valid JSON only. Rules for JSON response: 1. The response must be wrapped in ```json and ``` markers 2. No extra text, explanations, or thinking steps outside the JSON 3. No comments inside the JSON 4. No trailing commas 5. All strings must be double-quoted 6. No single quotes allowed 7. No extra whitespace or newlines inside the JSON Example response: ```json {\"query\": \"find all files modified in the last week\", \"intent\": \"search\", \"parameters\": {\"keywords\": [\"files\", \"modified\", \"last\", \"week\"], \"filters\": {\"modified_at\": \"last_week\"}}, \"confidence\": 0.9} ```",
		},
		{Role: "user", Content: query},
	}

	response, err := qe.client.GetChatCompletion(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("failed to get query understanding: %w", err)
	}

	// Debug raw response
	fmt.Printf("Raw Ollama Response: %q\n", response)

	// Clean up the response by removing thinking steps and extracting JSON
	cleanedResponse := cleanOllamaResponse(response)
	fmt.Printf("Cleaned Response: %q\n", cleanedResponse)

	// Parse response
	var understanding models.QueryUnderstanding
	if err := json.Unmarshal([]byte(cleanedResponse), &understanding); err != nil {
		return nil, fmt.Errorf("failed to parse query understanding (response: %q): %w", cleanedResponse, err)
	}

	return &understanding, nil
}
// cleanOllamaResponse removes thinking steps and extracts just the JSON content
func cleanOllamaResponse(response string) string {
	// Remove thinking steps (content between <think> tags)
	re := regexp.MustCompile(`<think>.*?</think>`)
	cleaned := re.ReplaceAllString(response, "")

	// Find the JSON content between ```json and ```
	jsonStart := strings.Index(cleaned, "```json")
	if jsonStart == -1 {
		// If no ```json marker, try to find just the JSON object
		jsonStart = strings.Index(cleaned, "{")
	} else {
		// Skip the ```json marker
		jsonStart += 7
	}

	jsonEnd := strings.LastIndex(cleaned, "}")
	if jsonEnd == -1 {
		fmt.Printf("No JSON found in response: %q\n", cleaned)
		return "{}"
	}

	// Include the closing brace
	jsonEnd++

	if jsonStart == -1 || jsonEnd == 0 {
		fmt.Printf("No JSON found in response: %q\n", cleaned)
		return "{}"
	}

	jsonContent := cleaned[jsonStart:jsonEnd]
	
	// Clean up any remaining whitespace
	jsonContent = strings.TrimSpace(jsonContent)
	
	// Validate that we have valid JSON
	var test map[string]interface{}
	if err := json.Unmarshal([]byte(jsonContent), &test); err != nil {
		fmt.Printf("Invalid JSON found: %q\n", jsonContent)
		return "{}"
	}

	fmt.Print(jsonContent)

	return jsonContent
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
