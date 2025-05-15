package models

import (
	"time"
)

// Tool represents a searchable tool in the system
type Tool struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Category    string                 `json:"category"`
    Tags        []string              `json:"tags"`
    InputSchema map[string]interface{} `json:"input_schema"`
    OutputSchema map[string]interface{} `json:"output_schema"`
    Version     string                 `json:"version"`
    CreatedAt   time.Time             `json:"created_at"`
    UpdatedAt   time.Time             `json:"updated_at"`
}

// SearchResult represents a single search result
type SearchResult struct {
    Tool            Tool    `json:"tool"`
    Score           float64 `json:"score"`
    VectorScore     float64 `json:"vector_score"`
    KeywordScore    float64 `json:"keyword_score"`
    RerankedScore   float64 `json:"reranked_score"`
    Confidence      float64 `json:"confidence"`
    Justification   string  `json:"justification,omitempty"`
}

// SearchRequest represents a search query
type SearchRequest struct {
    Query     string                 `json:"query"`
    Filters   map[string]interface{} `json:"filters,omitempty"`
    TopK      int                    `json:"top_k"`
    MinScore  float64               `json:"min_score,omitempty"`
}

// SearchResponse represents the search results
type SearchResponse struct {
    Results []SearchResult `json:"results"`
    Total   int           `json:"total"`
    Time    float64       `json:"time_ms"`
}

// QueryUnderstanding represents the analyzed query
type QueryUnderstanding struct {
    Intent        string   `json:"intent"`
    ExpandedTerms []string `json:"expanded_terms"`
    SubQueries    []string `json:"sub_queries"`
    Filters       map[string]interface{} `json:"filters"`
} 