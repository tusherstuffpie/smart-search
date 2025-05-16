package search

import (
	"context"
	"fmt"
	"smartsearch/internal/query"
	"smartsearch/internal/vector"
	"smartsearch/pkg/models"
	"time"
)

type Service struct {
	queryEngine  *query.QueryEngine
	vectorStore  *vector.VectorStore
	searchClient *OpenSearchClient
}

func NewService(
	queryEngine *query.QueryEngine,
	vectorStore *vector.VectorStore,
	searchClient *OpenSearchClient,
) *Service {
	return &Service{
		queryEngine:  queryEngine,
		vectorStore:  vectorStore,
		searchClient: searchClient,
	}
}

func (s *Service) Search(ctx context.Context, req models.SearchRequest) (*models.SearchResponse, error) {
	startTime := time.Now()

	// 1. Query Understanding
	understanding, err := s.queryEngine.UnderstandQuery(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("failed to understand query: %w", err)
	}

	// 2. Expand query
	expandedQueries, err := s.queryEngine.ExpandQuery(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("failed to expand query: %w", err)
	}

	// 3. Parallel search using both vector and keyword search
	var allResults []models.SearchResult
	for _, expandedQuery := range expandedQueries {
		vectorResults, err := s.vectorStore.Search(ctx, expandedQuery, req.TopK*2)
		if err != nil {
			return nil, fmt.Errorf("failed to perform vector search: %w", err)
		}
		allResults = append(allResults, vectorResults...)

		keywordResults, err := s.searchClient.Search(ctx, expandedQuery, req.TopK*2)
		if err != nil {
			return nil, fmt.Errorf("failed to perform keyword search: %w", err)
		}
		allResults = append(allResults, keywordResults...)
	}

	// 4. Merge and deduplicate results
	mergedResults := s.mergeResults(allResults[:len(allResults)/2], allResults[len(allResults)/2:])

	// 5. Apply filters from query understanding
	filteredResults := s.applyFilters(mergedResults, understanding.Filters)

	// 6. Apply minimum score threshold
	finalResults := s.applyScoreThreshold(filteredResults, req.MinScore)

	// 7. Limit to top K results
	if len(finalResults) > req.TopK {
		finalResults = finalResults[:req.TopK]
	}

	return &models.SearchResponse{
		Results: finalResults,
		Total:   len(finalResults),
		Time:    float64(time.Since(startTime).Milliseconds()),
	}, nil
}

func (s *Service) mergeResults(vectorResults, keywordResults []models.SearchResult) []models.SearchResult {
	// Create map to track seen tools
	seen := make(map[string]models.SearchResult)
	
	// Process vector results first
	for _, result := range vectorResults {
		seen[result.Tool.ID] = result
	}
	
	// Merge keyword results
	for _, result := range keywordResults {
		if existing, ok := seen[result.Tool.ID]; ok {
			// Update scores if tool was found in both searches
			existing.KeywordScore = result.KeywordScore
			existing.Score = (existing.VectorScore + result.KeywordScore) / 2
			seen[result.Tool.ID] = existing
		} else {
			seen[result.Tool.ID] = result
		}
	}
	
	// Convert map to slice
	results := make([]models.SearchResult, 0, len(seen))
	for _, result := range seen {
		results = append(results, result)
	}
	
	return results
}

func (s *Service) applyFilters(results []models.SearchResult, filters map[string]interface{}) []models.SearchResult {
	if len(filters) == 0 {
		return results
	}

	filtered := make([]models.SearchResult, 0, len(results))
	for _, result := range results {
		if s.matchesFilters(result, filters) {
			filtered = append(filtered, result)
		}
	}
	return filtered
}

func (s *Service) matchesFilters(result models.SearchResult, filters map[string]interface{}) bool {
	for key, value := range filters {
		switch key {
		case "category":
			if cat, ok := value.(string); ok && result.Tool.Category != cat {
				return false
			}
		case "tags":
			if tags, ok := value.([]string); ok {
				found := false
				for _, tag := range tags {
					for _, resultTag := range result.Tool.Tags {
						if tag == resultTag {
							found = true
							break
						}
					}
				}
				if !found {
					return false
				}
			}
		}
	}
	return true
}

func (s *Service) applyScoreThreshold(results []models.SearchResult, minScore float64) []models.SearchResult {
	if minScore <= 0 {
		return results
	}

	filtered := make([]models.SearchResult, 0, len(results))
	for _, result := range results {
		if result.Score >= minScore {
			filtered = append(filtered, result)
		}
	}
	return filtered
} 