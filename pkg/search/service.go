package search

import (
	"context"
	"fmt"
	"log"
	"smartsearch/internal/query"
	"smartsearch/internal/vector"
	"smartsearch/pkg/models"
	"sort"
	"time"

	"golang.org/x/sync/errgroup"
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

	// Verify OpenSearch index settings
	if err := s.searchClient.VerifyIndexSettings(ctx); err != nil {
		log.Printf("Warning: OpenSearch index verification failed: %v", err)
	}

	// Calculate k value
	k := req.TopK * 2
	if k < 1 {
		k = 10
	}

	// Create error group for parallel execution
	g, ctx := errgroup.WithContext(ctx)
	var vectorResults, keywordResults []models.SearchResult

	// Run vector search
	g.Go(func() error {
		results, err := s.vectorStore.Search(ctx, req.Query, k)
		if err != nil {
			return fmt.Errorf("vector search failed: %w", err)
		}
		vectorResults = results
		return nil
	})

	// Run keyword search
	g.Go(func() error {
		results, err := s.searchClient.Search(ctx, req.Query, k)
		if err != nil {
			return fmt.Errorf("keyword search failed: %w", err)
		}
		keywordResults = results
		return nil
	})

	// Wait for both searches to complete
	if err := g.Wait(); err != nil {
		return nil, err
	}

	// Merge and deduplicate results
	mergedResults := s.mergeResults(vectorResults, keywordResults)
	log.Printf("Merged results count: %d", len(mergedResults))

	// Apply minimum score threshold
	minScore := req.MinScore
	if minScore <= 0 {
		minScore = 0.3 // Default minimum score threshold
	}
	finalResults := s.applyScoreThreshold(mergedResults, minScore)
	log.Printf("Results after score threshold: %d", len(finalResults))

	// Limit to top K results
	if len(finalResults) > k {
		finalResults = finalResults[:k]
	}

	// Create simplified results with only required fields
	simplifiedResults := make([]models.SearchResult, len(finalResults))
	for i, result := range finalResults {
		simplifiedResults[i] = models.SearchResult{
			Tool: models.Tool{
				ID:          result.Tool.ID,
				Name:        result.Tool.Name,
				Description: result.Tool.Description,
			},
			Score:         result.Score,
			VectorScore:   result.VectorScore,
			KeywordScore:  result.KeywordScore,
			RerankedScore: result.RerankedScore,
		}
	}

	return &models.SearchResponse{
		Results: simplifiedResults,
		Total:   len(simplifiedResults),
		Time:    float64(time.Since(startTime).Milliseconds()),
	}, nil
}

func (s *Service) mergeResults(vectorResults, keywordResults []models.SearchResult) []models.SearchResult {
	// Create map to track seen tools
	seen := make(map[string]models.SearchResult)
	
	// Process vector results first
	for _, result := range vectorResults {
		result.VectorScore = result.Score // Store vector score separately
		seen[result.Tool.ID] = result
	}
	
	// Merge keyword results
	for _, result := range keywordResults {
		if existing, ok := seen[result.Tool.ID]; ok {
			// Update scores if tool was found in both searches
			existing.KeywordScore = result.Score
			// Combine scores with 70% weight to vector and 30% to keyword
			existing.Score = (existing.VectorScore * 0.7) + (result.Score * 0.3)
			seen[result.Tool.ID] = existing
		} else {
			result.KeywordScore = result.Score
			result.VectorScore = 0
			seen[result.Tool.ID] = result
		}
	}
	
	// Convert map to slice and sort by score
	results := make([]models.SearchResult, 0, len(seen))
	for _, result := range seen {
		results = append(results, result)
	}
	
	// Sort results by score in descending order
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	
	return results
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