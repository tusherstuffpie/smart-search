package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"smartsearch/internal/query"
	"smartsearch/internal/vector"
	"smartsearch/pkg/config"
	"smartsearch/pkg/models"
	"smartsearch/pkg/search"

	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq"
	"github.com/opensearch-project/opensearch-go"
)

func main() {
	// Load configuration
	cfg, err := config.LoadConfig("config.json")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize PostgreSQL connection
	db, err := sql.Open("postgres", cfg.PostgreSQL.DSN)
	if err != nil {
		log.Fatalf("Failed to connect to PostgreSQL: %v", err)
	}
	defer db.Close()

	// Initialize OpenSearch client
	opensearchConfig := opensearch.Config{
		Addresses: []string{cfg.OpenSearch.URL},
		Username:  cfg.OpenSearch.Username,
		Password:  cfg.OpenSearch.Password,
	}
	opensearchClient, err := opensearch.NewClient(opensearchConfig)
	if err != nil {
		log.Fatalf("Failed to create OpenSearch client: %v", err)
	}

	// Initialize search service
	service := search.NewService(
		query.NewQueryEngine(cfg.Ollama.URL, cfg.Ollama.ChatModel),
		vector.NewVectorStore(db, cfg.Ollama.URL, cfg.Ollama.EmbedModel, cfg),
		search.NewOpenSearchClient(opensearchClient, "tools"),
	)

	// Initialize Gin router
	router := gin.Default()

	// Register routes
	router.POST("/search", func(c *gin.Context) {
		var req models.SearchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		results, err := service.Search(c.Request.Context(), req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, results)
	})

	// Create HTTP server
	srv := &http.Server{
		Addr:    fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port),
		Handler: router,
	}

	// Start server in a goroutine
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exiting")
} 