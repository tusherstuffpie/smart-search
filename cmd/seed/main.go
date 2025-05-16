package main

import (
	"context"
	"database/sql"
	"log"
	"smartsearch/pkg/config"
	"smartsearch/seed"

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

	// Run seeding
	if err := seed.SeedDatabase(context.Background(), db, opensearchClient, cfg); err != nil {
		log.Fatalf("Failed to seed database: %v", err)
	}

	log.Println("Database seeding completed successfully")
} 