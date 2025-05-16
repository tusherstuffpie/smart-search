package config

import (
	"encoding/json"
	"os"
)

type Config struct {
	PostgreSQL struct {
		DSN string `json:"dsn"`
	} `json:"postgresql"`
	OpenSearch struct {
		URL      string `json:"url"`
		Username string `json:"username"`
		Password string `json:"password"`
	} `json:"opensearch"`
	Ollama struct {
		URL        string `json:"url"`
		EmbedModel string `json:"embed_model"`
		ChatModel  string `json:"chat_model"`
		Options    struct {
			Dimensionality int `json:"dimensionality"`
		} `json:"options"`
	} `json:"ollama"`
	Server struct {
		Host string `json:"host"`
		Port int    `json:"port"`
	} `json:"server"`
}

func LoadConfig(path string) (*Config, error) {
	file, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	if err := json.Unmarshal(file, &config); err != nil {
		return nil, err
	}

	return &config, nil
} 