package config

import (
	"encoding/json"
	"os"
)

type Config struct {
    OpenAI struct {
        APIKey string `json:"api_key"`
        Model  string `json:"model"`
    } `json:"openai"`
    PostgreSQL struct {
        DSN string `json:"dsn"`
    } `json:"postgresql"`
    OpenSearch struct {
        URL      string `json:"url"`
        Username string `json:"username"`
        Password string `json:"password"`
    } `json:"opensearch"`
    Server struct {
        Port int    `json:"port"`
        Host string `json:"host"`
    } `json:"server"`
}

func LoadConfig(path string) (*Config, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var config Config
    decoder := json.NewDecoder(file)
    if err := decoder.Decode(&config); err != nil {
        return nil, err
    }

    return &config, nil
} 