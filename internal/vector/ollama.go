package vector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
)

type OllamaClient struct {
	url   string
	model string
}

type EmbeddingRequest struct {
	Model  string `json:"model"`
	Input  string `json:"input"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func NewOllamaClient(url, model string) *OllamaClient {
	return &OllamaClient{
		url:   url,
		model: model,
	}
}

func (c *OllamaClient) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	reqBody := EmbeddingRequest{
		Model: c.model,
		Input: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}


	req, err := http.NewRequestWithContext(ctx, "POST", c.url+"/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama API error: %s - %s", resp.Status, string(body))
	}

	var embeddingResp EmbeddingResponse
	if err := json.Unmarshal(body, &embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(embeddingResp.Data) == 0 {
		return nil, fmt.Errorf("empty embedding data in response")
	}

	embedding := embeddingResp.Data[0].Embedding
	if len(embedding) == 0 {
		return nil, fmt.Errorf("empty embedding vector in response")
	}

	log.Printf("Received embedding with %d dimensions", len(embedding))
	return embedding, nil
} 