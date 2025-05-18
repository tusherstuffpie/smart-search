package ollama

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
)

type Client struct {
	client openai.Client
	model  string
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	Format   string        `json:"format,omitempty"`
}

type ChatResponse struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
}

type EmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

func NewClient(url, model string) *Client {
	client := openai.NewClient(
		option.WithBaseURL(url),
		option.WithAPIKey("ollama"),
	)
	return &Client{
		client: client,
		model:  model,
	}
}

func (c *Client) GetChatCompletion(ctx context.Context, messages []ChatMessage) (string, error) {
	// Convert our ChatMessage format to OpenAI's format
	openAIMessages := make([]openai.ChatCompletionMessageParamUnion, len(messages))
	for i, msg := range messages {
		openAIMessages[i] = openai.ChatCompletionMessageParamUnion{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String(msg.Content),
				},
			},
		}
	}

	// Create a channel to collect the streamed response
	responseChan := make(chan string)
	errorChan := make(chan error)

	// Start streaming in a goroutine
	go func() {
		stream := c.client.Chat.Completions.NewStreaming(
			ctx,
			openai.ChatCompletionNewParams{
				Messages: openAIMessages,
				Model:    shared.ChatModel(c.model),
			},
		)

		var fullResponse strings.Builder
		for stream.Next() {
			chunk := stream.Current()
			if len(chunk.Choices) > 0 {
				content := chunk.Choices[0].Delta.Content
				if content != "" {
					fullResponse.WriteString(content)
				}
			}
		}

		if err := stream.Err(); err != nil {
			errorChan <- fmt.Errorf("error during streaming: %w", err)
			return
		}

		responseChan <- fullResponse.String()
	}()

	// Wait for either the response or an error
	select {
	case response := <-responseChan:
		if response == "" {
			return "", fmt.Errorf("empty response from OpenAI")
		}
		return response, nil
	case err := <-errorChan:
		return "", err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (c *Client) GetEmbedding(ctx context.Context, text string) ([]float32, error) {
	resp, err := c.client.Embeddings.New(
		ctx,
		openai.EmbeddingNewParams{
			Model: openai.EmbeddingModel(c.model),
			Input: openai.EmbeddingNewParamsInputUnion{
				OfString: openai.String(text),
			},
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %w", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	// Convert []float64 to []float32
	embedding := make([]float32, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
} 