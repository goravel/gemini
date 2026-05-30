package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime"
	"path/filepath"
	"reflect"
	"strings"

	frameworkai "github.com/goravel/framework/ai"
	contractsai "github.com/goravel/framework/contracts/ai"
	contractsconfig "github.com/goravel/framework/contracts/config"
	"github.com/goravel/framework/errors"
	"google.golang.org/genai"
)

const DefaultTextModel = "gemini-2.5-flash"
const DefaultImageModel = "imagen-4.0-generate-001"

const fileIDSeparator = "|"

type Provider struct {
	client *genai.Client
	config contractsai.ProviderConfig
}

func NewGemini(config contractsconfig.Config, provider string) (*Provider, error) {
	var providerConfig contractsai.ProviderConfig
	if err := config.UnmarshalKey("ai.providers."+provider, &providerConfig); err != nil {
		return nil, err
	}

	if providerConfig.Models.Text.Default == "" {
		providerConfig.Models.Text.Default = DefaultTextModel
	}
	if providerConfig.Models.Image.Default == "" {
		providerConfig.Models.Image.Default = DefaultImageModel
	}

	clientConfig := &genai.ClientConfig{
		APIKey:  providerConfig.Key,
		Backend: genai.BackendGeminiAPI,
	}
	if providerConfig.Url != "" {
		clientConfig.HTTPOptions.BaseURL = providerConfig.Url
	}

	client, err := genai.NewClient(context.Background(), clientConfig)
	if err != nil {
		return nil, err
	}

	return &Provider{client: client, config: providerConfig}, nil
}

func (r *Provider) Prompt(ctx context.Context, prompt contractsai.AgentPrompt) (contractsai.AgentResponse, error) {
	contents, config, err := r.buildGenerateContentRequest(ctx, prompt, false)
	if err != nil {
		return nil, err
	}

	response, err := r.client.Models.GenerateContent(ctx, r.resolveModel(prompt.Model), contents, config)
	if err != nil {
		return nil, err
	}

	text, toolCalls := r.parseGenerateContentResponse(response)
	return frameworkai.NewTextResponse(text, r.parseUsage(response), toolCalls), nil
}

func (r *Provider) Stream(ctx context.Context, prompt contractsai.AgentPrompt) (contractsai.StreamableAgentResponse, error) {
	contents, config, err := r.buildGenerateContentRequest(ctx, prompt, false)
	if err != nil {
		return nil, err
	}

	return frameworkai.NewStreamableResponse(ctx, func(streamCtx context.Context, emit func(contractsai.StreamEvent) error) (contractsai.AgentResponse, error) {
		text := strings.Builder{}
		var finalToolCalls []contractsai.ToolCall
		var finalUsage contractsai.Usage = frameworkai.NewUsage(0, 0, 0)

		for chunk, streamErr := range r.client.Models.GenerateContentStream(streamCtx, r.resolveModel(prompt.Model), contents, config) {
			if streamErr != nil {
				if streamCtx.Err() == nil {
					if emitErr := emit(contractsai.StreamEvent{
						Type:  contractsai.StreamEventTypeError,
						Error: streamErr.Error(),
					}); emitErr != nil {
						return nil, emitErr
					}
				}

				return nil, streamErr
			}

			delta := chunk.Text()
			if delta != "" {
				text.WriteString(delta)
				if err := emit(contractsai.StreamEvent{
					Type:  contractsai.StreamEventTypeTextDelta,
					Delta: delta,
				}); err != nil {
					return nil, err
				}
			}

			toolCalls := r.parseFunctionCalls(chunk.FunctionCalls())
			if len(toolCalls) > 0 {
				finalToolCalls = toolCalls
				if err := emit(contractsai.StreamEvent{
					Type:      contractsai.StreamEventTypeToolCall,
					ToolCalls: toolCalls,
				}); err != nil {
					return nil, err
				}
			}

			finalUsage = r.parseUsage(chunk)
		}

		if err := emit(contractsai.StreamEvent{
			Type:  contractsai.StreamEventTypeDone,
			Usage: finalUsage,
		}); err != nil {
			return nil, err
		}

		return frameworkai.NewTextResponse(text.String(), finalUsage, finalToolCalls), nil
	}), nil
}

func (r *Provider) PutFile(ctx context.Context, file contractsai.StorableFile) (contractsai.FileResponse, error) {
	content, err := file.Content(ctx)
	if err != nil {
		return nil, err
	}

	mimeType := file.MimeType()
	if mimeType == "" {
		mimeType = fallbackMimeType(file.FileName())
	}

	uploaded, err := r.client.Files.Upload(ctx, bytes.NewReader(content), &genai.UploadFileConfig{
		DisplayName: file.FileName(),
		MIMEType:    mimeType,
	})
	if err != nil {
		return nil, err
	}

	return frameworkai.NewFileResponse(encodeFileID(uploaded), uploaded.MIMEType, nil), nil
}

func (r *Provider) GetFile(ctx context.Context, id string) (contractsai.FileResponse, error) {
	name, _, _, err := decodeFileID(id)
	if err != nil {
		name = id
	}

	file, err := r.client.Files.Get(ctx, name, nil)
	if err != nil {
		return nil, err
	}

	content, err := r.client.Files.Download(ctx, genai.NewDownloadURIFromFile(file), nil)
	if err != nil {
		return nil, err
	}

	return frameworkai.NewFileResponse(encodeFileID(file), file.MIMEType, content), nil
}

func (r *Provider) DeleteFile(ctx context.Context, id string) error {
	name, _, _, err := decodeFileID(id)
	if err != nil {
		name = id
	}

	_, err = r.client.Files.Delete(ctx, name, nil)
	return err
}

func (r *Provider) Image(ctx context.Context, prompt contractsai.ImagePrompt) (contractsai.ImageResponse, error) {
	if prompt.Prompt == "" {
		return nil, errors.AIImagePromptRequired
	}
	for _, attachment := range prompt.Attachments {
		if attachment.Kind() != contractsai.AttachmentKindImage {
			return nil, errors.AIImageAttachmentRequired
		}
	}

	if len(prompt.Attachments) == 0 {
		response, err := r.client.Models.GenerateImages(ctx, r.resolveImageModel(prompt.Model), prompt.Prompt, &genai.GenerateImagesConfig{
			AspectRatio: r.resolveAspectRatio(prompt.Size),
		})
		if err != nil {
			return nil, err
		}

		return r.parseGeneratedImages(response.GeneratedImages)
	}

	return nil, fmt.Errorf("gemini provider does not support attachment-based image editing on Gemini API")
}

func (r *Provider) buildGenerateContentRequest(ctx context.Context, prompt contractsai.AgentPrompt, imageOutput bool) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	contents := make([]*genai.Content, 0)
	history := prompt.Agent.Messages()
	toolCallNames := make(map[string]string)
	attachmentIndex := -1
	if prompt.Input == "" && len(prompt.Attachments) > 0 {
		for i := len(history) - 1; i >= 0; i-- {
			if history[i].Role == contractsai.RoleUser {
				attachmentIndex = i
				break
			}
		}
	}

	for i, message := range history {
		attachments := []contractsai.Attachment(nil)
		if i == attachmentIndex {
			attachments = prompt.Attachments
		}

		content, err := r.buildMessageContent(ctx, message, attachments, toolCallNames)
		if err != nil {
			return nil, nil, err
		}
		if content != nil {
			contents = append(contents, content)
		}
	}

	if prompt.Input != "" || (len(prompt.Attachments) > 0 && attachmentIndex == -1) {
		content, err := r.buildMessageContent(ctx, contractsai.Message{
			Role:    contractsai.RoleUser,
			Content: prompt.Input,
		}, prompt.Attachments, toolCallNames)
		if err != nil {
			return nil, nil, err
		}
		if content != nil {
			contents = append(contents, content)
		}
	}

	config := &genai.GenerateContentConfig{}
	if instructions := prompt.Agent.Instructions(); instructions != "" {
		config.SystemInstruction = &genai.Content{Parts: []*genai.Part{{Text: instructions}}}
	}
	if len(prompt.Tools) > 0 {
		config.Tools = []*genai.Tool{{FunctionDeclarations: r.buildTools(prompt.Tools)}}
	}
	if imageOutput {
		config.ResponseModalities = []string{string(genai.ModalityImage), string(genai.ModalityText)}
		config.ImageConfig = &genai.ImageConfig{AspectRatio: r.resolveAspectRatio(contractsai.ImageSizeSquare)}
	}

	return contents, config, nil
}

func (r *Provider) buildMessageContent(ctx context.Context, message contractsai.Message, attachments []contractsai.Attachment, toolCallNames map[string]string) (*genai.Content, error) {
	parts := make([]*genai.Part, 0)

	switch message.Role {
	case contractsai.RoleUser:
		if message.Content != "" {
			parts = append(parts, genai.NewPartFromText(message.Content))
		}
		for _, attachment := range attachments {
			part, err := r.buildAttachmentPart(ctx, attachment)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		if len(parts) == 0 {
			return nil, nil
		}
		return genai.NewContentFromParts(parts, genai.RoleUser), nil
	case contractsai.RoleAssistant:
		if message.Content != "" {
			parts = append(parts, genai.NewPartFromText(message.Content))
		}
		for _, toolCall := range message.ToolCalls {
			args := cloneMap(toolCall.Args)
			if args == nil && toolCall.RawArgs != "" {
				_ = json.Unmarshal([]byte(toolCall.RawArgs), &args)
			}
			if toolCall.ID != "" {
				toolCallNames[toolCall.ID] = toolCall.Name
			}
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   toolCall.ID,
				Name: toolCall.Name,
				Args: args,
			}})
		}
		if len(parts) == 0 {
			return nil, nil
		}
		return genai.NewContentFromParts(parts, genai.RoleModel), nil
	case contractsai.RoleToolResult:
		response := map[string]any{"output": message.Content}
		toolName := toolCallNames[message.ToolCallID]
		if toolName == "" {
			toolName = message.ToolCallID
		}
		parts = append(parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID:       message.ToolCallID,
			Name:     toolName,
			Response: response,
		}})
		return genai.NewContentFromParts(parts, genai.RoleUser), nil
	default:
		return nil, nil
	}
}

func (r *Provider) buildAttachmentPart(ctx context.Context, attachment contractsai.Attachment) (*genai.Part, error) {
	if stored, ok := attachment.(contractsai.ProviderFile); ok && stored.ID() != "" {
		_, uri, mimeType, err := decodeFileID(stored.ID())
		if err == nil && uri != "" && mimeType != "" {
			return genai.NewPartFromURI(uri, mimeType), nil
		}
	}

	content, err := attachment.Content(ctx)
	if err != nil {
		return nil, err
	}

	mimeType := attachment.MimeType()
	if mimeType == "" {
		mimeType = fallbackMimeType(attachment.FileName())
	}

	return genai.NewPartFromBytes(content, mimeType), nil
}

func (r *Provider) buildTools(tools []contractsai.Tool) []*genai.FunctionDeclaration {
	declarations := make([]*genai.FunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		declaration := &genai.FunctionDeclaration{
			Name:        tool.Name(),
			Description: tool.Description(),
		}
		if parameters := tool.Parameters(); parameters != nil {
			declaration.ParametersJsonSchema = parameters
		}
		declarations = append(declarations, declaration)
	}

	return declarations
}

func (r *Provider) parseGenerateContentResponse(response *genai.GenerateContentResponse) (string, []contractsai.ToolCall) {
	if response == nil {
		return "", nil
	}

	return response.Text(), r.parseFunctionCalls(response.FunctionCalls())
}

func (r *Provider) parseFunctionCalls(calls []*genai.FunctionCall) []contractsai.ToolCall {
	if len(calls) == 0 {
		return nil
	}

	toolCalls := make([]contractsai.ToolCall, 0, len(calls))
	for _, call := range calls {
		if call == nil {
			continue
		}

		rawArgs := "{}"
		if len(call.Args) > 0 {
			if encoded, err := json.Marshal(call.Args); err == nil {
				rawArgs = string(encoded)
			}
		}

		toolCalls = append(toolCalls, contractsai.ToolCall{
			ID:      call.ID,
			Name:    call.Name,
			Args:    cloneMap(call.Args),
			RawArgs: rawArgs,
		})
	}

	if len(toolCalls) == 0 {
		return nil
	}

	return toolCalls
}

func (r *Provider) parseUsage(response *genai.GenerateContentResponse) contractsai.Usage {
	if response == nil || response.UsageMetadata == nil {
		return frameworkai.NewUsage(0, 0, 0)
	}

	return frameworkai.NewUsage(
		int(response.UsageMetadata.PromptTokenCount),
		int(response.UsageMetadata.CandidatesTokenCount),
		int(response.UsageMetadata.TotalTokenCount),
	)
}

func (r *Provider) parseGeneratedImages(images []*genai.GeneratedImage) (contractsai.ImageResponse, error) {
	if len(images) == 0 || images[0] == nil || images[0].Image == nil || len(images[0].Image.ImageBytes) == 0 {
		return nil, errors.AIImageResponseIsEmpty
	}

	mimeType := images[0].Image.MIMEType
	if mimeType == "" {
		mimeType = "image/png"
	}

	return frameworkai.NewImageResponse(images[0].Image.ImageBytes, mimeType, frameworkai.NewUsage(0, 0, 0)), nil
}

func (r *Provider) resolveModel(model string) string {
	if model != "" {
		return model
	}

	return r.config.Models.Text.Default
}

func (r *Provider) resolveImageModel(model string) string {
	if model != "" {
		return model
	}

	return r.config.Models.Image.Default
}

func (r *Provider) resolveAspectRatio(size contractsai.ImageSize) string {
	switch size {
	case contractsai.ImageSizePortrait:
		return "3:4"
	case contractsai.ImageSizeLandscape:
		return "4:3"
	default:
		return "1:1"
	}
}

func encodeFileID(file *genai.File) string {
	if file == nil {
		return ""
	}

	return strings.Join([]string{file.Name, file.URI, file.MIMEType}, fileIDSeparator)
}

func decodeFileID(id string) (name, uri, mimeType string, err error) {
	parts := strings.SplitN(id, fileIDSeparator, 3)
	if len(parts) != 3 {
		return "", "", "", fmt.Errorf("invalid gemini file id")
	}

	return parts[0], parts[1], parts[2], nil
}

func fallbackMimeType(fileName string) string {
	mimeType := mime.TypeByExtension(filepath.Ext(fileName))
	if mimeType == "" {
		return "application/octet-stream"
	}

	mediaType, _, err := mime.ParseMediaType(mimeType)
	if err != nil || mediaType == "" {
		return mimeType
	}

	return mediaType
}

func cloneMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return nil
	}

	cloned := make(map[string]any, len(input))
	for key, value := range input {
		cloned[key] = cloneValue(value)
	}

	return cloned
}

func cloneValue(value any) any {
	if value == nil {
		return nil
	}

	reflectValue := reflect.ValueOf(value)
	switch reflectValue.Kind() {
	case reflect.Map:
		if typed, ok := value.(map[string]any); ok {
			return cloneMap(typed)
		}
	case reflect.Slice:
		if typed, ok := value.([]any); ok {
			out := make([]any, len(typed))
			for i, item := range typed {
				out[i] = cloneValue(item)
			}
			return out
		}
	}

	return value
}
