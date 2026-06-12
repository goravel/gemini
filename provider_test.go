package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"
	"time"

	frameworkai "github.com/goravel/framework/ai"
	contractsai "github.com/goravel/framework/contracts/ai"
	frameworkerrors "github.com/goravel/framework/errors"
	mocksconfig "github.com/goravel/framework/mocks/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

type stubAgent struct {
	instructions string
	messages     []contractsai.Message
}

func (s stubAgent) Instructions() string { return s.instructions }

func (s stubAgent) Messages() []contractsai.Message {
	if len(s.messages) == 0 {
		return nil
	}

	return append([]contractsai.Message(nil), s.messages...)
}

func (s stubAgent) Middleware() []contractsai.Middleware { return nil }

func (s stubAgent) Tools() []contractsai.Tool { return nil }

func TestNewGeminiFailoverRules(t *testing.T) {
	t.Run("compiles configured failover rules", func(t *testing.T) {
		mockConfig := mocksconfig.NewConfig(t)
		mockConfig.EXPECT().UnmarshalKey("ai.providers.gemini", new(contractsai.ProviderConfig)).RunAndReturn(func(_ string, rawVal any) error {
			cfg := rawVal.(*contractsai.ProviderConfig)
			cfg.Key = "test-key"
			cfg.Failover = map[contractsai.FailoverReason][]string{
				"context_length_exceeded": {"context length"},
			}
			return nil
		}).Once()

		provider, err := NewGemini(mockConfig, "gemini")

		require.NoError(t, err)
		require.NotNil(t, provider)
		require.NotNil(t, provider.failoverRules)
		assert.Equal(t, "gemini", provider.name)

		reason, ok := provider.failoverRules.Match(providerTestError("maximum context length exceeded"))
		assert.True(t, ok)
		assert.Equal(t, contractsai.FailoverReason("context_length_exceeded"), reason)
	})

	t.Run("returns invalid regex errors", func(t *testing.T) {
		mockConfig := mocksconfig.NewConfig(t)
		mockConfig.EXPECT().UnmarshalKey("ai.providers.gemini", new(contractsai.ProviderConfig)).RunAndReturn(func(_ string, rawVal any) error {
			cfg := rawVal.(*contractsai.ProviderConfig)
			cfg.Failover = map[contractsai.FailoverReason][]string{
				"bad_pattern": {"/[/"},
			}
			return nil
		}).Once()

		provider, err := NewGemini(mockConfig, "gemini")

		assert.Nil(t, provider)
		assert.ErrorIs(t, err, frameworkerrors.AIFailoverPatternInvalid)
	})
}

func TestProviderFailoverErrorWrapsDefaultStatusCodes(t *testing.T) {
	provider := &Provider{name: "gemini-primary"}

	tests := []struct {
		name       string
		statusCode int
		reason     contractsai.FailoverReason
	}{
		{
			name:       "rate limited",
			statusCode: http.StatusTooManyRequests,
			reason:     defaultFailoverReasonRateLimited,
		},
		{
			name:       "insufficient credits",
			statusCode: http.StatusPaymentRequired,
			reason:     defaultFailoverReasonInsufficientCredits,
		},
		{
			name:       "provider overloaded",
			statusCode: http.StatusServiceUnavailable,
			reason:     defaultFailoverReasonProviderOverloaded,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			apiErr := genai.APIError{Code: tt.statusCode, Message: "provider unavailable"}
			err := provider.failoverError(apiErr)

			var failoverErr contractsai.FailoverError
			require.ErrorAs(t, err, &failoverErr)
			assert.Equal(t, tt.reason, failoverErr.Reason())
			assert.Equal(t, "gemini-primary", failoverErr.Provider())

			var wrappedErr genai.APIError
			require.ErrorAs(t, err, &wrappedErr)
			assert.Equal(t, apiErr.Code, wrappedErr.Code)
			assert.Equal(t, apiErr.Message, wrappedErr.Message)
		})
	}
}

func TestProviderFailoverErrorReturnsOriginalError(t *testing.T) {
	provider := &Provider{name: "gemini-primary"}

	nonFailoverErr := genai.APIError{Code: http.StatusBadRequest, Message: "bad request"}
	err := provider.failoverError(nonFailoverErr)

	assert.Equal(t, nonFailoverErr.Error(), err.Error())
	var failoverErr contractsai.FailoverError
	assert.False(t, errors.As(err, &failoverErr))
	assert.Same(t, assert.AnError, provider.failoverError(assert.AnError))
}

func TestProviderFailoverErrorUsesConfiguredRules(t *testing.T) {
	customErr := providerTestError("maximum context length exceeded")
	rules, err := frameworkai.NewFailoverRules("gemini-primary", map[contractsai.FailoverReason][]string{
		"context_length_exceeded": {"context length"},
	})
	require.NoError(t, err)
	provider := &Provider{name: "gemini-primary", failoverRules: &rules}
	err = provider.failoverError(customErr)

	var failoverErr contractsai.FailoverError
	require.ErrorAs(t, err, &failoverErr)
	assert.Equal(t, contractsai.FailoverReason("context_length_exceeded"), failoverErr.Reason())
	assert.Equal(t, "gemini-primary", failoverErr.Provider())
	assert.ErrorIs(t, err, customErr)
}

func TestBuildGenerateContentRequestAttachesFollowUpAttachmentsToActiveUserTurn(t *testing.T) {
	provider := &Provider{}
	attachment := frameworkai.DocumentFromString("document", frameworkai.WithMimeType("text/plain"))

	contents, _, err := provider.buildGenerateContentRequest(context.Background(), contractsai.AgentPrompt{
		Agent: stubAgent{messages: []contractsai.Message{
			{Role: contractsai.RoleUser, Content: "question"},
			{Role: contractsai.RoleAssistant, ToolCalls: []contractsai.ToolCall{{ID: "call-1", Name: "lookup", RawArgs: `{"city":"London"}`}}},
			{Role: contractsai.RoleToolResult, Content: "result", ToolCallID: "call-1"},
		}},
		Attachments: []contractsai.Attachment{attachment},
	}, false)
	require.NoError(t, err)
	require.Len(t, contents, 3)

	assertContentRole(t, contents[0], string(genai.RoleUser))
	assertPartCount(t, contents[0], 2)
	assertTextPart(t, contents[0].Parts[0], "question")
	assertInlineDataPart(t, contents[0].Parts[1], "text/plain", []byte("document"))
}

func TestBuildGenerateContentRequestReplaysToolCallsAndResults(t *testing.T) {
	provider := &Provider{}

	contents, _, err := provider.buildGenerateContentRequest(context.Background(), contractsai.AgentPrompt{
		Agent: stubAgent{messages: []contractsai.Message{
			{Role: contractsai.RoleUser, Content: "question"},
			{Role: contractsai.RoleAssistant, ToolCalls: []contractsai.ToolCall{{ID: "call-1", Name: "lookup_weather", RawArgs: `{"city":"London"}`}}},
			{Role: contractsai.RoleToolResult, Content: "sunny", ToolCallID: "call-1"},
		}},
		Input: "thanks",
	}, false)
	require.NoError(t, err)
	require.Len(t, contents, 4)

	assertContentRole(t, contents[1], string(genai.RoleModel))
	assertPartCount(t, contents[1], 1)
	functionCall := contents[1].Parts[0].FunctionCall
	require.NotNil(t, functionCall)
	assert.Equal(t, "call-1", functionCall.ID)
	assert.Equal(t, "lookup_weather", functionCall.Name)
	assert.Equal(t, "London", functionCall.Args["city"])

	assertContentRole(t, contents[2], string(genai.RoleUser))
	assertPartCount(t, contents[2], 1)
	functionResponse := contents[2].Parts[0].FunctionResponse
	require.NotNil(t, functionResponse)
	assert.Equal(t, "call-1", functionResponse.ID)
	assert.Equal(t, "lookup_weather", functionResponse.Name)
	assert.Equal(t, "sunny", functionResponse.Response["output"])

	assertContentRole(t, contents[3], string(genai.RoleUser))
	assertPartCount(t, contents[3], 1)
	assertTextPart(t, contents[3].Parts[0], "thanks")
}

func TestBuildGenerateContentRequestReplaysEmptyToolCallIDByName(t *testing.T) {
	provider := &Provider{}

	contents, _, err := provider.buildGenerateContentRequest(context.Background(), contractsai.AgentPrompt{
		Agent: stubAgent{messages: []contractsai.Message{
			{Role: contractsai.RoleUser, Content: "question"},
			{Role: contractsai.RoleAssistant, ToolCalls: []contractsai.ToolCall{{Name: "get_weather", RawArgs: `{"city":"London"}`}}},
			{Role: contractsai.RoleToolResult, Content: "sunny"},
		}},
	}, false)
	require.NoError(t, err)
	require.Len(t, contents, 3)

	functionCall := contents[1].Parts[0].FunctionCall
	require.NotNil(t, functionCall)
	assert.Equal(t, "get_weather", functionCall.ID)
	assert.Equal(t, "get_weather", functionCall.Name)

	functionResponse := contents[2].Parts[0].FunctionResponse
	require.NotNil(t, functionResponse)
	assert.Equal(t, "get_weather", functionResponse.ID)
	assert.Equal(t, "get_weather", functionResponse.Name)
	assert.Equal(t, "sunny", functionResponse.Response["output"])
}

func TestBuildGenerateContentRequestReturnsErrorForInvalidToolCallArgs(t *testing.T) {
	provider := &Provider{}

	_, _, err := provider.buildGenerateContentRequest(context.Background(), contractsai.AgentPrompt{
		Agent: stubAgent{messages: []contractsai.Message{{
			Role:      contractsai.RoleAssistant,
			ToolCalls: []contractsai.ToolCall{{ID: "call-1", Name: "lookup_weather", RawArgs: `{"city":`}},
		}}},
	}, false)

	assert.EqualError(t, err, "invalid gemini tool call args for \"lookup_weather\": unexpected end of JSON input")
}

func TestBuildAttachmentPartUsesStoredGeminiFileURI(t *testing.T) {
	provider := &Provider{}
	file := &genai.File{
		Name:     "files/example-file",
		URI:      "https://example.invalid/files/example-file",
		MIMEType: "image/png",
	}

	part, err := provider.buildAttachmentPart(context.Background(), frameworkai.ImageFromID(encodeFileID(file)))
	require.NoError(t, err)
	require.NotNil(t, part)
	require.NotNil(t, part.FileData)
	assert.Equal(t, file.URI, part.FileData.FileURI)
	assert.Equal(t, file.MIMEType, part.FileData.MIMEType)
	assert.Nil(t, part.InlineData)
}

func TestEncodeDecodeFileIDRoundTrip(t *testing.T) {
	file := &genai.File{
		Name:     "files/sample",
		URI:      "https://example.invalid/files/sample",
		MIMEType: "application/pdf",
	}

	encoded := encodeFileID(file)
	name, uri, mimeType, err := decodeFileID(encoded)
	require.NoError(t, err)
	assert.Equal(t, file.Name, name)
	assert.Equal(t, file.URI, uri)
	assert.Equal(t, file.MIMEType, mimeType)
}

func TestParseAudioResponseReturnsFirstInlineAudioPart(t *testing.T) {
	provider := &Provider{}

	response, err := provider.parseAudioResponse(&genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{{
				InlineData: &genai.Blob{Data: []byte("audio-bytes"), MIMEType: "audio/wav"},
			}}},
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     2,
			CandidatesTokenCount: 3,
			TotalTokenCount:      5,
		},
	})
	require.NoError(t, err)

	content, err := response.Content()
	require.NoError(t, err)
	assert.Equal(t, []byte("audio-bytes"), content)
	assert.Equal(t, "audio/wav", response.MimeType())
	assert.Equal(t, 5, response.Usage().Total())
}

func TestResolveAudioVoiceMapsFrameworkDefaults(t *testing.T) {
	provider := &Provider{}

	assert.Equal(t, "Aoede", provider.resolveAudioVoice(""))
	assert.Equal(t, "Aoede", provider.resolveAudioVoice(frameworkai.DefaultFemaleVoice))
	assert.Equal(t, "Kore", provider.resolveAudioVoice(frameworkai.DefaultMaleVoice))
	assert.Equal(t, "CustomVoice", provider.resolveAudioVoice("CustomVoice"))
}

func TestTranscriptionPromptTextIncludesLanguageAndDiarizeHints(t *testing.T) {
	provider := &Provider{}
	text := provider.transcriptionPromptText(contractsai.TranscriptionPrompt{
		Language: "en",
		Diarize:  true,
	})

	assert.Equal(t, "Transcribe this audio exactly. The spoken language is en. If there are multiple speakers, label the speakers inline in the transcript.", text)
}

func TestApplyTimeoutSetsHTTPOptionsTimeout(t *testing.T) {
	timeout := 3 * time.Second
	contentConfig := &genai.GenerateContentConfig{}
	imageConfig := &genai.GenerateImagesConfig{}

	applyTimeout(contentConfig, timeout)
	applyTimeout(imageConfig, timeout)

	require.NotNil(t, contentConfig.HTTPOptions)
	require.NotNil(t, contentConfig.HTTPOptions.Timeout)
	require.NotNil(t, imageConfig.HTTPOptions)
	require.NotNil(t, imageConfig.HTTPOptions.Timeout)
	assert.Equal(t, timeout, *contentConfig.HTTPOptions.Timeout)
	assert.Equal(t, timeout, *imageConfig.HTTPOptions.Timeout)
}

func TestResolveAudioModelUsesAudioDefaultBeforeTextDefault(t *testing.T) {
	provider := &Provider{config: contractsai.ProviderConfig{}}
	provider.config.Models.Text.Default = "text-default"
	provider.config.Models.Audio.Default = "audio-default"

	assert.Equal(t, "audio-default", provider.resolveAudioModel(""))
}

func TestResolveTranscriptionModelUsesTranscriptionDefaultBeforeTextDefault(t *testing.T) {
	provider := &Provider{config: contractsai.ProviderConfig{}}
	provider.config.Models.Text.Default = "text-default"
	provider.config.Models.Transcription.Default = "transcription-default"

	assert.Equal(t, "transcription-default", provider.resolveTranscriptionModel(""))
}

func TestMergeToolCallsKeepsCallsAcrossStreamChunks(t *testing.T) {
	merged := mergeToolCalls([]contractsai.ToolCall{{ID: "call-1", Name: "first"}}, []contractsai.ToolCall{{ID: "call-2", Name: "second"}})

	require.Len(t, merged, 2)
	assert.Equal(t, "call-1", merged[0].ID)
	assert.Equal(t, "call-2", merged[1].ID)
}

func TestMergeToolCallsReplacesExistingCallWithSameID(t *testing.T) {
	merged := mergeToolCalls([]contractsai.ToolCall{{ID: "call-1", Name: "first", RawArgs: `{}`}}, []contractsai.ToolCall{{ID: "call-1", Name: "updated", RawArgs: `{"city":"London"}`}})

	require.Len(t, merged, 1)
	assert.Equal(t, "updated", merged[0].Name)
}

func TestParseToolCallArgsUsesExistingArgs(t *testing.T) {
	args, err := parseToolCallArgs(contractsai.ToolCall{
		Name: "lookup_weather",
		Args: map[string]any{"city": "London"},
	})
	require.NoError(t, err)
	assert.Equal(t, "London", args["city"])
}

func TestParseToolCallArgsReturnsErrorForInvalidJSON(t *testing.T) {
	_, err := parseToolCallArgs(contractsai.ToolCall{Name: "lookup_weather", RawArgs: `{"city":`})
	require.Error(t, err)
	var syntaxErr *json.SyntaxError
	assert.True(t, errors.As(err, &syntaxErr))
}

func TestParseFunctionCallsGeneratesIDWhenGeminiOmitsID(t *testing.T) {
	provider := &Provider{}

	toolCalls := provider.parseFunctionCalls([]*genai.FunctionCall{{
		Name: "get_weather",
		Args: map[string]any{"city": "London"},
	}})

	require.Len(t, toolCalls, 1)
	assert.Equal(t, "get_weather", toolCalls[0].ID)
	assert.Equal(t, "get_weather", toolCalls[0].Name)
	assert.Equal(t, map[string]any{"city": "London"}, toolCalls[0].Args)
}

func TestParseFunctionCallsDisambiguatesRepeatedGeneratedIDs(t *testing.T) {
	provider := &Provider{}

	toolCalls := provider.parseFunctionCalls([]*genai.FunctionCall{
		{Name: "get_weather", Args: map[string]any{"city": "London"}},
		{Name: "get_weather", Args: map[string]any{"city": "Paris"}},
	})

	require.Len(t, toolCalls, 2)
	assert.Equal(t, "get_weather", toolCalls[0].ID)
	assert.Equal(t, "get_weather_2", toolCalls[1].ID)
}

type providerTestError string

func (e providerTestError) Error() string {
	return string(e)
}

func assertContentRole(t *testing.T, content *genai.Content, expected string) {
	t.Helper()
	require.NotNil(t, content)
	assert.Equal(t, expected, content.Role)
}

func assertPartCount(t *testing.T, content *genai.Content, expected int) {
	t.Helper()
	require.NotNil(t, content)
	assert.Len(t, content.Parts, expected)
}

func assertTextPart(t *testing.T, part *genai.Part, expected string) {
	t.Helper()
	require.NotNil(t, part)
	assert.Equal(t, expected, part.Text)
}

func assertInlineDataPart(t *testing.T, part *genai.Part, expectedMimeType string, expectedData []byte) {
	t.Helper()
	require.NotNil(t, part)
	require.NotNil(t, part.InlineData)
	assert.Equal(t, expectedMimeType, part.InlineData.MIMEType)
	assert.Equal(t, expectedData, part.InlineData.Data)
}
