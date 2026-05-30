package gemini

import (
	"bytes"
	"context"
	"testing"

	frameworkai "github.com/goravel/framework/ai"
	contractsai "github.com/goravel/framework/contracts/ai"
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
	if err != nil {
		t.Fatalf("buildGenerateContentRequest returned error: %v", err)
	}
	if len(contents) != 3 {
		t.Fatalf("expected 3 content items, got %d", len(contents))
	}

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
	if err != nil {
		t.Fatalf("buildGenerateContentRequest returned error: %v", err)
	}
	if len(contents) != 4 {
		t.Fatalf("expected 4 content items, got %d", len(contents))
	}

	assertContentRole(t, contents[1], string(genai.RoleModel))
	assertPartCount(t, contents[1], 1)
	functionCall := contents[1].Parts[0].FunctionCall
	if functionCall == nil {
		t.Fatalf("expected assistant function call part")
	}
	if functionCall.ID != "call-1" {
		t.Fatalf("expected function call id call-1, got %q", functionCall.ID)
	}
	if functionCall.Name != "lookup_weather" {
		t.Fatalf("expected function call name lookup_weather, got %q", functionCall.Name)
	}
	if got := functionCall.Args["city"]; got != "London" {
		t.Fatalf("expected parsed raw args city=London, got %#v", got)
	}

	assertContentRole(t, contents[2], string(genai.RoleUser))
	assertPartCount(t, contents[2], 1)
	functionResponse := contents[2].Parts[0].FunctionResponse
	if functionResponse == nil {
		t.Fatalf("expected tool result function response part")
	}
	if functionResponse.ID != "call-1" {
		t.Fatalf("expected function response id call-1, got %q", functionResponse.ID)
	}
	if functionResponse.Name != "lookup_weather" {
		t.Fatalf("expected function response name lookup_weather, got %q", functionResponse.Name)
	}
	if got := functionResponse.Response["output"]; got != "sunny" {
		t.Fatalf("expected tool output sunny, got %#v", got)
	}

	assertContentRole(t, contents[3], string(genai.RoleUser))
	assertPartCount(t, contents[3], 1)
	assertTextPart(t, contents[3].Parts[0], "thanks")
}

func TestBuildAttachmentPartUsesStoredGeminiFileURI(t *testing.T) {
	provider := &Provider{}
	file := &genai.File{
		Name:     "files/example-file",
		URI:      "https://example.invalid/files/example-file",
		MIMEType: "image/png",
	}

	part, err := provider.buildAttachmentPart(context.Background(), frameworkai.ImageFromID(encodeFileID(file)))
	if err != nil {
		t.Fatalf("buildAttachmentPart returned error: %v", err)
	}
	if part.FileData == nil {
		t.Fatalf("expected file-data part for stored Gemini file")
	}
	if part.FileData.FileURI != file.URI {
		t.Fatalf("expected file uri %q, got %q", file.URI, part.FileData.FileURI)
	}
	if part.FileData.MIMEType != file.MIMEType {
		t.Fatalf("expected file mime type %q, got %q", file.MIMEType, part.FileData.MIMEType)
	}
	if part.InlineData != nil {
		t.Fatalf("expected stored file part to avoid inline bytes")
	}
}

func TestEncodeDecodeFileIDRoundTrip(t *testing.T) {
	file := &genai.File{
		Name:     "files/sample",
		URI:      "https://example.invalid/files/sample",
		MIMEType: "application/pdf",
	}

	encoded := encodeFileID(file)
	name, uri, mimeType, err := decodeFileID(encoded)
	if err != nil {
		t.Fatalf("decodeFileID returned error: %v", err)
	}
	if name != file.Name || uri != file.URI || mimeType != file.MIMEType {
		t.Fatalf("decoded file id mismatch: got (%q, %q, %q)", name, uri, mimeType)
	}
}

func assertContentRole(t *testing.T, content *genai.Content, expected string) {
	t.Helper()
	if content == nil {
		t.Fatalf("expected content, got nil")
	}
	if content.Role != expected {
		t.Fatalf("expected role %q, got %q", expected, content.Role)
	}
}

func assertPartCount(t *testing.T, content *genai.Content, expected int) {
	t.Helper()
	if content == nil {
		t.Fatalf("expected content, got nil")
	}
	if len(content.Parts) != expected {
		t.Fatalf("expected %d parts, got %d", expected, len(content.Parts))
	}
}

func assertTextPart(t *testing.T, part *genai.Part, expected string) {
	t.Helper()
	if part == nil {
		t.Fatalf("expected part, got nil")
	}
	if part.Text != expected {
		t.Fatalf("expected text %q, got %q", expected, part.Text)
	}
}

func assertInlineDataPart(t *testing.T, part *genai.Part, expectedMimeType string, expectedData []byte) {
	t.Helper()
	if part == nil || part.InlineData == nil {
		t.Fatalf("expected inline data part")
	}
	if part.InlineData.MIMEType != expectedMimeType {
		t.Fatalf("expected inline mime type %q, got %q", expectedMimeType, part.InlineData.MIMEType)
	}
	if !bytes.Equal(part.InlineData.Data, expectedData) {
		t.Fatalf("expected inline data %q, got %q", string(expectedData), string(part.InlineData.Data))
	}
}
