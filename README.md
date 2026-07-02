# Gemini

The Gemini provider for `facades.AI()` of Goravel.

## Install

Run the command below in your project to install the package automatically:

```bash
./artisan package:install github.com/goravel/gemini
```

This registers the service provider and updates `config/ai.go` so `ai.providers.gemini.via` resolves through `geminifacades.Gemini("gemini")`.

Or check [the setup file](./setup/setup.go) to install the package manually.

## Custom Failover

The provider marks these Gemini HTTP errors as failoverable by default:

| Status | Reason |
|--------|--------|
| `429 Too Many Requests` | `rate_limited` |
| `402 Payment Required` | `insufficient_credits` |
| `503 Service Unavailable` | `provider_overloaded` |

Configure `failover` rules to add Gemini-specific error message mappings. Plain strings use substring matching, and slash-delimited strings use Go regular expressions.

```go
"gemini": map[string]any{
	"key": config.Env("GEMINI_API_KEY", ""),
	"failover": map[string][]string{
		"context_length_exceeded": {
			"maximum context length",
			"/(?i)context.*length/",
		},
	},
	"via": func() (ai.Provider, error) {
		return geminifacades.Gemini("gemini")
	},
}
```

## Supported capabilities

- Text prompting
- Streaming responses
- Tool calling
- Prompt attachments
- Provider-managed files via the Gemini files API
- Image generation
- Audio generation
- Transcription

## Not supported

- Attachment-based image editing on the Gemini Developer API
- Structured speaker segments in transcription responses

Gemini's Go SDK supports multimodal prompting, file uploads, image generation, and audio output, but image editing is only available through Vertex-specific APIs. This driver also treats diarization as prompt guidance only, so transcripts do not currently return segmented speaker metadata.

## Testing

Run the command below to run all tests:

```bash
go test ./...
```

Run the live Gemini smoke test with a real API key:

```bash
GEMINI_API_KEY=your-key go test -run '^TestProviderPromptIntegration$' -v ./...
```

The smoke test skips automatically when `GEMINI_API_KEY` is not set.
