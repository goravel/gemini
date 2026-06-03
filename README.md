# Gemini

The Gemini provider for `facades.AI()` of Goravel.

## Version

| goravel/gemini | goravel/framework |
|----------------|-------------------|
| v1.17.x        | v1.17.x           |

## Install

Run the command below in your project to install the package automatically:

```bash
./artisan package:install github.com/goravel/gemini
```

This registers the service provider and updates `config/ai.go` so `ai.providers.gemini.via` resolves through `geminifacades.Gemini("gemini")`.

Or check [the setup file](./setup/setup.go) to install the package manually.

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
