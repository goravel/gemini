package facades

import (
	"context"
	"testing"

	contractsai "github.com/goravel/framework/contracts/ai"
	mocksfoundation "github.com/goravel/framework/mocks/foundation"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/goravel/gemini"
)

func TestGeminiReturnsErrorForUnexpectedBindingType(t *testing.T) {
	app := mocksfoundation.NewApplication(t)
	gemini.App = app
	t.Cleanup(func() {
		gemini.App = nil
	})

	app.EXPECT().MakeWith(gemini.Binding, map[string]any{"provider": "gemini"}).Return("not-a-provider", nil).Once()

	provider, err := Gemini("gemini")

	require.Error(t, err)
	assert.Nil(t, provider)
	assert.Equal(t, "resolved gemini binding is string, not ai.Provider", err.Error())
}

func TestGeminiReturnsProviderFromBinding(t *testing.T) {
	app := mocksfoundation.NewApplication(t)
	gemini.App = app
	t.Cleanup(func() {
		gemini.App = nil
	})

	expected := &testProvider{}
	app.EXPECT().MakeWith(gemini.Binding, map[string]any{"provider": "gemini"}).Return(contractsai.Provider(expected), nil).Once()

	provider, err := Gemini("gemini")

	require.NoError(t, err)
	assert.Same(t, expected, provider)
}

type testProvider struct{}

func (t *testProvider) Prompt(_ context.Context, _ contractsai.AgentPrompt) (contractsai.AgentResponse, error) {
	return nil, nil
}

func (t *testProvider) Stream(_ context.Context, _ contractsai.AgentPrompt) (contractsai.StreamableAgentResponse, error) {
	return nil, nil
}
